"""
room_wall_split.py — Split exterior walls into per-room sub-segments.

Used by preprocess.analyze_page() after polygon exterior walls are snapped.
Builds a geometric room map (connected components inside the footprint),
walks each exterior wall inward to read adjacent room IDs, and emits
sub-segments tagged with room_id (R1, R2, …).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Geometry helpers ───────────────────────────────────────────────────────

def seg_angle_deg(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def project_point_onto_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return 0.0, math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return t, math.hypot(px - proj_x, py - proj_y)


def point_in_contour(px, py, contour):
    return cv2.pointPolygonTest(
        contour.reshape(-1, 1, 2).astype(np.float32),
        (float(px), float(py)),
        measureDist=False,
    ) >= 0


# ── Interior wall detection ────────────────────────────────────────────────

def detect_hough_segments(wall_mask, bbox, margin=30):
    """Run HoughLinesP on the wall mask cropped to the footprint bbox."""
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(wall_mask.shape[1], x_max + margin)
    y_max = min(wall_mask.shape[0], y_max + margin)
    crop = wall_mask[y_min:y_max, x_min:x_max]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    crop = cv2.dilate(crop, kernel, iterations=1)
    lines = cv2.HoughLinesP(
        crop, rho=1, theta=np.pi / 180,
        threshold=30, minLineLength=15, maxLineGap=8,
    )
    if lines is None:
        return []
    return [
        (int(l[0][0] + x_min), int(l[0][1] + y_min),
         int(l[0][2] + x_min), int(l[0][3] + y_min))
        for l in lines
    ]


def _clamped_distance_to_segment(px, py, x1, y1, x2, y2):
    """Distance from point to the segment itself (projection clamped to it)."""
    t, d = project_point_onto_segment(px, py, x1, y1, x2, y2)
    if t < 0.0:
        return math.hypot(px - x1, py - y1)
    if t > 1.0:
        return math.hypot(px - x2, py - y2)
    return d


def segment_traces_exterior(seg, exterior_walls, near_tol):
    """True if seg is parallel to and lies on an exterior wall (both endpoints near it).

    Distances are measured to the exterior segment itself, not its infinite
    line: a collinear wall far beyond the exterior segment's end (a different
    step of the perimeter, or an interior wall continuing past it) does not
    trace it.
    """
    x1, y1, x2, y2 = seg
    seg_ang = seg_angle_deg(x1, y1, x2, y2)
    for ew in exterior_walls:
        ew_ang = seg_angle_deg(*ew)
        if angle_diff(seg_ang, ew_ang) < 15:
            d1 = _clamped_distance_to_segment(x1, y1, *ew)
            d2 = _clamped_distance_to_segment(x2, y2, *ew)
            if d1 < near_tol and d2 < near_tol:
                return True
    return False


def find_interior_segments(all_segs, exterior_walls, footprint_contour, near_tol):
    """Keep Hough segments that T-junction into the exterior boundary."""
    interior = []
    for seg in all_segs:
        x1, y1, x2, y2 = seg
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if not point_in_contour(mx, my, footprint_contour):
            continue

        if segment_traces_exterior(seg, exterior_walls, near_tol):
            continue

        def near_any_exterior(px, py):
            return any(
                project_point_onto_segment(px, py, *ew)[1] < near_tol
                for ew in exterior_walls
            )

        if near_any_exterior(x1, y1) or near_any_exterior(x2, y2):
            interior.append(seg)
    return interior


# ── Room cell detection ──────────────────────────────────────────────────────

# Cells whose bbox aspect ratio is at least this are corridor-like and use the
# lower corridor area floor instead of the room floor. The doorway close fills
# any interior span narrower than doorway_close_ft (2.5 ft), so a sub-25 ft²
# cell is at least ~3 ft wide and tops out near aspect 2.8 — a threshold of 3
# would never fire on real geometry.
CORRIDOR_ASPECT = 2.5

# Max separation (px) between a sub-floor fragment and a kept room for the
# fragment to be reabsorbed — the scale of the 3x3 opening artifact, well
# below any real wall band (cut lines are >= 6 px thick).
ORPHAN_MERGE_PX = 3


def _contour_ink_gap(footprint_contour, wall_mask):
    """Median distance (px) from the footprint contour to the nearest ink.

    The contour is traced on a dilated/closed mask, so it rides a roughly
    constant band outside the real wall strokes. The median over the outline
    is robust to door/window gaps and annotation bulges, where the nearest
    ink is far away.
    """
    h, w = wall_mask.shape[:2]
    cx, cy, cw, ch = cv2.boundingRect(footprint_contour)
    pad = 50
    x_lo, y_lo = max(0, cx - pad), max(0, cy - pad)
    x_hi, y_hi = min(w, cx + cw + pad), min(h, cy + ch + pad)
    crop_ink = wall_mask[y_lo:y_hi, x_lo:x_hi]
    if not crop_ink.any():
        return 0.0
    ink_dist = cv2.distanceTransform(
        (crop_ink == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    outline = np.zeros(crop_ink.shape, dtype=np.uint8)
    shifted = footprint_contour.reshape(-1, 2) - np.array([x_lo, y_lo])
    cv2.drawContours(
        outline, [shifted.reshape(-1, 1, 2).astype(np.int32)], -1, 255, 1
    )
    oys, oxs = np.nonzero(outline)
    if len(oys) == 0:
        return 0.0
    return float(np.median(ink_dist[oys, oxs]))


def build_room_label_map(
    footprint_contour,
    interior_segments,
    wall_mask,
    image_shape,
    wall_thickness_px,
    min_room_area_px,
    endpoint_extend_px,
    close_kernel_px,
    min_corridor_area_px=None,
    min_cell_width_px=0,
    debug_dir=None,
):
    h, w = image_shape[:2]
    if min_corridor_area_px is None:
        min_corridor_area_px = min_room_area_px

    interior_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(
        interior_mask, [footprint_contour], -1, 255, thickness=cv2.FILLED
    )
    # The erosion exists to remove the band between the morphologically
    # inflated footprint contour and the real wall ink. That inflation is a
    # fixed pixel amount (downscale dilation), not a multiple of wall
    # thickness — the old 2x-thickness erosion taxed every room's perimeter.
    # Measure the actual contour-to-ink gap and erode just past it, capped at
    # the legacy 2x thickness so a pathological mask can never do worse.
    inflation = _contour_ink_gap(footprint_contour, wall_mask)
    erode_size = max(wall_thickness_px, int(math.ceil(inflation)) + 2, 3)
    erode_size = min(erode_size, max(2 * wall_thickness_px, 12))
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
    interior_mask = cv2.erode(interior_mask, erode_k, iterations=1)

    cut_layer = (wall_mask > 0).astype(np.uint8) * 255

    for seg in interior_segments:
        x1, y1, x2, y2 = seg
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        ex1 = int(round(x1 - endpoint_extend_px * ux))
        ey1 = int(round(y1 - endpoint_extend_px * uy))
        ex2 = int(round(x2 + endpoint_extend_px * ux))
        ey2 = int(round(y2 + endpoint_extend_px * uy))
        cv2.line(cut_layer, (ex1, ey1), (ex2, ey2), 255,
                 thickness=wall_thickness_px)

    if close_kernel_px > 1:
        # Directional close: seal doorway gaps along each wall's own axis and
        # fill the band between a wall's paired strokes, without absorbing
        # interior spaces (corridors) narrower than the kernel in both axes
        # the way a square kernel does.
        ck_h = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_px, 1))
        ck_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_kernel_px))
        cut_layer = cv2.bitwise_or(
            cv2.morphologyEx(cut_layer, cv2.MORPH_CLOSE, ck_h),
            cv2.morphologyEx(cut_layer, cv2.MORPH_CLOSE, ck_v),
        )

    room_mask = cv2.bitwise_and(interior_mask, cv2.bitwise_not(cut_layer))

    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, open_k)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "interior_mask.png"), interior_mask)
        cv2.imwrite(str(debug_dir / "cut_layer.png"), cut_layer)
        cv2.imwrite(str(debug_dir / "room_mask.png"), room_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        room_mask
    )

    remap = np.zeros(num_labels, dtype=np.int32)
    rooms = []
    next_id = 1
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        rw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        rh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        # No genuine interior space is narrower than the doorway close kernel
        # (the close would have sealed it); cells below that width are
        # boundary/cavity slivers regardless of their area.
        if min(rw, rh) < min_cell_width_px:
            continue
        # Elongated cells (corridors, closets) use a lower area floor: a 25 ft²
        # compact floor rejects most real corridors outright.
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        floor_px = (
            min_corridor_area_px if aspect >= CORRIDOR_ASPECT
            else min_room_area_px
        )
        if area < floor_px:
            continue
        remap[lbl] = next_id
        cx, cy = centroids[lbl]
        rooms.append({
            "id": f"R{next_id}",
            "area_px": area,
            "centroid_px": [int(round(cx)), int(round(cy))],
            "bbox_px": [x, y, x + rw, y + rh],
        })
        next_id += 1

    relabeled = remap[labels].astype(np.int32)
    _reassign_orphan_fragments(labels, relabeled, remap, stats, rooms)
    return relabeled, rooms


def _reassign_orphan_fragments(labels, relabeled, remap, stats, rooms):
    """Merge sub-floor fragments back into the single kept room they border.

    The 3x3 opening (and cut-line crossings) can shear small slivers off a
    room cell; those pixels are real interior area lost from the coverage
    numerator. A fragment separated from exactly one kept room by no more
    than ORPHAN_MERGE_PX is reabsorbed into it (label, area, bbox). Fragments
    bordering zero or 2+ kept rooms are left dropped — a sliver must never
    bridge two rooms, and isolated specks stay out.
    """
    if not rooms:
        return
    h, w = labels.shape
    k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * ORPHAN_MERGE_PX + 1, 2 * ORPHAN_MERGE_PX + 1)
    )
    for lbl in range(1, len(remap)):
        if remap[lbl] != 0:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        x0 = max(0, x - ORPHAN_MERGE_PX)
        y0 = max(0, y - ORPHAN_MERGE_PX)
        x1 = min(w, x + bw + ORPHAN_MERGE_PX)
        y1 = min(h, y + bh + ORPHAN_MERGE_PX)
        comp = (labels[y0:y1, x0:x1] == lbl).astype(np.uint8)
        halo = cv2.dilate(comp, k) > 0
        view = relabeled[y0:y1, x0:x1]
        neighbors = np.unique(view[halo])
        neighbors = neighbors[neighbors > 0]
        if neighbors.size != 1:
            continue
        target = int(neighbors[0])
        view[comp > 0] = target
        room = rooms[target - 1]
        room["area_px"] += int(stats[lbl, cv2.CC_STAT_AREA])
        bx0, by0, bx1, by1 = room["bbox_px"]
        room["bbox_px"] = [
            min(bx0, x), min(by0, y), max(bx1, x + bw), max(by1, y + bh),
        ]


def exterior_axis_envelope(
    exterior_segments: list[tuple],
) -> Optional[tuple[int, int, int, int]]:
    """(x_lo, x_hi, y_lo, y_hi) spanned by the snapped exterior wall axes.

    Uses the perpendicular axis position of each segment (midpoint of the
    perpendicular coordinates), not the endpoints: endpoints retain contour
    bump overshoot past corners, axes are pair-validated onto wall ink.
    Returns None when either orientation is missing.
    """
    h_axes, v_axes = [], []
    for x1, y1, x2, y2 in exterior_segments:
        if abs(x2 - x1) >= abs(y2 - y1):
            h_axes.append((y1 + y2) // 2)
        else:
            v_axes.append((x1 + x2) // 2)
    if not h_axes or not v_axes:
        return None
    return min(v_axes), max(v_axes), min(h_axes), max(h_axes)


def clamp_segments_to_envelope(
    segments: list[tuple],
    pad_px: int,
) -> list[tuple]:
    """Clamp segment spans to the exterior axis envelope of the set itself.

    Footprint-polygon edges can run past the building along an annotation
    strip merged into the footprint (e.g. down the side of a dimension
    string). The snapped axes bound the building; no wall extends beyond the
    outermost perpendicular axis by more than half a wall thickness, so spans
    are clamped to the envelope expanded by pad_px. Degenerate results are
    left untouched.
    """
    env = exterior_axis_envelope(segments)
    if env is None:
        return segments
    x_lo, x_hi = env[0] - pad_px, env[1] + pad_px
    y_lo, y_hi = env[2] - pad_px, env[3] + pad_px
    out = []
    for x1, y1, x2, y2 in segments:
        cx1 = min(max(x1, x_lo), x_hi)
        cx2 = min(max(x2, x_lo), x_hi)
        cy1 = min(max(y1, y_lo), y_hi)
        cy2 = min(max(y2, y_lo), y_hi)
        if (cx1, cy1) == (cx2, cy2):
            out.append((x1, y1, x2, y2))
        else:
            out.append((cx1, cy1, cx2, cy2))
    return out


def drop_segments_outside_exterior(
    segments: list[tuple],
    exterior_segments: list[tuple],
    margin_px: int = 0,
) -> list[tuple]:
    """Remove candidate interior segments lying outside the exterior envelope.

    Dimension strings and other annotation strokes sit outside the building's
    exterior walls. Once the exterior segments are pair-validated onto true
    wall ink, anything whose midpoint falls outside their axis envelope
    (expanded by margin_px) cannot be an interior wall.
    """
    env = exterior_axis_envelope(exterior_segments)
    if env is None:
        return segments
    x_lo, x_hi = env[0] - margin_px, env[1] + margin_px
    y_lo, y_hi = env[2] - margin_px, env[3] + margin_px
    kept = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if x_lo <= mx <= x_hi and y_lo <= my <= y_hi:
            kept.append(seg)
    return kept


def drop_rooms_outside_exterior(
    rooms: list[dict],
    room_labels: np.ndarray,
    exterior_segments: list[tuple],
    margin_px: int,
) -> tuple[list[dict], np.ndarray]:
    """Remove phantom room cells lying outside the snapped exterior walls.

    The footprint contour is traced on a morphologically inflated mask, so it
    can bulge past the real exterior walls to nearby annotation ink (dimension
    strings, leaders). The sliver between the wall and that ink then survives
    as a phantom room cell — bounded by real ink on every side, but outside
    the building.

    The snapped exterior segments are pair-validated onto true wall ink, so
    their perpendicular axis positions bound the building reliably: the
    outermost horizontal axes give the y-range, the outermost vertical axes
    the x-range. (Segment endpoints are NOT used — they retain the contour's
    bump overshoot past corners.) Cells whose centroid falls outside that
    range, shrunk inward by margin_px, are dropped and the label map is
    renumbered compactly.

    Skips filtering when either orientation is missing or the resulting box
    is degenerate (exterior detection failure) — better to keep a phantom
    than to drop real rooms.
    """
    if not rooms or not exterior_segments:
        return rooms, room_labels

    env = exterior_axis_envelope(exterior_segments)
    if env is None:
        return rooms, room_labels

    x_lo, x_hi = env[0] + margin_px, env[1] - margin_px
    y_lo, y_hi = env[2] + margin_px, env[3] - margin_px
    if (x_hi - x_lo) < 4 * margin_px or (y_hi - y_lo) < 4 * margin_px:
        return rooms, room_labels

    kept = []
    remap = np.zeros(len(rooms) + 1, dtype=np.int32)
    for idx, room in enumerate(rooms, start=1):
        cx, cy = room["centroid_px"]
        if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
            remap[idx] = len(kept) + 1
            kept.append(room)

    if len(kept) == len(rooms):
        return rooms, room_labels

    for new_idx, room in enumerate(kept, start=1):
        room["id"] = f"R{new_idx}"
    return kept, remap[room_labels]


def inward_normal(x1, y1, x2, y2, footprint_contour, probe_px=8.0):
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if point_in_contour(mx + probe_px * nx, my + probe_px * ny, footprint_contour):
        return nx, ny
    return -nx, -ny


# ── Wall splitting ───────────────────────────────────────────────────────────

def walk_wall_and_split_by_room(
    ext_wall, room_labels, footprint_contour,
    probe_offsets_px, min_segment_px,
):
    x1, y1, x2, y2 = ext_wall
    wall_len_px = math.hypot(x2 - x1, y2 - y1)
    if wall_len_px < 1:
        return [(0.0, 1.0, 0)]

    nx, ny = inward_normal(x1, y1, x2, y2, footprint_contour,
                           probe_px=max(probe_offsets_px))
    h, w = room_labels.shape

    n_samples = max(20, int(round(wall_len_px / 3.0)))
    labels_along = []
    for i in range(n_samples + 1):
        t = i / n_samples
        bx = x1 + t * (x2 - x1)
        by = y1 + t * (y2 - y1)
        lbl_found = 0
        for d in probe_offsets_px:
            px = int(round(bx + d * nx))
            py = int(round(by + d * ny))
            if 0 <= px < w and 0 <= py < h:
                lbl = int(room_labels[py, px])
                if lbl > 0:
                    lbl_found = lbl
                    break
        labels_along.append(lbl_found)

    smoothed = labels_along[:]
    for i in range(1, len(labels_along) - 1):
        if (smoothed[i] != smoothed[i - 1]
                and smoothed[i] != smoothed[i + 1]
                and smoothed[i - 1] == smoothed[i + 1]):
            smoothed[i] = smoothed[i - 1]

    runs = []
    cur_lbl = smoothed[0]
    cur_start = 0
    for i in range(1, len(smoothed)):
        if smoothed[i] != cur_lbl:
            runs.append((cur_start / n_samples, i / n_samples, cur_lbl))
            cur_lbl = smoothed[i]
            cur_start = i
    runs.append((cur_start / n_samples, 1.0, cur_lbl))

    if wall_len_px > 0:
        min_t = min_segment_px / wall_len_px
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for i in range(len(runs)):
                t0, t1, lbl = runs[i]
                if (t1 - t0) >= min_t:
                    continue
                left = i - 1 if i > 0 else None
                right = i + 1 if i < len(runs) - 1 else None
                if left is not None and right is not None:
                    len_l = runs[left][1] - runs[left][0]
                    len_r = runs[right][1] - runs[right][0]
                    merge_to = left if len_l >= len_r else right
                elif left is not None:
                    merge_to = left
                else:
                    merge_to = right
                if merge_to == left:
                    runs[left] = (runs[left][0], t1, runs[left][2])
                else:
                    runs[right] = (t0, runs[right][1], runs[right][2])
                runs.pop(i)
                changed = True
                break

    return runs


def runs_to_sub_segments(
    wall_id, name_prefix, ext_wall, runs,
    px_per_unit, unit_label,
    contour, footprint_bbox,
):
    from preprocess import assign_segment_facings, wall_angle_deg

    x1, y1, x2, y2 = ext_wall
    dx, dy = x2 - x1, y2 - y1
    sub_segments = []

    for i, (ta, tb, room_lbl) in enumerate(runs, start=1):
        sx1 = int(round(x1 + ta * dx))
        sy1 = int(round(y1 + ta * dy))
        sx2 = int(round(x1 + tb * dx))
        sy2 = int(round(y1 + tb * dy))

        px_len = math.hypot(sx2 - sx1, sy2 - sy1)
        real_len = px_len / px_per_unit
        angle = wall_angle_deg(sx1, sy1, sx2, sy2)
        facing = assign_segment_facings(
            [(sx1, sy1, sx2, sy2)], contour, footprint_bbox, px_per_unit,
        )[0]
        room_id = f"R{room_lbl}" if room_lbl > 0 else None

        sub_segments.append({
            "id": f"{wall_id}.s{i}",
            "name": f"{name_prefix} part {i}"
                    + (f" → {room_id}" if room_id else " → (no room)"),
            "facing": facing,
            "length": f"{real_len:.2f} {unit_label}",
            "length_raw": round(real_len, 2),
            "angle_deg": round(angle, 1),
            "px_coords": [sx1, sy1, sx2, sy2],
            "room_id": room_id,
            "parent_wall_id": wall_id,
            "segment_index": i,
            "segment_count": len(runs),
            "is_exterior": True,
        })
    return sub_segments


def colorize_room_labels(labels: np.ndarray) -> np.ndarray:
    """BGR visualization of connected room cells."""
    h, w = labels.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    max_lbl = int(labels.max())
    for lbl in range(1, max_lbl + 1):
        hue = int((lbl * 47) % 180)
        color = cv2.cvtColor(
            np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
        vis[labels == lbl] = color
    return vis


# ── Public API ───────────────────────────────────────────────────────────────

def split_exterior_walls_by_room(
    exterior_segments: list[tuple],
    wall_pair_mask: np.ndarray,
    contour: np.ndarray,
    footprint_bbox: list[int],
    image_shape: tuple,
    px_per_unit: float,
    unit_label: str,
    near_tol: int = 15,
    min_room_ft2: float = 25.0,
    min_corridor_ft2: float = 8.0,
    doorway_close_ft: float = 2.5,
    min_segment_ft: float = 4.0,
    interior_segments: Optional[list[tuple]] = None,
    debug_dir: Optional[str] = None,
    crop_mode: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Split snapped polygon exterior walls into per-room sub-segments.

    Returns (rooms, exterior_sub_segment_walls).
    """
    from preprocess import assign_segment_facings

    if not exterior_segments:
        return [], []

    wall_thickness_px = max(int(round(0.5 * px_per_unit)), 6)
    if crop_mode:
        min_room_ft2 = max(min_room_ft2, 30.0)
        min_corridor_ft2 = max(min_corridor_ft2, 10.0)
    min_room_area_px = int(round(min_room_ft2 * px_per_unit ** 2))
    min_corridor_area_px = int(round(min_corridor_ft2 * px_per_unit ** 2))
    endpoint_extend_px = max(int(round(1.0 * px_per_unit)), 8)
    close_kernel_px = max(int(round(doorway_close_ft * px_per_unit)), 3)

    if interior_segments is None:
        all_hough = detect_hough_segments(wall_pair_mask, footprint_bbox)
        interior_segments = find_interior_segments(
            all_hough, exterior_segments, contour, near_tol,
        )

    room_labels, rooms = build_room_label_map(
        contour, interior_segments, wall_pair_mask, image_shape,
        wall_thickness_px, min_room_area_px,
        endpoint_extend_px, close_kernel_px,
        min_corridor_area_px=min_corridor_area_px,
        min_cell_width_px=close_kernel_px,
        debug_dir=debug_dir,
    )

    envelope_margin = wall_thickness_px * (2 if crop_mode else 1)
    rooms, room_labels = drop_rooms_outside_exterior(
        rooms, room_labels, exterior_segments, margin_px=envelope_margin,
    )

    if debug_dir is not None:
        cv2.imwrite(
            str(Path(debug_dir) / "room_labels_color.png"),
            colorize_room_labels(room_labels),
        )

    probe_offsets_px = [
        int(round(wall_thickness_px * f)) for f in (1.5, 2.5, 4.0, 6.0)
    ]
    min_seg_px = min_segment_ft * px_per_unit

    all_sub_segments = []
    parent_facings = assign_segment_facings(
        exterior_segments, contour, footprint_bbox, px_per_unit,
    )
    for i, ext in enumerate(exterior_segments):
        facing = parent_facings[i]
        runs = walk_wall_and_split_by_room(
            ext, room_labels, contour,
            probe_offsets_px, min_segment_px=min_seg_px,
        )
        subs = runs_to_sub_segments(
            f"w{i + 1}", f"{facing} Wall {i + 1}", ext, runs,
            px_per_unit, unit_label, contour, footprint_bbox,
        )
        all_sub_segments.extend(subs)

    area_unit = f"{unit_label}²"
    for r in rooms:
        area_real = r["area_px"] / (px_per_unit ** 2)
        r["area"] = f"{area_real:.1f} {area_unit}"
        r["area_raw"] = round(area_real, 2)

    return rooms, all_sub_segments
