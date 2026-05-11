"""
room_wall_split.py - Split exterior walls by adjacent interior room.

Standalone pipeline. For each exterior wall, the wall is broken into
sub-segments at every interior-wall T-junction, and each sub-segment is
labeled with the global ID of the room it bounds (the room on the
interior side of that portion of the wall).

Example: a 24 ft north wall that runs across a Kitchen, a Hall and a
Bedroom comes out as three sub-segments tagged R1 / R2 / R3 with their
own pixel coords and real-world lengths.

Pipeline
--------
  1. Rasterize the PDF (default 300 DPI).
  2. Recover the building footprint mask + contour (reuses preprocess.py).
  3. Extract exterior wall segments from the simplified polygon.
  4. Detect candidate interior wall segments via HoughLinesP, filtered to
     those that T-junction into the exterior boundary.
  5. Build a global room map: fill the footprint, peel the boundary band,
     burn the interior walls in as cuts, run connected components. Each
     component above the area threshold is a room (R1, R2, ...).
  6. For each exterior wall, find the T-junction split points, then for
     each sub-segment step a few pixels inward from its midpoint to
     read the room ID off the room map.
  7. Emit JSON: { rooms: [...], walls: [sub_segments...] }.

Usage
-----
  python room_wall_split.py <plan.pdf> --scale "1in=16ft"
       [--dpi 300] [--page 1] [--tol 15] [--min-room 8] [--output room_walls.json]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from preprocess import (
    _extract_wall_lines,
    angle_to_facing,
    find_footprint,
    find_footprint_contour,
    pdf_to_images,
    preprocess,
    simplify_polygon,
    wall_angle_deg,
)
from scale_parse import parse_scale
from extract_wall_segments_class import extract_wall_segments


# ── Geometry helpers ───────────────────────────────────────────────────────

def seg_angle_deg(x1, y1, x2, y2):
    """Angle of segment in [0, 180)."""
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


def find_interior_segments(all_segs, exterior_walls, footprint_contour, near_tol):
    """
    Keep only Hough segments that look like interior walls:
      - midpoint inside the footprint
      - not tracing the exterior boundary (parallel AND close on both ends)
      - at least one endpoint T-junctions into an exterior wall
    """
    interior = []
    for seg in all_segs:
        x1, y1, x2, y2 = seg
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if not point_in_contour(mx, my, footprint_contour):
            continue

        seg_ang = seg_angle_deg(x1, y1, x2, y2)

        tracing = False
        for ew in exterior_walls:
            ew_ang = seg_angle_deg(*ew)
            if angle_diff(seg_ang, ew_ang) < 15:
                _, d1 = project_point_onto_segment(x1, y1, *ew)
                _, d2 = project_point_onto_segment(x2, y2, *ew)
                if d1 < near_tol and d2 < near_tol:
                    tracing = True
                    break
        if tracing:
            continue

        def near_any_exterior(px, py):
            return any(
                project_point_onto_segment(px, py, *ew)[1] < near_tol
                for ew in exterior_walls
            )

        if near_any_exterior(x1, y1) or near_any_exterior(x2, y2):
            interior.append(seg)
    return interior


# ── Room cell detection ────────────────────────────────────────────────────

def build_room_label_map(
    footprint_contour,
    interior_segments,
    wall_mask,
    image_shape,
    wall_thickness_px,
    min_room_area_px,
    endpoint_extend_px,
    close_kernel_px,
    debug_dir=None,
):
    """
    Partition the building interior into discrete room cells.

    Steps:
      1. Fill the footprint polygon and erode by a full wall thickness so
         the perimeter wall band itself never participates in a room.
      2. Build a "cut layer" that marks every wall pixel:
            - start from wall_mask (dense, captures all detected wall pixels)
            - burn every interior Hough segment in too, with each line
              extended by endpoint_extend_px past its endpoints so
              T-junctions seal cleanly
            - morphological close with close_kernel_px to bridge
              pixel-level gaps (and small doorway openings if close
              kernel is set large enough).
      3. room_mask = interior_mask AND NOT cut_layer.
      4. Open with a small kernel to dissolve sliver components left by
         the subtraction.
      5. Connected components. Each component above min_room_area_px
         becomes a room (R1, R2, ...); smaller specks → label 0.

    Returns
    -------
    labels : np.ndarray (H, W) int32 — 0 means not-a-room
    rooms  : list[dict] with id, area_px, centroid_px, bbox_px
    """
    h, w = image_shape[:2]

    interior_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(
        interior_mask, [footprint_contour], -1, 255, thickness=cv2.FILLED
    )
    erode_size = max(wall_thickness_px * 2, 3)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
    interior_mask = cv2.erode(interior_mask, erode_k, iterations=1)

    cut_layer = wall_mask.copy()
    cut_layer = (cut_layer > 0).astype(np.uint8) * 255

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
        ck = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel_px, close_kernel_px)
        )
        cut_layer = cv2.morphologyEx(cut_layer, cv2.MORPH_CLOSE, ck)

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
        if area < min_room_area_px:
            continue
        remap[lbl] = next_id
        cx, cy = centroids[lbl]
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        rw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        rh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        rooms.append({
            "id": f"R{next_id}",
            "area_px": area,
            "centroid_px": [int(round(cx)), int(round(cy))],
            "bbox_px": [x, y, x + rw, y + rh],
        })
        next_id += 1

    relabeled = remap[labels].astype(np.int32)
    return relabeled, rooms


def inward_normal(x1, y1, x2, y2, footprint_contour, probe_px=8.0):
    """Unit perpendicular pointing into the footprint, picked empirically."""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if point_in_contour(mx + probe_px * nx, my + probe_px * ny, footprint_contour):
        return nx, ny
    return -nx, -ny


def sample_room_for_segment(
    x1, y1, x2, y2, room_labels, footprint_contour, probe_offsets_px,
):
    """
    Vote which room ID this sub-segment bounds.

    Walk to five anchor points along the segment (t = 0.15, 0.30, 0.5,
    0.70, 0.85), step inward past the wall by each probe offset until a
    non-zero label is hit, then tally. Returns (room_label_int, hits,
    total_anchors). 0/None means no room could be identified.
    """
    h, w = room_labels.shape
    nx, ny = inward_normal(x1, y1, x2, y2, footprint_contour,
                           probe_px=max(probe_offsets_px))

    sample_ts = [0.15, 0.30, 0.5, 0.70, 0.85]
    hits = {}
    for t in sample_ts:
        bx = x1 + t * (x2 - x1)
        by = y1 + t * (y2 - y1)
        for d in probe_offsets_px:
            px = int(round(bx + d * nx))
            py = int(round(by + d * ny))
            if 0 <= px < w and 0 <= py < h:
                lbl = int(room_labels[py, px])
                if lbl > 0:
                    hits[lbl] = hits.get(lbl, 0) + 1
                    break

    if not hits:
        return None, 0, len(sample_ts)
    best = max(hits.items(), key=lambda kv: kv[1])
    return best[0], best[1], len(sample_ts)


# ── Wall splitting (room-walk approach) ────────────────────────────────────

def walk_wall_and_split_by_room(
    ext_wall, room_labels, footprint_contour,
    probe_offsets_px, min_segment_px,
):
    """
    Walk along the exterior wall sampling the room map inward at fine
    intervals. Group contiguous same-room samples into runs. Each run
    becomes one sub-segment, labeled with the room it bounds.

    This replaces the T-junction approach: instead of trying to detect
    where an interior wall hits the exterior, we read the answer directly
    off the room map we already built.

    Returns
    -------
    list of (t_start, t_end, room_label_int)   — room_label_int = 0 means
    "no room identified" (the inward probe found only label-0 pixels).
    """
    x1, y1, x2, y2 = ext_wall
    wall_len_px = math.hypot(x2 - x1, y2 - y1)
    if wall_len_px < 1:
        return [(0.0, 1.0, 0)]

    nx, ny = inward_normal(x1, y1, x2, y2, footprint_contour,
                           probe_px=max(probe_offsets_px))
    h, w = room_labels.shape

    # Sample every ~3 px along the wall — fine enough to catch narrow rooms
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

    # Single-sample outliers: replace if both neighbors agree on a different label
    smoothed = labels_along[:]
    for i in range(1, len(labels_along) - 1):
        if (smoothed[i] != smoothed[i-1]
                and smoothed[i] != smoothed[i+1]
                and smoothed[i-1] == smoothed[i+1]):
            smoothed[i] = smoothed[i-1]

    # Group into runs
    runs = []
    cur_lbl = smoothed[0]
    cur_start = 0
    for i in range(1, len(smoothed)):
        if smoothed[i] != cur_lbl:
            runs.append((cur_start / n_samples, i / n_samples, cur_lbl))
            cur_lbl = smoothed[i]
            cur_start = i
    runs.append((cur_start / n_samples, 1.0, cur_lbl))

    # Merge runs shorter than min_segment_px into a neighbor
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
):
    """
    Convert a list of (t0, t1, room_label_int) runs into wall-sub-segment
    dicts ready for JSON output.
    """
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
        facing = angle_to_facing(angle)
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
        })
    return sub_segments


# ── Pipeline ───────────────────────────────────────────────────────────────

def analyze_page(
    image, scale_str, dpi,
    near_tol=15, min_room_ft2=25.0,
    doorway_close_ft=0.0, min_segment_ft=4.4,
    debug_dir=None,
):
    cal = parse_scale(scale_str, dpi, output_unit="ft")
    px_per_unit = cal["px_per_unit"]
    unit_label = cal["unit_label"]

    t0 = time.time()
    wall_mask = _extract_wall_lines(image)
    binary = preprocess(image)
    foot_mask = find_footprint(binary)
    if foot_mask is None:
        return {"error": "No building footprint found"}
    contour = find_footprint_contour(foot_mask)
    if contour is None:
        return {"error": "No building footprint found"}
    polygon = simplify_polygon(contour)
    print(f"  [pipeline] footprint: {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    exterior_segs = extract_wall_segments(polygon)
    print(
        f"  [pipeline] exterior segs: {time.time()-t0:.1f}s "
        f"({len(exterior_segs)} walls)", file=sys.stderr,
    )

    xs = polygon[:, 0]
    ys = polygon[:, 1]
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    t0 = time.time()
    all_hough = detect_hough_segments(wall_mask, bbox)
    interior_segs = find_interior_segments(
        all_hough, exterior_segs, contour, near_tol,
    )
    print(
        f"  [pipeline] interior segs: {time.time()-t0:.1f}s "
        f"({len(all_hough)} hough, {len(interior_segs)} kept)",
        file=sys.stderr,
    )

    # Wall thickness ~ 6" in real units; floor at 6 px so morphology has bite
    wall_thickness_px = max(int(round(0.5 * px_per_unit)), 6)
    min_room_area_px = int(round(min_room_ft2 * px_per_unit ** 2))
    # Extend Hough endpoints ~1 ft to seal T-junctions
    endpoint_extend_px = max(int(round(1.0 * px_per_unit)), 8)
    # Close kernel: closes gaps up to doorway_close_ft wide. 0 → just pixel
    # noise closing (3 px). Set this near a typical doorway width (~3 ft)
    # to seal interior walls across doors; risks bridging unrelated walls.
    close_kernel_px = max(int(round(doorway_close_ft * px_per_unit)), 3)

    t0 = time.time()
    room_labels, rooms = build_room_label_map(
        contour, interior_segs, wall_mask, image.shape,
        wall_thickness_px, min_room_area_px,
        endpoint_extend_px, close_kernel_px,
        debug_dir=debug_dir,
    )
    print(
        f"  [pipeline] room cells: {time.time()-t0:.1f}s "
        f"({len(rooms)} rooms)", file=sys.stderr,
    )

    # Probe offsets: step past the wall band, then a bit further for safety
    probe_offsets_px = [
        int(round(wall_thickness_px * f)) for f in (1.5, 2.5, 4.0, 6.0)
    ]

    min_seg_px = min_segment_ft * px_per_unit
    all_sub_segments = []
    for i, ext in enumerate(exterior_segs):
        x1, y1, x2, y2 = ext
        wall_ang = wall_angle_deg(x1, y1, x2, y2)
        facing = angle_to_facing(wall_ang)
        runs = walk_wall_and_split_by_room(
            ext, room_labels, contour,
            probe_offsets_px, min_segment_px=min_seg_px,
        )
        subs = runs_to_sub_segments(
            f"w{i+1}", f"{facing} Wall {i+1}", ext, runs,
            px_per_unit, unit_label,
        )
        all_sub_segments.extend(subs)

    for r in rooms:
        area_real = r["area_px"] / (px_per_unit ** 2)
        r["area"] = f"{area_real:.1f} {unit_label}²"
        r["area_raw"] = round(area_real, 2)

    h, w = image.shape[:2]
    return {
        "detected_scale": scale_str,
        "units": "metric" if unit_label == "m" else "imperial",
        "image_size_px": [w, h],
        "footprint_bbox_px": bbox,
        "px_per_ft": round(px_per_unit, 2),
        "rooms": rooms,
        "walls": all_sub_segments,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect rooms and split exterior walls by adjacent room.",
    )
    parser.add_argument("pdf", help="Path to the architectural plan PDF")
    parser.add_argument("--scale", required=True,
                        help='Drawing scale, e.g. "1in=16ft"')
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--tol", type=int, default=15,
                        help="T-junction near-miss tolerance in pixels")
    parser.add_argument("--min-segment", type=float, default=4.4,
                        help="Sub-segments shorter than this many feet are "
                             "merged into a neighbor (default: 4.4)")
    parser.add_argument("--doorway-close", type=float, default=0.0,
                        help="Close gaps up to this many feet in the cut layer "
                             "(useful to seal walls across doorways; default 0 "
                             "= just pixel-noise closing). Try 2.5 first.")
    parser.add_argument("--debug", action="store_true",
                        help="Dump room/cut/interior masks to ./debug_rooms/")
    parser.add_argument("--output", default="room_walls.json")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    images = pdf_to_images(str(pdf_path), dpi=args.dpi)
    idx = args.page - 1
    if idx < 0 or idx >= len(images):
        print(f"Error: page {args.page} out of range "
              f"(PDF has {len(images)} pages)", file=sys.stderr)
        sys.exit(1)

    result = analyze_page(
        images[idx], args.scale, args.dpi,
        near_tol=args.tol,
        doorway_close_ft=args.doorway_close,
        min_segment_ft=args.min_segment,
        debug_dir="debug_rooms" if args.debug else None,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
