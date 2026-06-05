"""
preprocess.py — Exterior wall measurement & facing from architectural plan PDFs.

Pipeline:
  1. Rasterize PDF pages at 300 DPI
  2. Preprocess: grayscale → blur → binary threshold → morphological cleanup
  3. Find the largest external contour (building footprint)parse_scale
  4. Simplify to a clean polygon (approxPolyDP)
  5. Segment polygon edges into individual wall segments
  6. Compute wall direction (facing) relative to North = image-up
  7. Convert pixel lengths to real-world measurements via scale

Usage:
  python preprocess.py <plan.pdf> --scale "1/4in=1ft" [--dpi 300] [--page 1] [--north-up]
"""

import argparse
import json
import math
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

# pylint: disable=no-member
import cv2 #what we use for image processing + computer vision
import fitz  # PyMuPDF — no external Poppler binaries needed
import numpy as np
from scale_parse import parse_scale
from extract_wall_segments_class import extract_wall_segments


# ─── Step 1: PDF → high-res raster ──────────────────────────────────────────

DPI = 300

def pdf_to_images(path: str, dpi: int = DPI) -> list[np.ndarray]:
    doc = fitz.open(path)
    images = []
    zoom = dpi / 72  # PDF base resolution is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:  # RGBA → RGB
            img = img[:, :, :3]
        images.append(img)
    doc.close()
    return images


# ─── Step 2: Preprocessing ──────────────────────────────────────────────────


DOWNSCALE = 4  # heavy morphology runs on 1/4-res image for speed


def _find_wall_pairs(
    mask: np.ndarray,
    scan_rows: bool,
    min_gap_px: int = 2,
    max_gap_px: int = 60,
) -> np.ndarray:
    """
    Retain only pixels that belong to a double-line (wall) pair.

    Walls are drawn as two parallel strokes separated by wall thickness
    (typically 3.5"–8" at drawing scale → ~10–60 px at 300 DPI / 1:48 scale).
    Single-stroke lines — dimension strings, title block borders, grid lines,
    scale bars — have no parallel partner and are filtered out.

    scan_rows=True  → horizontal lines: look for row pairs close in Y
    scan_rows=False → vertical lines:   look for column pairs close in X
    """
    if scan_rows:
        projection = (mask > 0).any(axis=1)   # (H,): True if row has any pixel
    else:
        projection = (mask > 0).any(axis=0)   # (W,): True if col has any pixel

    active = np.where(projection)[0]
    keep = np.zeros(len(projection), dtype=bool)

    n = len(active)
    i = 0
    while i < n:
        j = i + 1
        while j < n and (active[j] - active[i]) <= max_gap_px:
            if (active[j] - active[i]) >= min_gap_px:
                keep[active[i]] = True
                keep[active[j]] = True
            j += 1
        i += 1

    if scan_rows:
        return mask * keep[:, np.newaxis].astype(np.uint8)
    else:
        return mask * keep[np.newaxis, :].astype(np.uint8)


def _build_exclusion_mask(h: int, w: int) -> np.ndarray:
    """Zones that are never building walls: header, title block, margins."""
    mask = np.zeros((h, w), np.uint8)
    mask[: int(h * 0.12), :] = 255
    mask[int(h * 0.50) :, int(w * 0.58) :] = 255
    mask[int(h * 0.82) :, :] = 255
    mask[:, : int(w * 0.05)] = 255
    mask[:, int(w * 0.96) :] = 255
    return mask


def _point_in_exclusion(px: float, py: float, w: int, h: int) -> bool:
    xf, yf = px / w, py / h
    if yf < 0.12:
        return True
    if yf > 0.82:
        return True
    if xf < 0.05:
        return True
    if xf > 0.96:
        return True
    if xf > 0.58 and yf > 0.50:
        return True
    return False


def _blank_sheet_margins(gray: np.ndarray) -> np.ndarray:
    """
    Mask title block / header without erasing the full-height right strip.
    A vertical strip at blank_right_frac was clipping real floor-plan walls.
    """
    h, w = gray.shape
    out = gray.copy()
    out[_build_exclusion_mask(h, w) > 0] = 255
    return out


def _strip_spanning_grid_lines(mask: np.ndarray, span_frac: float = 0.42) -> np.ndarray:
    """
    Remove long H/V runs that span most of the sheet (structural grid, borders).
    span_frac controls the threshold: 0.42 for margin-aware passes (walls shorter
    than ~40 % of sheet); 0.90 for snap-mask passes so long exterior wall pixel
    runs are NOT stripped before snap_segments_to_walls uses them.
    """
    h, w = mask.shape
    out = mask.copy()
    kw = max(40, int(w * span_frac))
    kh = max(40, int(h * span_frac))
    long_h = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    )
    long_v = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    )
    out = cv2.bitwise_and(out, cv2.bitwise_not(long_h))
    out = cv2.bitwise_and(out, cv2.bitwise_not(long_v))
    return out


def _extract_wall_lines(
    image: np.ndarray,
    blank_right_frac: Optional[float] = None,
    apply_margins: bool = True,
    px_per_unit: float = 18.0,
) -> np.ndarray:
    """Threshold → clean noise → extract H/V wall lines via double-line pairing.

    apply_margins=True  (default): blank title-block/header zones before
        detection — used for footprint detection so sheet borders and title
        text don't pollute the wall-pair mask.
    apply_margins=False: skip the margin-blanking step — used when building
        a snap mask so exterior wall pixels near the sheet edge are preserved.

    px_per_unit drives the wall-pair gap range so it tracks drawing scale and
    DPI. Walls whose two faces are separated by 2–12 real-world inches are
    kept; everything else (single-stroke annotation lines, grid) is discarded.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = image.shape
    if apply_margins:
        image = _blank_sheet_margins(image)
    # Legacy: optional full-height right strip (off by default)
    if blank_right_frac is not None:
        image[:, int(w * blank_right_frac) :] = 255

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = 40
    keep = np.zeros(num_labels, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[label] = 255
    cleaned = keep[labels]

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    horiz = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vert_kernel)

    # Keep only lines that have a close parallel partner — the two faces of a wall.
    # Single-stroke annotation lines (dimensions, borders, grid) have no partner
    # and are discarded here.
    # Gap range is derived from px_per_unit so it scales with drawing scale & DPI:
    #   min = 2 real inches  (thinnest possible wall face-to-face gap)
    #   max = 12 real inches (thickest typical exterior wall)
    # At 1/4"=1ft / 150 DPI (px_per_unit≈37.5): min=6, max=55 (same as old hardcoded).
    # At 1/4"=1ft / 300 DPI (px_per_unit≈75): min=12, max=75 (catches thick walls).
    # At 1/8"=1ft / 150 DPI (px_per_unit≈18.75): min=3 (catches thin pairs at small scale).
    min_gap_px = max(3, int(px_per_unit / 6))   # 2 inches = px_per_unit/6
    max_gap_px = max(55, int(px_per_unit))        # 12 inches = px_per_unit
    print(f"  [wall-pairs] gap range {min_gap_px}–{max_gap_px} px "
          f"(px_per_unit={px_per_unit:.1f}, margins={apply_margins})", file=sys.stderr)
    horiz_walls = _find_wall_pairs(horiz, scan_rows=True, min_gap_px=min_gap_px, max_gap_px=max_gap_px)
    vert_walls = _find_wall_pairs(vert, scan_rows=False, min_gap_px=min_gap_px, max_gap_px=max_gap_px)
    combined = cv2.bitwise_or(horiz_walls, vert_walls)
    # Snap mask (apply_margins=False) must keep long exterior wall runs intact so
    # snap_segments_to_walls can find them; raise threshold to 0.90 for that pass.
    # Footprint pass raised to 0.65 so exterior walls spanning 42–65% of the sheet
    # are not incorrectly stripped as structural grid lines.
    span_frac = 0.65 if apply_margins else 0.90
    return _strip_spanning_grid_lines(combined, span_frac=span_frac)


def preprocess(
    image: np.ndarray,
    blank_right_frac: Optional[float] = None,
    px_per_unit: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-pass preprocessing:
      1. Extract H/V wall lines at full resolution.
         - Filtered mask (margins blanked): drives footprint detection so
           title-block borders and header text are excluded.
         - Full mask (no margin blanking): returned for segment snapping so
           exterior wall pixels near the sheet edge are preserved.
      2. Downscale → heavy dilation+closing to bridge doorways →
         upscale the solid footprint mask back to full resolution (big).

    px_per_unit drives the adaptive closing kernel so windows and doors
    (up to 12 real-world units wide) are always bridged regardless of scale/DPI.

    Returns (big, wall_pair_mask_full).
    """
    t0 = time.time()
    # Filtered mask: used for footprint morphology (excludes sheet margins).
    wall_pair_mask_filtered = _extract_wall_lines(
        image, blank_right_frac=blank_right_frac, apply_margins=True,
        px_per_unit=px_per_unit,
    )
    # Full mask: used for snapping — exterior wall pixels near the sheet edge
    # must NOT be erased, otherwise snap_segments_to_walls finds nothing there.
    wall_pair_mask_full = _extract_wall_lines(
        image, blank_right_frac=blank_right_frac, apply_margins=False,
        px_per_unit=px_per_unit,
    )
    print(f"  [preprocess] wall-line extraction: {time.time()-t0:.1f}s", file=sys.stderr)

    h, w = wall_pair_mask_filtered.shape
    small_h, small_w = h // DOWNSCALE, w // DOWNSCALE

    t0 = time.time()
    small = cv2.resize(wall_pair_mask_filtered, (small_w, small_h), interpolation=cv2.INTER_AREA)
    _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)

    small = cv2.dilate(
        small,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2,
    )

    # Adaptive closing kernel: bridge window/door gaps up to 12 real-world units.
    # At 1/8"=1'/144 DPI: 12*18/4=54; at 1/4"=1'/144 DPI: 12*36/4=108.
    # The hard-coded 25 only bridged ~2.7 ft gaps, causing flood-fill leakage.
    close_k_size = max(25, int(12 * px_per_unit / DOWNSCALE))
    print(f"  [preprocess] adaptive close_k_size={close_k_size} (px_per_unit={px_per_unit:.1f})", file=sys.stderr)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k_size, close_k_size))
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, close_k, iterations=1)

    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    print(f"  [preprocess] downscale morphology: {time.time()-t0:.1f}s", file=sys.stderr)

    return big, wall_pair_mask_full

def flood_fill_interior(binary: np.ndarray) -> np.ndarray:
    """
    Flood-fill from the image border to mark the exterior.
    Everything NOT reached (and not a wall) is building interior.
    Returns a solid building mask (walls + interior).
    """
    h, w = binary.shape

    # Pad so flood fill can reach all border-connected background
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1 : h + 1, 1 : w + 1] = binary

    flood = padded.copy()
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)

    # Interior = background pixels the flood couldn't reach
    interior = np.zeros((h + 2, w + 2), np.uint8)
    interior[flood == 0] = 255
    interior = interior[1 : h + 1, 1 : w + 1]

    if cv2.countNonZero(interior) < 100:
        return binary

    return cv2.bitwise_or(binary, interior)


def find_footprint(binary: np.ndarray):
    """
    Pick the building footprint: the largest connected component that
    is NOT the sheet border/title block.

    Strategy:
      - Pass 1: reject components that touch the image edge (sheet border artifacts).
      - Pass 2 (fallback): allow edge-touching blobs but use stricter aspect ratio
        (< 8) and lower max-area limit (92%) to avoid picking up the sheet border
        itself. Useful for plans where the building fills most of the sheet.
      - Reject components that fill most of the image (the border itself).
      - Reject components with extreme aspect ratios (title blocks, scale bars).
      - Among survivors, pick the largest by area × compactness.
    """
    filled = flood_fill_interior(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled)

    h, w = filled.shape
    image_area = h * w
    edge_margin_x = w * 0.01
    edge_margin_y = h * 0.01
    excl = _build_exclusion_mask(h, w)

    def _score_components(allow_edge_touch: bool) -> Optional[int]:
        best_label = None
        best_score = -1.0
        max_area_frac = 0.92 if allow_edge_touch else 0.85
        max_aspect = 8 if allow_edge_touch else 12

        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            if area < image_area * 0.005:
                continue
            if area > image_area * max_area_frac:
                continue

            aspect = width / max(height, 1)
            if aspect > max_aspect or aspect < 1 / max_aspect:
                continue

            if not allow_edge_touch:
                touches_edge = (
                    x <= edge_margin_x
                    or y <= edge_margin_y
                    or (x + width) >= (w - edge_margin_x)
                    or (y + height) >= (h - edge_margin_y)
                )
                if touches_edge:
                    continue

            comp = (labels == label).astype(np.uint8)
            overlap = np.logical_and(comp, excl > 0).sum() / max(comp.sum(), 1)
            if overlap > 0.12:
                continue

            cx = x + width / 2
            cy = y + height / 2
            if _point_in_exclusion(cx, cy, w, h):
                continue

            compactness = area / max(width * height, 1)
            score = area * compactness
            if score > best_score:
                best_score = score
                best_label = label

        return best_label

    best_label = _score_components(allow_edge_touch=False)
    if best_label is None:
        print("  [find_footprint] pass 1 found nothing; retrying with edge-touching blobs allowed",
              file=sys.stderr)
        best_label = _score_components(allow_edge_touch=True)

    if best_label is None:
        return None

    mask = np.zeros_like(filled)
    mask[labels == best_label] = 255
    return mask

def find_footprint_contour(mask: np.ndarray):
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest


# ─── Step 4: Simplify contour → clean polygon ──────────────────────────────
def simplify_polygon(contour: np.ndarray, epsilon_factor: float = 0.001) -> np.ndarray:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    return approx.reshape(-1, 2)


# ─── Step 6: Wall direction (facing) ───────────────────────────────────────

def wall_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Angle of the wall segment in degrees, measured clockwise from
    image-up (North).  Image coords: y increases downward.
    """
    dx = x2 - x1
    dy = -(y2 - y1)  # flip y so up is positive
    angle = math.degrees(math.atan2(dx, dy)) % 360
    return angle


def angle_to_facing(angle_deg: float) -> str:
    """
    The wall *faces* perpendicular to its run direction (outward normal).
    A wall running East–West has its outer face pointing North or South.

    We compute the wall's run angle, then assign the facing as the
    outward-perpendicular (+90°) snapped to the nearest cardinal direction.
    """
    # Outward normal is ambiguous without winding order — we assume the
    # contour is CCW in image coords (CW in math coords), so +90° points
    # outward.
    normal_angle = (angle_deg + 90) % 360

    if 315 <= normal_angle or normal_angle < 45:
        return "North"
    elif 45 <= normal_angle < 135:
        return "East"
    elif 135 <= normal_angle < 225:
        return "South"
    else:
        return "West"



def pixel_length(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _filter_wall_segments(
    segments: list[tuple],
    img_w: int,
    img_h: int,
    max_span_frac: float = 0.38,
    roi: Optional[dict] = None,
) -> list[tuple]:
    """Drop title-block/grid segments outside the building footprint."""
    max_span = min(img_w, img_h) * max_span_frac
    kept = []
    for x1, y1, x2, y2 in segments:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if roi:
            xf, yf = mx / img_w, my / img_h
            if (
                xf < roi["x0_pct"]
                or xf > roi["x1_pct"]
                or yf < roi["y0_pct"]
                or yf > roi["y1_pct"]
            ):
                continue
        elif _point_in_exclusion(mx, my, img_w, img_h):
            continue
        if pixel_length(x1, y1, x2, y2) > max_span:
            continue
        kept.append((x1, y1, x2, y2))
    return kept


def measure_walls(segments: list[tuple], px_per_unit: float, unit_label: str) -> list[dict]:
    walls = []
    for i, (x1, y1, x2, y2) in enumerate(segments):
        px_len = pixel_length(x1, y1, x2, y2)
        real_len = px_len / px_per_unit
        angle = wall_angle_deg(x1, y1, x2, y2)
        facing = angle_to_facing(angle)

        walls.append({
            "id": f"w{i + 1}",
            "name": f"{facing} Wall {i + 1}",
            "facing": facing,
            "length": f"{real_len:.2f} {unit_label}",
            "length_raw": round(real_len, 2),
            "angle_deg": round(angle, 1),
            "px_coords": [x1, y1, x2, y2],
        })
    return walls


# ─── Wall-position snapping ─────────────────────────────────────────────────

def snap_segments_to_walls(
    segments: list[tuple],
    wall_mask: np.ndarray,
    search_radius: int = 80,
    min_pixels: int = 10,
) -> list[tuple]:
    """
    Shift each polygon-derived segment onto the nearest actual wall-pair pixels.

    The polygon contour is traced on an inflated binary mask, so its edges sit
    outside the real wall lines.  For each segment we scan the undilated
    wall_pair_mask perpendicular to the segment direction, find the median
    position of wall pixels within search_radius, and snap the segment there.
    Falls back to the original position when no wall pixels are found.
    """
    h, w = wall_mask.shape
    snapped = []
    for (x1, y1, x2, y2) in segments:
        is_horiz = abs(x2 - x1) >= abs(y2 - y1)
        if is_horiz:
            cy = (y1 + y2) // 2
            y_lo = max(0, cy - search_radius)
            y_hi = min(h - 1, cy + search_radius)
            x_lo = max(0, min(x1, x2) + 5)
            x_hi = min(w - 1, max(x1, x2) - 5)
            if x_lo < x_hi and y_lo < y_hi:
                strip = wall_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
                ys, _ = np.where(strip > 0)
                if len(ys) >= min_pixels:
                    ny = int(np.median(ys)) + y_lo
                    snapped.append((x1, ny, x2, ny))
                    continue
        else:
            cx = (x1 + x2) // 2
            x_lo = max(0, cx - search_radius)
            x_hi = min(w - 1, cx + search_radius)
            y_lo = max(0, min(y1, y2) + 5)
            y_hi = min(h - 1, max(y1, y2) - 5)
            if x_lo < x_hi and y_lo < y_hi:
                strip = wall_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
                _, xs = np.where(strip > 0)
                if len(xs) >= min_pixels:
                    nx = int(np.median(xs)) + x_lo
                    snapped.append((nx, y1, nx, y2))
                    continue
        snapped.append((x1, y1, x2, y2))
    return snapped


def merge_and_deduplicate_segments(
    segments: list[tuple],
    axis_tol_px: int = 12,
    gap_tol_px: int = 8,
) -> list[tuple]:
    """Collapse coaxial duplicate segments and subsummed fragments.

    Fixes two sources of measurement overlap that arise from combining polygon
    and Hough segments:

    1. Double-lined walls: the footprint polygon and Hough lines can each
       detect a different one of the two ink lines that form a drawn wall,
       producing two near-parallel segments on the same axis.  These are
       merged into one by clustering segments whose perpendicular (off-axis)
       distance is within ``axis_tol_px`` and union-merging their 1D extents.

    2. Window/door stubs vs. spanning segment: the polygon traces a
       continuous span across a window gap (bridged by morphological close),
       while Hough finds the two real wall stubs on either side.  All three
       land on the same axis and are collapsed into the longest span.

    Algorithm
    ---------
    Pass 1 – coaxial merge:
        • Split into H and V groups.
        • Sort each group by perpendicular coordinate.
        • Single-linkage cluster by perpendicular distance (≤ axis_tol_px).
        • Within each cluster, sort 1D projections and union-merge intervals
          that overlap or are within gap_tol_px of each other.
        • Emit one merged segment per interval, at the cluster's median
          perpendicular coordinate.

    Pass 2 – subsumption drop (safety net):
        • Drop any segment whose 1D extent is fully contained within a longer
          co-axial segment (same orientation, perp within axis_tol_px).
    """
    if not segments:
        return segments

    def _perp(seg: tuple, horiz: bool) -> int:
        return (seg[1] + seg[3]) // 2 if horiz else (seg[0] + seg[2]) // 2

    def _proj(seg: tuple, horiz: bool) -> tuple[int, int]:
        if horiz:
            return min(seg[0], seg[2]), max(seg[0], seg[2])
        return min(seg[1], seg[3]), max(seg[1], seg[3])

    merged_result: list[tuple] = []

    for is_horiz in (True, False):
        segs = [s for s in segments
                if (abs(s[2] - s[0]) >= abs(s[3] - s[1])) == is_horiz]
        if not segs:
            continue

        segs.sort(key=lambda s: _perp(s, is_horiz))

        clusters: list[list[tuple]] = []
        for seg in segs:
            p = _perp(seg, is_horiz)
            if clusters and p - _perp(clusters[-1][-1], is_horiz) <= axis_tol_px:
                clusters[-1].append(seg)
            else:
                clusters.append([seg])

        for cluster in clusters:
            perps = sorted(_perp(s, is_horiz) for s in cluster)
            median_perp = perps[len(perps) // 2]
            intervals = sorted(_proj(s, is_horiz) for s in cluster)

            lo, hi = intervals[0]
            for nlo, nhi in intervals[1:]:
                if nlo <= hi + gap_tol_px:
                    hi = max(hi, nhi)
                else:
                    if is_horiz:
                        merged_result.append((lo, median_perp, hi, median_perp))
                    else:
                        merged_result.append((median_perp, lo, median_perp, hi))
                    lo, hi = nlo, nhi
            if is_horiz:
                merged_result.append((lo, median_perp, hi, median_perp))
            else:
                merged_result.append((median_perp, lo, median_perp, hi))

    # Pass 2: drop any segment fully contained within a longer co-axial one.
    final: list[tuple] = []
    for i, seg_a in enumerate(merged_result):
        is_horiz_a = abs(seg_a[2] - seg_a[0]) >= abs(seg_a[3] - seg_a[1])
        pa = _perp(seg_a, is_horiz_a)
        lo_a, hi_a = _proj(seg_a, is_horiz_a)
        subsumed = False
        for j, seg_b in enumerate(merged_result):
            if i == j:
                continue
            is_horiz_b = abs(seg_b[2] - seg_b[0]) >= abs(seg_b[3] - seg_b[1])
            if is_horiz_a != is_horiz_b:
                continue
            pb = _perp(seg_b, is_horiz_b)
            if abs(pa - pb) > axis_tol_px:
                continue
            lo_b, hi_b = _proj(seg_b, is_horiz_b)
            if lo_b <= lo_a and hi_b >= hi_a and (hi_b - lo_b) > (hi_a - lo_a):
                subsumed = True
                break
        if not subsumed:
            final.append(seg_a)

    return final


def _hough_supplement(
    wall_mask: np.ndarray,
    existing_segments: list[tuple],
    min_length_px: float = 60,
    max_gap_px: int = 20,
    dedup_tol_px: int = 22,
) -> list[tuple]:
    """
    Run HoughLinesP on wall_mask to find wall segments missed by the polygon approach.

    The polygon-based pipeline only extracts the building footprint's outer edges; any
    interior walls and recesses smoothed away by approxPolyDP are invisible to it.
    This supplement detects all significant H/V runs in the wall_pair_mask and adds any
    that are not already represented in existing_segments.

    Returns a list of new (x1, y1, x2, y2) tuples in image-pixel coordinates.
    """
    # Light dilation bridges thin ink gaps so Hough gets longer continuous runs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(wall_mask, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        dilated,
        rho=1,
        theta=np.pi / 180,
        threshold=max(20, int(min_length_px * 0.6)),
        minLineLength=int(min_length_px),
        maxLineGap=max_gap_px,
    )
    if lines is None:
        return []

    def _is_duplicate(seg: tuple) -> bool:
        """True if seg is close to and overlaps an existing or already-accepted segment."""
        nx1, ny1, nx2, ny2 = seg
        n_horiz = abs(nx2 - nx1) >= abs(ny2 - ny1)
        for ex1, ey1, ex2, ey2 in existing_segments:
            e_horiz = abs(ex2 - ex1) >= abs(ey2 - ey1)
            if n_horiz != e_horiz:
                continue
            if n_horiz:
                if abs((ny1 + ny2) / 2 - (ey1 + ey2) / 2) > dedup_tol_px:
                    continue
                n_lo, n_hi = min(nx1, nx2), max(nx1, nx2)
                e_lo, e_hi = min(ex1, ex2), max(ex1, ex2)
                overlap = min(n_hi, e_hi) - max(n_lo, e_lo)
                if n_hi > n_lo and overlap / (n_hi - n_lo) >= 0.30:
                    return True
            else:
                if abs((nx1 + nx2) / 2 - (ex1 + ex2) / 2) > dedup_tol_px:
                    continue
                n_lo, n_hi = min(ny1, ny2), max(ny1, ny2)
                e_lo, e_hi = min(ey1, ey2), max(ey1, ey2)
                overlap = min(n_hi, e_hi) - max(n_lo, e_lo)
                if n_hi > n_lo and overlap / (n_hi - n_lo) >= 0.30:
                    return True
        return False

    accepted: list[tuple] = []
    all_check = list(existing_segments)  # grows as new segs are accepted

    for line in lines:
        # Cast to Python int immediately — numpy.intc from HoughLinesP is not JSON-serializable.
        x1, y1, x2, y2 = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])

        # Orthogonal check: within 10° of H or V axis.
        angle = abs(math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1))))
        is_horiz = angle < 10
        is_vert = angle > 80
        if not (is_horiz or is_vert):
            continue

        # Snap to exact H or V so coordinates are axis-aligned.
        if is_horiz:
            cy = (y1 + y2) // 2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1 = y2 = cy
        else:
            cx = (x1 + x2) // 2
            y1, y2 = min(y1, y2), max(y1, y2)
            x1 = x2 = cx

        seg = (x1, y1, x2, y2)
        if not _is_duplicate(seg):
            accepted.append(seg)
            all_check.append(seg)

    return accepted


def detect_wall_at_point(
    wall_pair_mask: np.ndarray,
    x_px: int,
    y_px: int,
    search_radius: int = 150,
    min_run_px: int = 15,
) -> Optional[dict]:
    """
    Find the wall segment closest to (x_px, y_px) in wall_pair_mask.

    Scans a horizontal strip and a vertical strip of width/height 1 px centred
    on the click point.  Whichever axis has more wall pixels is taken as the
    wall orientation, and a connected run of wall pixels along that axis is
    returned as the segment.

    Returns a dict {'px_coords': [x1,y1,x2,y2], 'facing': str} or None.
    """
    h, w = wall_pair_mask.shape
    x_lo = max(0, x_px - search_radius)
    x_hi = min(w - 1, x_px + search_radius)
    y_lo = max(0, y_px - search_radius)
    y_hi = min(h - 1, y_px + search_radius)

    # Sample horizontal and vertical strips around the click point.
    h_strip = wall_pair_mask[y_px, x_lo:x_hi + 1]
    v_strip = wall_pair_mask[y_lo:y_hi + 1, x_px]
    h_count = int(np.count_nonzero(h_strip))
    v_count = int(np.count_nonzero(v_strip))

    # #region agent log
    import json as _json, os as _os
    _log_path = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'debug-7104c9.log'))
    _total_px = int(np.count_nonzero(wall_pair_mask))
    with open(_log_path, 'a') as _lf:
        _lf.write(_json.dumps({'sessionId':'7104c9','location':'preprocess.py:detect_wall_at_point','message':'entry','data':{'mask_shape':[h,w],'total_nonzero_px':_total_px,'x_px':x_px,'y_px':y_px,'h_count':h_count,'v_count':v_count,'search_radius':search_radius,'min_run_px':min_run_px},'timestamp':int(__import__('time').time()*1000),'hypothesisId':'WALL-MASK'}) + '\n')
    # #endregion

    if h_count == 0 and v_count == 0:
        return None

    is_horiz = h_count >= v_count

    if is_horiz:
        # Scan the full row at y_px across the search window; find the median
        # wall-pixel row in ±search_radius to snap to the true wall centre.
        region = wall_pair_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
        ys, _ = np.where(region > 0)
        if len(ys) < min_run_px:
            return None
        wall_row = int(np.median(ys)) + y_lo

        # Extend the segment along the snapped row as far as wall pixels go.
        row_pixels = wall_pair_mask[wall_row, :]
        xs = np.where(row_pixels > 0)[0]
        near_xs = xs[(xs >= x_lo) & (xs <= x_hi)]
        if len(near_xs) < min_run_px:
            return None
        x1, x2 = int(near_xs[0]), int(near_xs[-1])
        facing = "North" if wall_row < h // 2 else "South"
        return {"px_coords": [x1, wall_row, x2, wall_row], "facing": facing}
    else:
        region = wall_pair_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
        _, xs = np.where(region > 0)
        if len(xs) < min_run_px:
            return None
        wall_col = int(np.median(xs)) + x_lo

        col_pixels = wall_pair_mask[:, wall_col]
        ys = np.where(col_pixels > 0)[0]
        near_ys = ys[(ys >= y_lo) & (ys <= y_hi)]
        if len(near_ys) < min_run_px:
            return None
        y1, y2 = int(near_ys[0]), int(near_ys[-1])
        facing = "West" if wall_col < w // 2 else "East"
        return {"px_coords": [wall_col, y1, wall_col, y2], "facing": facing}


# ─── Main pipeline ──────────────────────────────────────────────────────────

def _crop_to_roi(image: np.ndarray, roi: dict) -> tuple[np.ndarray, tuple[int, int], int, int]:
    """Crop to user box; return (crop, (offset_x, offset_y), full_w, full_h)."""
    full_h, full_w = image.shape[:2]
    x0 = int(max(0.0, min(1.0, roi["x0_pct"])) * full_w)
    y0 = int(max(0.0, min(1.0, roi["y0_pct"])) * full_h)
    x1 = int(max(0.0, min(1.0, roi["x1_pct"])) * full_w)
    y1 = int(max(0.0, min(1.0, roi["y1_pct"])) * full_h)
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return image, (0, 0), full_w, full_h
    return image[y0:y1, x0:x1].copy(), (x0, y0), full_w, full_h


def _shift_px_coords(coords: list, offset: tuple[int, int]) -> list:
    ox, oy = offset
    x1, y1, x2, y2 = coords
    return [x1 + ox, y1 + oy, x2 + ox, y2 + oy]


def analyze_page(image: np.ndarray, scale_str: str, dpi: int, roi: Optional[dict] = None) -> dict:
    full_w = image.shape[1]
    full_h = image.shape[0]
    roi_offset = (0, 0)
    if roi:
        image, roi_offset, full_w, full_h = _crop_to_roi(image, roi)

    # Parse scale first so px_per_unit is available for scale-adaptive parameters.
    cal = parse_scale(scale_str, dpi, output_unit="ft")
    px_per_unit = cal["px_per_unit"]
    unit_label = cal["unit_label"]
    is_metric = unit_label == "m"
    area_unit = f"{unit_label}²"

    t0 = time.time()
    binary, wall_pair_mask = preprocess(image, px_per_unit=px_per_unit)
    print(f"  [pipeline] preprocess total: {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    component_mask = find_footprint(binary)
    print(f"  [pipeline] find_footprint: {time.time()-t0:.1f}s", file=sys.stderr)

    if component_mask is None:
        return {"error": "No building footprint found"}

    contour = find_footprint_contour(component_mask)
    if contour is None:
        return {"error": "No building footprint found"}

    eps = 0.006 if roi else 0.001
    polygon = simplify_polygon(contour, epsilon_factor=eps)
    analyze_page._last_polygon = polygon
    img_h, img_w = image.shape[:2]
    # Scale-adaptive minimum: walls shorter than 12 real-world units are polygon
    # artifacts, not real exterior walls. At 1/8"=1'/144 DPI this is 216 px.
    min_seg_px = max(60, int(12 * px_per_unit))
    print(f"  [pipeline] min_seg_px={min_seg_px} (px_per_unit={px_per_unit:.1f})", file=sys.stderr)
    segments = extract_wall_segments(polygon, min_length_px=min_seg_px)
    raw_count = len(segments)

    # Build a filter ROI from the polygon's own bounding box (+ 3 % padding).
    # Passing roi= instead of roi=None means _filter_wall_segments uses a
    # bounds check rather than _point_in_exclusion, which has hard-coded
    # sheet-fraction exclusion zones (top 12 %, bottom 18 %, etc.) that
    # incorrectly drop exterior walls of buildings that fill the sheet.
    # max_span_frac is 0.95 because polygon edges can legitimately span most
    # of the image (the old 0.38 default was sized for raw grid-line detection).
    xs_poly, ys_poly = polygon[:, 0], polygon[:, 1]
    pad = int(min(img_w, img_h) * 0.03)
    poly_roi = {
        "x0_pct": max(0.0, float(xs_poly.min() - pad) / img_w),
        "y0_pct": max(0.0, float(ys_poly.min() - pad) / img_h),
        "x1_pct": min(1.0, float(xs_poly.max() + pad) / img_w),
        "y1_pct": min(1.0, float(ys_poly.max() + pad) / img_h),
    }
    segments = _filter_wall_segments(segments, img_w, img_h, roi=poly_roi, max_span_frac=0.95)
    print(f"  [pipeline] segments: raw={raw_count} after_filter={len(segments)}", file=sys.stderr)
    segments = snap_segments_to_walls(segments, wall_pair_mask)
    print(f"  [pipeline] segments after snap={len(segments)}", file=sys.stderr)

    # Supplement with Hough lines detected directly on the wall_pair_mask so that
    # interior walls and recesses smoothed out by approxPolyDP are also captured.
    hough_segs = _hough_supplement(
        wall_pair_mask, segments,
        min_length_px=min_seg_px,
        dedup_tol_px=max(55, int(px_per_unit)),
    )
    if hough_segs:
        hough_segs = _filter_wall_segments(
            hough_segs, img_w, img_h, roi=poly_roi, max_span_frac=0.95
        )
        hough_segs = snap_segments_to_walls(hough_segs, wall_pair_mask)
        print(f"  [pipeline] Hough supplement: +{len(hough_segs)} segments "
              f"(total will be {len(segments) + len(hough_segs)})", file=sys.stderr)
        segments = segments + hough_segs

    # Deduplicate and merge the combined segment list.  Two common sources of
    # overlap are collapsed here:
    #   • Double-lined walls: polygon and Hough each detect a different ink
    #     line of the same drawn wall → coaxial merge within axis_tol_px.
    #   • Window/door stubs: polygon spans the full wall; Hough finds the two
    #     stubs on either side of the opening → subsumed into the longer span.
    axis_tol_px = max(55, int(px_per_unit))   # must cover full wall-pair gap so both ink lines collapse
    gap_tol_px = max(5, int(0.3 * px_per_unit))
    before_dedup = len(segments)
    segments = merge_and_deduplicate_segments(
        segments, axis_tol_px=axis_tol_px, gap_tol_px=gap_tol_px
    )
    print(f"  [pipeline] dedup: {before_dedup} → {len(segments)} segments "
          f"(axis_tol={axis_tol_px}px, gap_tol={gap_tol_px}px)", file=sys.stderr)

    walls = measure_walls(segments, px_per_unit, unit_label)
    min_real = 8.0 if unit_label == "ft" else 2.5
    walls_before_len = list(walls)
    if roi:
        walls = [w for w in walls if w["length_raw"] >= min_real]
        if len(walls) < 8:
            walls = sorted(walls_before_len, key=lambda w: -w["length_raw"])[:20]

    if roi_offset != (0, 0):
        ox, oy = roi_offset
        for w in walls:
            w["px_coords"] = _shift_px_coords(w["px_coords"], roi_offset)
        polygon = polygon + np.array([ox, oy])
        contour_shifted = contour + np.array([[ox, oy]])
        total_area_px = cv2.contourArea(contour_shifted)
    else:
        total_area_px = cv2.contourArea(contour)

    img_w, img_h = full_w, full_h
    if not walls and walls_before_len:
        walls = sorted(walls_before_len, key=lambda w: -w["length_raw"])[:12]
    total_area_real = total_area_px / (px_per_unit ** 2)

    xs = polygon[:, 0]
    ys = polygon[:, 1]
    fp_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    poly_out = polygon.astype(int).tolist()

    # Cache wall_pair_mask so detect_wall_at_point can reload it instantly
    # without re-running the expensive _extract_wall_lines pass.
    mask_cache_path = None
    try:
        cache_name = f"arqen_mask_{uuid.uuid4().hex[:12]}.png"
        mask_cache_path = str(Path(tempfile.gettempdir()) / cache_name)
        cv2.imwrite(mask_cache_path, wall_pair_mask)
    except Exception as e:
        print(f"  [pipeline] mask cache write failed: {e}", file=sys.stderr)
        mask_cache_path = None

    return {
        "detected_scale": scale_str,
        "total_area": f"{total_area_real:.1f} {area_unit}",
        "units": "metric" if is_metric else "imperial",
        "polygon_vertices": len(polygon),
        "footprint_polygon_px": poly_out,
        "image_size_px": [full_w, full_h],
        "footprint_bbox_px": fp_bbox,
        "px_per_ft": round(px_per_unit, 2),
        "walls": walls,
        "mask_cache_path": mask_cache_path,
        # roi_offset is the pixel offset of the cropped image within the full image.
        # detect_wall_at_point needs this to translate full-image click coords into
        # crop-image coords before querying the mask.
        "mask_roi_offset": list(roi_offset),
    }


def visualize(image: np.ndarray, polygon: np.ndarray, output_path: str):
    vis = image.copy()
    pts = polygon.reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    for i, (x, y) in enumerate(polygon):
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract exterior wall measurements from a plan PDF")
    parser.add_argument("pdf", help="Path to the architectural plan PDF")
    parser.add_argument("--scale", required=True,
                        help='Drawing scale, e.g. "1/4in=1ft" or "1:100"')
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI (default: 300)")
    parser.add_argument("--page", type=int, default=1, help="Page number to analyze (1-indexed)")
    parser.add_argument("--visualize", action="store_true", help="Save annotated image")
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    images = pdf_to_images(str(pdf_path), dpi=args.dpi)
    page_idx = args.page - 1
    if page_idx < 0 or page_idx >= len(images):
        print(f"Error: page {args.page} out of range (PDF has {len(images)} pages)", file=sys.stderr)
        sys.exit(1)

    image = images[page_idx]
    result = analyze_page(image, args.scale, args.dpi)

    if args.visualize and "walls" in result:
        polygon = analyze_page._last_polygon
        vis_path = str(pdf_path.with_suffix(".annotated.png"))
        visualize(image, polygon, vis_path)
        result["visualization"] = vis_path
        print(f"Saved annotated image to {vis_path}", file=sys.stderr)

    output = json.dumps(result, indent=2)
    with open("out.json", "w") as f:
        f.write(output)
    print(f"Results written to out.json", file=sys.stderr)


if __name__ == "__main__":
    main()