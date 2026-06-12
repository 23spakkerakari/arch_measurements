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
from calibration_validate import (
    check_dpi_alternatives,
    issue_to_log_line,
    summarize_calibration,
    validate_dpi,
    validate_footprint_span,
    validate_px_per_unit,
    validate_total_area,
)
from door_detect import detect_doors, ink_mask_from_image
from window_detect import detect_windows
from extract_wall_segments_class import extract_wall_segments
from room_wall_split import (
    clamp_segments_to_envelope,
    drop_segments_outside_exterior,
    segment_traces_exterior,
    split_exterior_walls_by_room,
)


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


def dedup_axis_tol_px(px_per_unit: float) -> int:
    """
    Perpendicular tolerance for coaxial segment merge in dedup.

    ~8 real inches: enough to collapse the two ink strokes of one double-drawn
    wall, but below the typical face-to-face wall thickness at px_per_unit
    (12 in) so the north exterior's two parallel lines are not merged into a
    single segment at the wrong y.  Complete-linkage clustering prevents
    chain-merging distinct walls farther apart.
    """
    return max(12, int(0.6 * px_per_unit))


def wall_pair_gap_range(px_per_unit: float) -> tuple[int, int]:
    """
    Perpendicular gap range (px) between the two ink strokes of a drawn wall.

    min = 2 real inches (thinnest wall face-to-face gap) = px_per_unit / 6
    max = 12 real inches + 50% margin = 1.5 * px_per_unit

    The old code floored max at a hard 55 px (tuned for 1/4"=1ft @ 150 DPI,
    px_per_unit≈37.5 → 1.5x≈56, same value). At smaller scales such as
    px_per_unit=18 that floor was 3 real feet: dimension strings 2–3 ft from a
    wall got "paired", and — used as the dedup axis tolerance — two distinct
    parallel walls up to 3 ft apart were chain-merged into one, deleting real
    walls. Deriving it from px_per_unit keeps it at ~12–18 real inches at
    every scale.
    """
    min_gap_px = max(3, int(px_per_unit / 6))
    max_gap_px = max(12, int(1.5 * px_per_unit))
    return min_gap_px, max_gap_px


def _find_wall_pairs(
    mask: np.ndarray,
    scan_rows: bool,
    min_gap_px: int = 2,
    max_gap_px: int = 60,
    strip_px: int = 128,
) -> np.ndarray:
    """
    Retain only pixels that belong to a double-line (wall) pair.

    Walls are drawn as two parallel strokes separated by wall thickness
    (typically 3.5"–8" at drawing scale → ~10–60 px at 300 DPI / 1:48 scale).
    Single-stroke lines — dimension strings, title block borders, grid lines,
    scale bars — have no parallel partner and are filtered out.

    Pairing is evaluated per strip of ``strip_px`` along the line direction
    (X strips for horizontal lines, Y strips for vertical) rather than over a
    whole-image projection. A real wall's two parallel strokes co-exist in
    every strip along their run, so they always pair locally; an isolated
    dimension string only pairs if unrelated ink happens to sit within the
    gap range in the *same* strip — instead of anywhere on the sheet — which
    eliminates most annotation-line false positives.

    scan_rows=True  → horizontal lines: look for row pairs close in Y
    scan_rows=False → vertical lines:   look for column pairs close in X
    """
    h, w = mask.shape
    out = np.zeros_like(mask)
    extent = w if scan_rows else h

    for s0 in range(0, extent, strip_px):
        s1 = min(extent, s0 + strip_px)
        strip = mask[:, s0:s1] if scan_rows else mask[s0:s1, :]
        if scan_rows:
            projection = (strip > 0).any(axis=1)   # (H,): True if row has any pixel
        else:
            projection = (strip > 0).any(axis=0)   # (W,): True if col has any pixel

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
            out[:, s0:s1] = strip * keep[:, np.newaxis].astype(np.uint8)
        else:
            out[s0:s1, :] = strip * keep[np.newaxis, :].astype(np.uint8)

    return out


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


def _strip_spanning_grid_lines(
    mask: np.ndarray,
    span_frac: float = 0.42,
    strip_vertical: bool = True,
) -> np.ndarray:
    """
    Remove long H/V runs that span most of the sheet (structural grid, borders).
    span_frac controls the threshold: 0.42 for margin-aware passes (walls shorter
    than ~40 % of sheet); 0.90 for snap-mask passes so long exterior wall pixel
    runs are NOT stripped before snap_segments_to_walls uses them.

    strip_vertical=False skips the vertical pass — used in ROI crops where west/
    east exterior walls legitimately span ~90–100 % of crop height.
    """
    h, w = mask.shape
    out = mask.copy()
    kw = max(40, int(w * span_frac))
    long_h = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    )
    out = cv2.bitwise_and(out, cv2.bitwise_not(long_h))
    if strip_vertical:
        kh = max(40, int(h * span_frac))
        long_v = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
        )
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
    # and are discarded here. Gap range is derived from px_per_unit so it scales
    # with drawing scale & DPI (see wall_pair_gap_range).
    min_gap_px, max_gap_px = wall_pair_gap_range(px_per_unit)
    print(f"  [wall-pairs] gap range {min_gap_px}–{max_gap_px} px "
          f"(px_per_unit={px_per_unit:.1f}, margins={apply_margins})", file=sys.stderr)
    horiz_walls = _find_wall_pairs(horiz, scan_rows=True, min_gap_px=min_gap_px, max_gap_px=max_gap_px)
    vert_walls = _find_wall_pairs(vert, scan_rows=False, min_gap_px=min_gap_px, max_gap_px=max_gap_px)
    combined = cv2.bitwise_or(horiz_walls, vert_walls)
    # Snap mask (apply_margins=False) must keep long exterior wall runs intact so
    # snap_segments_to_walls can find them; raise threshold to 0.90 for that pass.
    # Footprint pass raised to 0.65 so exterior walls spanning 42–65% of the sheet
    # are not incorrectly stripped as structural grid lines.
    # ROI crops (apply_margins=False): use 0.98 for horizontal stripping only —
    # north exterior walls span 90%+ of crop width; west/east walls span 90%+
    # of crop height and must not be stripped as vertical grid lines.
    if apply_margins:
        span_frac, strip_vertical = 0.65, True
    else:
        span_frac, strip_vertical = 0.98, False
    return _strip_spanning_grid_lines(
        combined, span_frac=span_frac, strip_vertical=strip_vertical
    )


def preprocess(
    image: np.ndarray,
    blank_right_frac: Optional[float] = None,
    px_per_unit: float = 18.0,
    apply_margins: bool = True,
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

    apply_margins=False is used when the caller already cropped the image to a
    user-drawn ROI: the hard-coded sheet-fraction exclusion zones (top 12 %,
    bottom 18 %, bottom-right quadrant, …) are sized for a full sheet with a
    title block, and applied to a tight crop they erase real building walls —
    the footprint then never extends to those walls and every segment found
    there is later discarded by the polygon-bbox filter.

    Returns (big, wall_pair_mask_full).
    """
    t0 = time.time()
    # Full mask: used for snapping — exterior wall pixels near the sheet edge
    # must NOT be erased, otherwise snap_segments_to_walls finds nothing there.
    wall_pair_mask_full = _extract_wall_lines(
        image, blank_right_frac=blank_right_frac, apply_margins=False,
        px_per_unit=px_per_unit,
    )
    if apply_margins:
        # Filtered mask: used for footprint morphology (excludes sheet margins).
        wall_pair_mask_filtered = _extract_wall_lines(
            image, blank_right_frac=blank_right_frac, apply_margins=True,
            px_per_unit=px_per_unit,
        )
    else:
        wall_pair_mask_filtered = wall_pair_mask_full
    print(f"  [preprocess] wall-line extraction: {time.time()-t0:.1f}s "
          f"(margins={apply_margins})", file=sys.stderr)

    h, w = wall_pair_mask_filtered.shape
    small_h, small_w = h // DOWNSCALE, w // DOWNSCALE

    t0 = time.time()
    big = footprint_binary(wall_pair_mask_filtered, px_per_unit)
    print(f"  [preprocess] downscale morphology: {time.time()-t0:.1f}s", file=sys.stderr)

    return big, wall_pair_mask_full


def footprint_binary(
    mask_filtered: np.ndarray,
    px_per_unit: float,
    any_ink: bool = False,
) -> np.ndarray:
    """Downscale + morphology pass that turns the wall-pair mask into the solid
    footprint binary used by find_footprint.

    any_ink=False (default) keeps a downscaled cell only when it is mostly ink
    (threshold 127 after INTER_AREA). At low working resolutions the pair-mask
    strokes are only 1-2 px wide, so the averaging dilutes them below 127 and
    entire perimeter walls vanish from the footprint binary — the footprint
    then collapses to the thickest wall and every wall segment outside it is
    discarded downstream. any_ink=True keeps any cell containing ink; used as
    a retry when the strict footprint covers too little of the wall ink.
    """
    h, w = mask_filtered.shape
    small_h, small_w = h // DOWNSCALE, w // DOWNSCALE

    small = cv2.resize(mask_filtered, (small_w, small_h), interpolation=cv2.INTER_AREA)
    _, small = cv2.threshold(small, 0 if any_ink else 127, 255, cv2.THRESH_BINARY)

    small = cv2.dilate(
        small,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2,
    )

    # Adaptive closing kernel: bridge window/door gaps up to 12 real-world units.
    # At 1/8"=1'/144 DPI: 12*18/4=54; at 3/8"=1'/144 DPI: 12*54/4=162.
    close_k_size = max(25, int(12 * px_per_unit / DOWNSCALE))
    # Cap relative to the downscaled crop — at high px_per_unit an oversized
    # square close merges the entire ROI into one blob (area_frac≈1.0) and
    # find_footprint rejects it as the sheet border.
    close_k_size = min(close_k_size, min(small_w, small_h) // 6)
    close_k_size = max(close_k_size, 25)
    print(f"  [preprocess] adaptive close_k_size={close_k_size} (px_per_unit={px_per_unit:.1f})", file=sys.stderr)
    # Directional close bridges H/V doorway gaps separately; a square kernel
    # also connects distant parallel wall rows into one solid block.
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k_size, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_k_size))
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kh, iterations=1)
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kv, iterations=1)

    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

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


def find_footprint(binary: np.ndarray, use_exclusion: bool = True):
    """
    Pick the building footprint: the largest connected component that
    is NOT the sheet border/title block.

    use_exclusion=False skips the sheet-fraction exclusion-zone checks; pass
    this when the image is a user-drawn ROI crop, where those zones cover real
    building area instead of the title block.

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

            if use_exclusion:
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

    # ROI crops: building exterior walls touch the crop boundary (especially the
    # top edge).  Prefer edge-touching pass first so the footprint polygon
    # includes the north wall; full-sheet mode keeps the conservative order.
    if not use_exclusion:
        best_label = _score_components(allow_edge_touch=True)
        if best_label is None:
            best_label = _score_components(allow_edge_touch=False)
    else:
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


def angle_to_facing(normal_angle_deg: float) -> str:
    """Snap a normal direction angle (clockwise from North) to a cardinal facing."""
    normal_angle_deg = normal_angle_deg % 360
    if 315 <= normal_angle_deg or normal_angle_deg < 45:
        return "North"
    elif 45 <= normal_angle_deg < 135:
        return "East"
    elif 135 <= normal_angle_deg < 225:
        return "South"
    else:
        return "West"


def vector_to_facing(nx: float, ny: float) -> str:
    """Cardinal facing for a unit normal in image coordinates (y increases downward)."""
    angle = math.degrees(math.atan2(nx, -ny)) % 360
    return angle_to_facing(angle)


def point_in_footprint(px: float, py: float, contour: np.ndarray) -> bool:
    return cv2.pointPolygonTest(
        contour.reshape(-1, 1, 2).astype(np.float32),
        (float(px), float(py)),
        measureDist=False,
    ) >= 0


def _segment_is_horizontal(x1: int, y1: int, x2: int, y2: int) -> bool:
    return abs(x2 - x1) >= abs(y2 - y1)


def outward_facing(
    x1: int, y1: int, x2: int, y2: int,
    contour: np.ndarray,
    probe_px: float = 12.0,
) -> Optional[str]:
    """
    Facing from the outward normal: probe both sides of the segment midpoint
    and pick the direction outside the footprint.  Returns None when both
    probes land inside (interior partition).
    """
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    n1x, n1y = -dy / length, dx / length
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    inside1 = point_in_footprint(mx + probe_px * n1x, my + probe_px * n1y, contour)
    inside2 = point_in_footprint(mx - probe_px * n1x, my - probe_px * n1y, contour)
    if inside1 and not inside2:
        return vector_to_facing(-n1x, -n1y)
    if inside2 and not inside1:
        return vector_to_facing(n1x, n1y)
    return None


def _on_footprint_edge(mx: float, my: float, bbox: list[int], edge_tol_px: float) -> bool:
    x_min, y_min, x_max, y_max = bbox
    return (
        mx - x_min <= edge_tol_px
        or x_max - mx <= edge_tol_px
        or my - y_min <= edge_tol_px
        or y_max - my <= edge_tol_px
    )


def bbox_edge_facing(
    x1: int, y1: int, x2: int, y2: int,
    bbox: list[int],
    edge_tol_px: float,
) -> Optional[str]:
    """Assign facing from proximity to footprint bbox edges."""
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x_min, y_min, x_max, y_max = bbox
    if _segment_is_horizontal(x1, y1, x2, y2):
        if my - y_min <= edge_tol_px:
            return "North"
        if y_max - my <= edge_tol_px:
            return "South"
    else:
        if mx - x_min <= edge_tol_px:
            return "West"
        if x_max - mx <= edge_tol_px:
            return "East"
    return None


def _endpoints_near(seg_a: tuple, seg_b: tuple, tol: float) -> bool:
    pts_a = [(seg_a[0], seg_a[1]), (seg_a[2], seg_a[3])]
    pts_b = [(seg_b[0], seg_b[1]), (seg_b[2], seg_b[3])]
    for ax, ay in pts_a:
        for bx, by in pts_b:
            if math.hypot(ax - bx, ay - by) <= tol:
                return True
    return False


def _point_on_segment(
    px: float, py: float, seg: tuple, tol: float,
) -> bool:
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1:
        return False
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    if t < -0.02 or t > 1.02:
        return False
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y) <= tol


def build_wall_adjacency(segments: list[tuple], corner_tol_px: float) -> list[list[int]]:
    """Index segments that meet at corners or T-junctions."""
    n = len(segments)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            linked = _endpoints_near(segments[i], segments[j], corner_tol_px)
            if not linked:
                for px, py in [(segments[i][0], segments[i][1]), (segments[i][2], segments[i][3])]:
                    if _point_on_segment(px, py, segments[j], corner_tol_px):
                        linked = True
                        break
                if not linked:
                    for px, py in [(segments[j][0], segments[j][1]), (segments[j][2], segments[j][3])]:
                        if _point_on_segment(px, py, segments[i], corner_tol_px):
                            linked = True
                            break
            if linked:
                adj[i].append(j)
                adj[j].append(i)
    return adj


_HORIZ_FACINGS = frozenset({"North", "South"})
_VERT_FACINGS = frozenset({"East", "West"})


def classify_interior_facing(
    seg_idx: int,
    seg: tuple,
    adjacency: list[list[int]],
    facings: list[Optional[str]],
    footprint_bbox: list[int],
) -> str:
    """Derive facing for an interior segment from adjacent classified walls."""
    x1, y1, x2, y2 = seg
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x_min, y_min, x_max, y_max = footprint_bbox
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    is_horiz = _segment_is_horizontal(x1, y1, x2, y2)
    neighbor_facings = [
        facings[j] for j in adjacency[seg_idx]
        if facings[j] is not None
    ]

    if is_horiz:
        for f in neighbor_facings:
            if f in _HORIZ_FACINGS:
                return f
        for j in adjacency[seg_idx]:
            nf = facings[j]
            if nf in _VERT_FACINGS:
                return "North" if my < cy else "South"
        return "North" if my < cy else "South"

    for f in neighbor_facings:
        if f in _VERT_FACINGS:
            return f
    for j in adjacency[seg_idx]:
        nf = facings[j]
        if nf in _HORIZ_FACINGS:
            return "East" if mx > cx else "West"
    return "West" if mx < cx else "East"


def assign_segment_facings(
    segments: list[tuple],
    contour: np.ndarray,
    footprint_bbox: list[int],
    px_per_unit: float,
) -> list[str]:
    """Two-pass exterior (outward probe) then interior (adjacency) facing assignment."""
    probe_px = max(8.0, 0.5 * px_per_unit)
    corner_tol_px = max(8, int(0.75 * px_per_unit))
    edge_tol_px = max(12, int(1.0 * px_per_unit))

    n = len(segments)
    facings: list[Optional[str]] = [None] * n

    for i, seg in enumerate(segments):
        x1, y1, x2, y2 = seg
        facing = outward_facing(x1, y1, x2, y2, contour, probe_px)
        if facing is None:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if _on_footprint_edge(mx, my, footprint_bbox, edge_tol_px):
                facing = bbox_edge_facing(x1, y1, x2, y2, footprint_bbox, edge_tol_px)
        facings[i] = facing

    adjacency = build_wall_adjacency(segments, corner_tol_px)

    for i, seg in enumerate(segments):
        if facings[i] is None:
            facings[i] = classify_interior_facing(
                i, seg, adjacency, facings, footprint_bbox,
            )

    x_min, y_min, x_max, y_max = footprint_bbox
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    for i, seg in enumerate(segments):
        if facings[i] is None:
            x1, y1, x2, y2 = seg
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if _segment_is_horizontal(x1, y1, x2, y2):
                facings[i] = "North" if my < cy else "South"
            else:
                facings[i] = "West" if mx < cx else "East"

    return [f or "South" for f in facings]


def pixel_length(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _expand_poly_roi(
    poly_roi: dict,
    wall_mask: np.ndarray,
    img_w: int,
    img_h: int,
    pad_frac: float = 0.04,
) -> dict:
    """
    Expand footprint bbox filter to include wall-pair ink near the polygon edge.

    The footprint polygon is traced on a morphologically closed mask and can sit
    slightly inside real wall pixels — especially the north exterior run in a
    tight ROI.  Union the polygon bbox with the wall_mask ink extent so Hough
    segments on the true top edge are not clipped.
    """
    rows = np.where((wall_mask > 0).any(axis=1))[0]
    cols = np.where((wall_mask > 0).any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return poly_roi
    return {
        "x0_pct": max(0.0, min(poly_roi["x0_pct"], cols[0] / img_w - pad_frac)),
        "y0_pct": max(0.0, min(poly_roi["y0_pct"], rows[0] / img_h - pad_frac)),
        "x1_pct": min(1.0, max(poly_roi["x1_pct"], (cols[-1] + 1) / img_w + pad_frac)),
        "y1_pct": min(1.0, max(poly_roi["y1_pct"], (rows[-1] + 1) / img_h + pad_frac)),
    }


def _filter_wall_segments(
    segments: list[tuple],
    img_w: int,
    img_h: int,
    max_span_frac: float = 0.38,
    roi: Optional[dict] = None,
    min_overlap_frac: float = 0.25,
) -> list[tuple]:
    """Drop title-block/grid segments outside the building footprint."""
    kept = []
    for x1, y1, x2, y2 in segments:
        if roi:
            if _segment_overlap_frac(x1, y1, x2, y2, roi, img_w, img_h) < min_overlap_frac:
                continue
        else:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if _point_in_exclusion(mx, my, img_w, img_h):
                continue
        # Use the axis-aligned extent for span check: a north exterior wall can
        # be wider than min(w,h) in a landscape ROI crop.
        is_horiz = abs(x2 - x1) >= abs(y2 - y1)
        axis_extent = img_w if is_horiz else img_h
        if pixel_length(x1, y1, x2, y2) > axis_extent * max_span_frac:
            continue
        kept.append((x1, y1, x2, y2))
    return kept


def measure_walls(
    segments: list[tuple],
    px_per_unit: float,
    unit_label: str,
    contour: Optional[np.ndarray] = None,
    footprint_bbox: Optional[list[int]] = None,
) -> list[dict]:
    if contour is not None and footprint_bbox is not None:
        facings = assign_segment_facings(segments, contour, footprint_bbox, px_per_unit)
    else:
        facings = [
            angle_to_facing(wall_angle_deg(x1, y1, x2, y2) + 90)
            for x1, y1, x2, y2 in segments
        ]

    walls = []
    for i, (x1, y1, x2, y2) in enumerate(segments):
        px_len = pixel_length(x1, y1, x2, y2)
        real_len = px_len / px_per_unit
        angle = wall_angle_deg(x1, y1, x2, y2)
        facing = facings[i]

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

def _cluster_1d_positions(values: np.ndarray, link_px: int = 4) -> list[int]:
    """Cluster sorted 1-D ink coordinates into runs separated by > link_px.

    Returns the center of each run. A 3 px stroke yields one cluster at its
    centerline; the two strokes of a drawn wall (gap >= ~min wall gap) yield
    two clusters.
    """
    if values.size == 0:
        return []
    vals = np.unique(values)
    centers: list[int] = []
    start = prev = int(vals[0])
    for v in vals[1:]:
        v = int(v)
        if v - prev > link_px:
            centers.append((start + prev) // 2)
            start = v
        prev = v
    centers.append((start + prev) // 2)
    return centers


def _stroke_partner_stats(
    wall_mask: np.ndarray,
    seg: tuple,
    gap_lo: int,
    gap_hi: int,
    n_samples: int = 24,
) -> tuple[float, float]:
    """(ink presence, partner coverage given presence) along a stroke row/col.

    Samples n_samples points along seg. presence = fraction of samples where
    the stroke itself has ink (within +/-2 px perpendicular). partner = of
    those present samples, the fraction with a parallel partner stroke within
    [gap_lo, gap_hi] perpendicular.

    Unlike a partner check over the whole span, this stays valid for a
    polygon edge spanning a stepped perimeter (wall ink present on only part
    of the span, but fully partnered where present) while still failing
    single-stroke annotation ink (present everywhere, partnered only near
    tick marks and text).
    """
    h, w = wall_mask.shape
    x1, y1, x2, y2 = seg
    is_horiz = abs(x2 - x1) >= abs(y2 - y1)

    present = 0
    partnered = 0
    for k in range(n_samples):
        t = (k + 0.5) / n_samples
        px = int(round(x1 + t * (x2 - x1)))
        py = int(round(y1 + t * (y2 - y1)))
        if not (0 <= px < w and 0 <= py < h):
            continue
        if is_horiz:
            self_lo = max(0, py - 2)
            self_hi = min(h - 1, py + 2)
            has_self = (wall_mask[self_lo:self_hi + 1, px] > 0).any()
        else:
            self_lo = max(0, px - 2)
            self_hi = min(w - 1, px + 2)
            has_self = (wall_mask[py, self_lo:self_hi + 1] > 0).any()
        if not has_self:
            continue
        present += 1
        if is_horiz:
            lo_a = max(0, py - gap_hi)
            hi_a = max(0, py - gap_lo)
            lo_b = min(h - 1, py + gap_lo)
            hi_b = min(h - 1, py + gap_hi)
            has_partner = ((wall_mask[lo_a:hi_a + 1, px] > 0).any()
                           or (wall_mask[lo_b:hi_b + 1, px] > 0).any())
        else:
            lo_a = max(0, px - gap_hi)
            hi_a = max(0, px - gap_lo)
            lo_b = min(w - 1, px + gap_lo)
            hi_b = min(w - 1, px + gap_hi)
            has_partner = ((wall_mask[py, lo_a:hi_a + 1] > 0).any()
                           or (wall_mask[py, lo_b:hi_b + 1] > 0).any())
        if has_partner:
            partnered += 1

    if present == 0:
        return 0.0, 0.0
    return present / n_samples, partnered / present


def _snap_axis_position(
    wall_mask: np.ndarray,
    is_horiz: bool,
    center: int,
    span_lo: int,
    span_hi: int,
    radius: int,
    min_pixels: int,
    edge_frac: float,
    validate: bool,
    gap_lo: int,
    gap_hi: int,
    allow_hop: bool = False,
) -> Optional[int]:
    """Snapped perpendicular coordinate for one segment, or None if no ink.

    Legacy behavior (validate=False): median of ink coords in the search
    strip, or the outermost ink row/col near crop edges.

    With validate=True the legacy target is checked against a double-stroke
    partner test (`_stroke_partner_stats`). If the stroke cluster at the
    target passes, the snap is only refined to the centerline of its stroke
    pair (bounded by the wall gap).

    With allow_hop=True the segment may additionally move to a different ink
    cluster — the rescue for footprint-polygon edges that ride an annotation
    bulge (dimension string near the building). Because walls on some plans
    are drawn single-stroke (no partner at all), hopping is RELATIVE, not
    absolute: it requires the legacy target to be clearly unpartnered AND the
    destination to be clearly partnered (a drawn wall pair). Clusters are
    tried in the path's preference order (nearest-to-original for the median
    path, outermost-first near crop edges), retrying at double radius because
    the real wall can sit beyond the bulge. Segments already sitting on their
    own ink (Hough interiors) must never hop onto a neighboring parallel
    wall; keep allow_hop off for them.
    """
    h, w = wall_mask.shape
    limit = h if is_horiz else w
    if span_lo >= span_hi:
        return None

    def _ink_counts(grow: int) -> tuple[int, Optional[np.ndarray]]:
        """(window_start, per-row/col ink counts along the span)."""
        lo = max(0, center - radius * grow)
        hi = min(limit - 1, center + radius * grow)
        if lo >= hi:
            return lo, None
        if is_horiz:
            strip = wall_mask[lo:hi + 1, span_lo:span_hi + 1]
            counts = (strip > 0).sum(axis=1)
        else:
            strip = wall_mask[span_lo:span_hi + 1, lo:hi + 1]
            counts = (strip > 0).sum(axis=0)
        return lo, counts

    def _clusters(lo: int, counts: np.ndarray) -> list[int]:
        # Only positions carrying a significant share of ink along the span
        # can be wall strokes. Rows/cols crossed merely by perpendicular
        # walls or text carry a few pixels each; without this threshold they
        # fuse separate stroke clusters into one blob whose center lies off
        # any real wall.
        sig_thresh = max(min_pixels, int(0.2 * counts.max()))
        sig = np.where(counts >= sig_thresh)[0] + lo
        return _cluster_1d_positions(sig)

    def _stats(c: int) -> tuple[float, float]:
        test_seg = (span_lo, c, span_hi, c) if is_horiz else (c, span_lo, c, span_hi)
        return _stroke_partner_stats(wall_mask, test_seg, gap_lo, gap_hi)

    def _pair_midpoint(c: int, centers: list[int]) -> int:
        # Centerline of the stroke pair: midpoint of this stroke and the
        # farthest partner stroke on the partner side (skipping intermediate
        # strokes such as window sills).
        in_range = [o for o in centers if o != c and gap_lo <= abs(o - c) <= gap_hi]
        if not in_range:
            return c
        nearest = min(in_range, key=lambda o: abs(o - c))
        side = 1 if nearest > c else -1
        far = max((o for o in in_range if (o - c) * side > 0),
                  key=lambda o: abs(o - c))
        return (c + far) // 2

    lo, counts = _ink_counts(1)
    if counts is None or int(counts.sum()) < min_pixels:
        return None
    coords = np.repeat(np.arange(lo, lo + len(counts)), counts)

    # Near the crop top/left, snap to the outermost wall-pair position
    # (minimum) instead of the median — median lands on the inner face or a
    # dimension row inside the wall. Symmetrically for bottom/right.
    if center < limit * edge_frac:
        legacy, order = int(coords.min()), "asc"
    elif center > limit * (1 - edge_frac):
        legacy, order = int(coords.max()), "desc"
    else:
        legacy, order = int(np.median(coords)), "near"
    if not validate:
        return legacy

    centers = _clusters(lo, counts)

    # Anchor on the legacy target: if the stroke cluster it landed on is a
    # partnered wall pair, keep it (refined to the pair centerline).
    anchored = [c for c in centers if abs(c - legacy) <= gap_hi]
    anchored.sort(key=lambda v: abs(v - legacy))
    for c in anchored:
        present, partnered = _stats(c)
        if present >= 0.15 and partnered >= 0.5:
            return _pair_midpoint(c, centers)

    if not allow_hop:
        return legacy

    # Hop only to a destination that is clearly more wall-like than the
    # legacy target: strongly partnered ink (a drawn wall pair) where the
    # target is mostly unpartnered (annotation ink). The relative margin
    # protects plans whose walls are drawn single-stroke: there the
    # alternatives are unpartnered too, so nothing hops.
    #
    # Score the legacy target on the stroke cluster it sits on, not the raw
    # coordinate: coords.min()/max() can land on a sliver of tick ink whose
    # few present samples all see the adjacent stroke, yielding a meaningless
    # partnered=1.0 that would block every hop.
    leg_present, q_legacy = _stats(anchored[0] if anchored else legacy)
    if leg_present < 0.15:
        q_legacy = 0.0
    for grow in (1, 2):
        if grow > 1:
            if order != "near":
                break
            lo, counts = _ink_counts(grow)
            if counts is None or int(counts.sum()) < min_pixels:
                continue
            centers = _clusters(lo, counts)
        if order == "asc":
            centers.sort()
        elif order == "desc":
            centers.sort(reverse=True)
        else:
            centers.sort(key=lambda v: abs(v - center))
        for c in centers:
            if abs(c - legacy) <= gap_hi:
                continue
            present, partnered = _stats(c)
            if (present >= 0.3 and partnered >= 0.6
                    and partnered >= q_legacy + 0.25):
                return _pair_midpoint(c, centers)
    return legacy


def snap_segments_to_walls(
    segments: list[tuple],
    wall_mask: np.ndarray,
    search_radius: int = 80,
    min_pixels: int = 10,
    px_per_unit: float = 18.0,
    validate_pairs: bool = False,
    allow_hop: bool = False,
) -> list[tuple]:
    """
    Shift each polygon-derived segment onto the nearest actual wall-pair pixels.

    The polygon contour is traced on an inflated binary mask, so its edges sit
    outside the real wall lines.  For each segment we scan the undilated
    wall_pair_mask perpendicular to the segment direction, find the median
    position of wall pixels within search_radius, and snap the segment there.
    Falls back to the original position when no wall pixels are found.

    validate_pairs=True additionally requires the snap target to look like a
    double-stroke wall (see _snap_axis_position), skipping single-stroke
    annotation ink such as dimension strings near the building. allow_hop=True
    lets a segment whose target fails that test move to the nearest validated
    wall cluster — intended for footprint-polygon edges that ride an
    annotation bulge; keep it off for segments already sitting on their ink.
    """
    h, w = wall_mask.shape
    radius = max(search_radius, int(2 * px_per_unit))
    edge_frac = 0.12
    gap_lo, gap_hi = wall_pair_gap_range(px_per_unit)

    def _ink_extent(is_horiz: bool, axis: int, s_lo: int, s_hi: int,
                    e_lo: int, e_hi: int) -> tuple[int, int]:
        """Trim a snapped segment's span to the ink that justified the snap.

        Polygon edges can run past the building into an annotation bulge
        (e.g. down the side of a dimension-string strip merged into the
        footprint); the snapped axis has wall ink only along the real wall.
        Collapse a band of +/-gap_hi around the snapped axis and shrink the
        span ends to the first/last ink. Never expands the original span.
        """
        b_lo = max(0, axis - gap_hi)
        b_hi = min((h if is_horiz else w) - 1, axis + gap_hi)
        if is_horiz:
            band = wall_mask[b_lo:b_hi + 1, s_lo:s_hi + 1]
            profile = (band > 0).any(axis=0)
        else:
            band = wall_mask[s_lo:s_hi + 1, b_lo:b_hi + 1]
            profile = (band > 0).any(axis=1)
        idx = np.flatnonzero(profile)
        if idx.size == 0:
            return e_lo, e_hi
        ink_lo, ink_hi = s_lo + int(idx[0]), s_lo + int(idx[-1])
        return max(e_lo, ink_lo), min(e_hi, ink_hi)

    snapped = []
    for (x1, y1, x2, y2) in segments:
        is_horiz = abs(x2 - x1) >= abs(y2 - y1)
        if is_horiz:
            cy = (y1 + y2) // 2
            x_lo = max(0, min(x1, x2) + 5)
            x_hi = min(w - 1, max(x1, x2) - 5)
            ny = _snap_axis_position(
                wall_mask, True, cy, x_lo, x_hi,
                radius, min_pixels, edge_frac, validate_pairs, gap_lo, gap_hi,
                allow_hop=allow_hop,
            )
            if ny is not None:
                nx1, nx2 = x1, x2
                if validate_pairs and x_lo < x_hi:
                    t_lo, t_hi = _ink_extent(
                        True, ny, x_lo, x_hi, min(x1, x2), max(x1, x2))
                    nx1, nx2 = (t_lo, t_hi) if x1 <= x2 else (t_hi, t_lo)
                snapped.append((nx1, ny, nx2, ny))
                continue
        else:
            cx = (x1 + x2) // 2
            y_lo = max(0, min(y1, y2) + 5)
            y_hi = min(h - 1, max(y1, y2) - 5)
            nx = _snap_axis_position(
                wall_mask, False, cx, y_lo, y_hi,
                radius, min_pixels, edge_frac, validate_pairs, gap_lo, gap_hi,
                allow_hop=allow_hop,
            )
            if nx is not None:
                ny1, ny2 = y1, y2
                if validate_pairs and y_lo < y_hi:
                    t_lo, t_hi = _ink_extent(
                        False, nx, y_lo, y_hi, min(y1, y2), max(y1, y2))
                    ny1, ny2 = (t_lo, t_hi) if y1 <= y2 else (t_hi, t_lo)
                snapped.append((nx, ny1, nx, ny2))
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

        # Complete-linkage clustering: a segment joins a cluster only if the
        # cluster's full perpendicular span stays within axis_tol_px.  Single-
        # linkage allowed chain-merging walls A→B→C when A–C exceeded the tol.
        clusters: list[list[tuple]] = []
        for seg in segs:
            p = _perp(seg, is_horiz)
            placed = False
            for cluster in clusters:
                perps = [_perp(s, is_horiz) for s in cluster] + [p]
                if max(perps) - min(perps) <= axis_tol_px:
                    cluster.append(seg)
                    placed = True
                    break
            if not placed:
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


def _coaxial_wall_seg(w: dict) -> tuple[int, int, int, int]:
    c = w["px_coords"]
    return (c[0], c[1], c[2], c[3])


def _union_interval_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged = [intervals[0]]
    for lo, hi in sorted(intervals[1:]):
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return sum(hi - lo for lo, hi in merged)


def _perp_band_tol(px_per_unit: float, axis_tol_px: int) -> int:
    """Perpendicular tolerance for grouping parallel wall strokes / dimension bands."""
    return max(axis_tol_px * 2, int(4.0 * px_per_unit))


def _wall_is_horiz(w: dict) -> bool:
    c = w["px_coords"]
    return abs(c[2] - c[0]) >= abs(c[3] - c[1])


def _wall_perp(seg: tuple, horiz: bool) -> int:
    return (seg[1] + seg[3]) // 2 if horiz else (seg[0] + seg[2]) // 2


def _wall_proj(seg: tuple, horiz: bool) -> tuple[int, int]:
    if horiz:
        return min(seg[0], seg[2]), max(seg[0], seg[2])
    return min(seg[1], seg[3]), max(seg[1], seg[3])


def coaxial_spanning_wall_indices(
    walls: list[dict],
    axis_tol_px: int,
    cover_frac: float = 0.85,
    perp_band_tol: Optional[int] = None,
) -> set[int]:
    """Indices of walls tiled along the same axis by at least two other segments."""
    if len(walls) < 2:
        return set()

    band_tol = perp_band_tol if perp_band_tol is not None else axis_tol_px

    drop_indices: set[int] = set()

    for is_horiz in (True, False):
        indexed = [
            (i, _coaxial_wall_seg(w))
            for i, w in enumerate(walls)
            if _wall_is_horiz(w) == is_horiz
        ]
        if len(indexed) < 2:
            continue

        indexed.sort(key=lambda t: _wall_perp(t[1], is_horiz))

        clusters: list[list[tuple[int, tuple]]] = []
        for idx, seg in indexed:
            p = _wall_perp(seg, is_horiz)
            placed = False
            for cluster in clusters:
                perps = [_wall_perp(s, is_horiz) for _, s in cluster] + [p]
                if max(perps) - min(perps) <= band_tol:
                    cluster.append((idx, seg))
                    placed = True
                    break
            if not placed:
                clusters.append([(idx, seg)])

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            for i, seg_l in cluster:
                lo_l, hi_l = _wall_proj(seg_l, is_horiz)
                len_l = hi_l - lo_l
                if len_l < 1:
                    continue

                contributor_intervals: list[tuple[int, int]] = []
                max_contrib_len = 0
                for j, seg_j in cluster:
                    if i == j:
                        continue
                    lo_j, hi_j = _wall_proj(seg_j, is_horiz)
                    overlap_lo = max(lo_l, lo_j)
                    overlap_hi = min(hi_l, hi_j)
                    if overlap_hi <= overlap_lo:
                        continue
                    contributor_intervals.append((overlap_lo, overlap_hi))
                    max_contrib_len = max(max_contrib_len, hi_j - lo_j)

                if len(contributor_intervals) < 2:
                    continue
                if max_contrib_len >= len_l:
                    continue

                covered = _union_interval_length(contributor_intervals)
                threshold = cover_frac
                if walls[i].get("is_exterior") and walls[i]["length_raw"] >= 40:
                    threshold = min(cover_frac, 0.55)
                elif not walls[i].get("is_exterior") and walls[i]["length_raw"] >= 50:
                    threshold = min(cover_frac, 0.45)
                if covered / len_l >= threshold:
                    drop_indices.add(i)

    return drop_indices


def drop_spanning_coaxial_walls(
    walls: list[dict],
    axis_tol_px: int,
    cover_frac: float = 0.85,
    px_per_unit: Optional[float] = None,
) -> list[dict]:
    """Drop long coaxial walls when shorter segments already tile the same run."""
    band_tol = _perp_band_tol(px_per_unit, axis_tol_px) if px_per_unit else axis_tol_px
    drop_indices = coaxial_spanning_wall_indices(
        walls, axis_tol_px, cover_frac, perp_band_tol=band_tol,
    )
    if not drop_indices:
        return walls
    return [w for i, w in enumerate(walls) if i not in drop_indices]


def drop_duplicate_exterior_strokes(
    walls: list[dict],
    px_per_unit: float,
) -> list[dict]:
    """
    Drop parallel exterior polygon strokes (double-drawn wall ink).

    The footprint polygon often traces both faces of a double-line wall as
    separate parents.  Keep the parent with the most room splits; drop the
    others when their spans overlap on the same side.
    """
    stroke_tol = max(int(2.5 * px_per_unit), 40)
    exterior = [w for w in walls if w.get("is_exterior")]
    if len(exterior) < 2:
        return walls

    drop_parents: set[str] = set()

    for facing in ("North", "South", "East", "West"):
        by_parent: dict[str, list[dict]] = {}
        for w in exterior:
            if w.get("facing") != facing:
                continue
            pid = w.get("parent_wall_id") or w["id"]
            by_parent.setdefault(pid, []).append(w)

        if len(by_parent) < 2:
            continue

        parent_info = []
        for pid, segs in by_parent.items():
            horiz = _wall_is_horiz(segs[0])
            perps = [_wall_perp(_coaxial_wall_seg(s), horiz) for s in segs]
            lo = min(_wall_proj(_coaxial_wall_seg(s), horiz)[0] for s in segs)
            hi = max(_wall_proj(_coaxial_wall_seg(s), horiz)[1] for s in segs)
            max_seg = max(s["length_raw"] for s in segs)
            parent_info.append({
                "pid": pid,
                "perp": sum(perps) / len(perps),
                "lo": lo,
                "hi": hi,
                "n_segs": len(segs),
                "max_seg": max_seg,
                "horiz": horiz,
            })

        n = len(parent_info)
        parent = list(range(n))

        def _find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if parent_info[i]["horiz"] != parent_info[j]["horiz"]:
                    continue
                if abs(parent_info[i]["perp"] - parent_info[j]["perp"]) > stroke_tol:
                    continue
                # Double-drawn faces of the same wall overlap along their
                # axis; same-facing parents at close perp but disjoint spans
                # are distinct steps of the perimeter, not duplicates.
                overlap = (min(parent_info[i]["hi"], parent_info[j]["hi"])
                           - max(parent_info[i]["lo"], parent_info[j]["lo"]))
                shorter = min(parent_info[i]["hi"] - parent_info[i]["lo"],
                              parent_info[j]["hi"] - parent_info[j]["lo"])
                if overlap >= max(20, 0.3 * shorter):
                    _union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(_find(i), []).append(i)

        for members in clusters.values():
            if len(members) < 2:
                continue
            best = max(
                members,
                key=lambda k: (
                    parent_info[k]["n_segs"],
                    -(parent_info[k]["max_seg"]),
                ),
            )
            for k in members:
                if k != best:
                    drop_parents.add(parent_info[k]["pid"])

    if not drop_parents:
        return walls
    return [
        w for w in walls
        if not (w.get("is_exterior") and (w.get("parent_wall_id") or w["id"]) in drop_parents)
    ]


def consolidate_coaxial_wall_duplicates(
    walls: list[dict],
    axis_tol_px: int,
    px_per_unit: float,
    unit_label: str,
) -> list[dict]:
    """Merge overlapping coaxial walls (double ink strokes) into a single entry."""
    if len(walls) < 2:
        return walls

    merge_tol = max(axis_tol_px + 6, int(0.8 * px_per_unit))
    merged_flags = [False] * len(walls)
    result: list[dict] = []

    for is_horiz in (True, False):
        indices = [i for i, w in enumerate(walls) if _wall_is_horiz(w) == is_horiz]
        if len(indices) < 2:
            continue

        indices.sort(key=lambda i: _wall_perp(_coaxial_wall_seg(walls[i]), is_horiz))

        clusters: list[list[int]] = []
        for i in indices:
            p = _wall_perp(_coaxial_wall_seg(walls[i]), is_horiz)
            placed = False
            for cluster in clusters:
                perps = [_wall_perp(_coaxial_wall_seg(walls[k]), is_horiz) for k in cluster] + [p]
                if max(perps) - min(perps) <= merge_tol:
                    cluster.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            remaining = [i for i in cluster if not merged_flags[i]]
            while len(remaining) >= 2:
                remaining.sort(
                    key=lambda i: _wall_proj(_coaxial_wall_seg(walls[i]), is_horiz)[0]
                )
                base = remaining[0]
                base_seg = _coaxial_wall_seg(walls[base])
                lo_b, hi_b = _wall_proj(base_seg, is_horiz)
                merged_any = False
                for other in remaining[1:]:
                    if merged_flags[other]:
                        continue
                    if walls[base].get("is_exterior") != walls[other].get("is_exterior"):
                        continue
                    lo_o, hi_o = _wall_proj(_coaxial_wall_seg(walls[other]), is_horiz)
                    overlap = min(hi_b, hi_o) - max(lo_b, lo_o)
                    min_len = max(1, min(hi_b - lo_b, hi_o - lo_o))
                    if overlap / min_len < 0.45:
                        continue

                    lo_b, hi_b = min(lo_b, lo_o), max(hi_b, hi_o)
                    pick = base
                    if walls[other].get("is_exterior") and not walls[base].get("is_exterior"):
                        pick = other
                    elif walls[other]["length_raw"] > walls[base]["length_raw"]:
                        if not walls[base].get("is_exterior") or walls[other].get("is_exterior"):
                            pick = other

                    perp = _wall_perp(_coaxial_wall_seg(walls[pick]), is_horiz)
                    if is_horiz:
                        new_coords = [lo_b, perp, hi_b, perp]
                    else:
                        new_coords = [perp, lo_b, perp, hi_b]

                    px_len = pixel_length(*new_coords)
                    real_len = px_len / px_per_unit
                    walls[pick]["px_coords"] = new_coords
                    walls[pick]["length_raw"] = round(real_len, 2)
                    walls[pick]["length"] = f"{real_len:.2f} {unit_label}"
                    walls[pick]["angle_deg"] = round(
                        wall_angle_deg(*new_coords), 1
                    )

                    merged_flags[other] = True
                    base = pick
                    merged_any = True

                remaining = [i for i in cluster if not merged_flags[i]]
                if not merged_any:
                    break

    for i, w in enumerate(walls):
        if not merged_flags[i]:
            result.append(w)
    return result


def drop_dimension_like_walls(
    walls: list[dict],
    axis_tol_px: int,
    px_per_unit: float,
) -> list[dict]:
    """
    Drop interior segments parallel to a nearby wall at dimension-line offset.

    Dimension extension lines sit just outside a wall (typically 1–2 ft away)
    and overlap the wall run; they are not structural walls.
    """
    offset_min = axis_tol_px + 1
    offset_max = max(int(2.5 * px_per_unit), axis_tol_px * 2)
    drop_indices: set[int] = set()

    for i, w_i in enumerate(walls):
        if w_i.get("is_exterior"):
            continue
        if w_i["length_raw"] >= 14.0:
            continue
        seg_i = _coaxial_wall_seg(w_i)
        horiz_i = _wall_is_horiz(w_i)
        lo_i, hi_i = _wall_proj(seg_i, horiz_i)
        len_i = max(1, hi_i - lo_i)
        perp_i = _wall_perp(seg_i, horiz_i)

        for j, w_j in enumerate(walls):
            if i == j:
                continue
            if w_j.get("facing") != w_i.get("facing"):
                continue
            seg_j = _coaxial_wall_seg(w_j)
            if _wall_is_horiz(w_j) != horiz_i:
                continue
            perp_j = _wall_perp(seg_j, horiz_i)
            perp_dist = abs(perp_i - perp_j)
            if perp_dist < offset_min or perp_dist > offset_max:
                continue
            lo_j, hi_j = _wall_proj(seg_j, horiz_i)
            overlap = min(hi_i, hi_j) - max(lo_i, lo_j)
            if overlap <= 0:
                continue
            min_len = min(len_i, max(1, hi_j - lo_j))
            if overlap / min_len >= 0.35:
                drop_indices.add(i)
                break

    if not drop_indices:
        return walls
    return [w for i, w in enumerate(walls) if i not in drop_indices]


def drop_redundant_exterior_spans(
    walls: list[dict],
    px_per_unit: float,
    axis_tol_px: int,
    min_span_ft: float = 30.0,
    cover_frac: float = 0.50,
) -> list[dict]:
    """
    Drop single-run exterior walls when shorter same-side segments already
    cover the same physical wall run (common when polygon + Hough disagree).
    """
    band_tol = _perp_band_tol(px_per_unit, axis_tol_px)
    drop_indices: set[int] = set()

    for i, w_long in enumerate(walls):
        if not w_long.get("is_exterior"):
            continue
        if (w_long.get("segment_count") or 1) > 1:
            continue
        if w_long["length_raw"] < min_span_ft:
            continue

        seg_l = _coaxial_wall_seg(w_long)
        horiz = _wall_is_horiz(w_long)
        lo_l, hi_l = _wall_proj(seg_l, horiz)
        len_l = hi_l - lo_l
        if len_l < 1:
            continue
        perp_l = _wall_perp(seg_l, horiz)

        contributor_intervals: list[tuple[int, int]] = []
        for j, w_short in enumerate(walls):
            if i == j or w_short.get("facing") != w_long.get("facing"):
                continue
            if _wall_is_horiz(w_short) != horiz:
                continue
            if w_short["length_raw"] >= w_long["length_raw"]:
                continue
            seg_s = _coaxial_wall_seg(w_short)
            if abs(_wall_perp(seg_s, horiz) - perp_l) > band_tol:
                continue
            lo_s, hi_s = _wall_proj(seg_s, horiz)
            overlap_lo = max(lo_l, lo_s)
            overlap_hi = min(hi_l, hi_s)
            if overlap_hi > overlap_lo:
                contributor_intervals.append((overlap_lo, overlap_hi))

        if len(contributor_intervals) < 2:
            continue
        if _union_interval_length(contributor_intervals) / len_l >= cover_frac:
            drop_indices.add(i)

    if not drop_indices:
        return walls
    return [w for i, w in enumerate(walls) if i not in drop_indices]


def _wall_list_length_ft(walls: list[dict]) -> float:
    return sum(float(w.get("length_raw") or 0) for w in walls)


def _run_cleanup_pass(
    walls: list[dict],
    pass_name: str,
    apply_fn,
    stats: dict[str, int],
    audit: dict,
    *,
    max_drop_frac: float = 0.40,
) -> list[dict]:
    """Apply one cleanup pass; skip if it would remove more than max_drop_frac of total length."""
    before_ids = {w["id"] for w in walls}
    before_len = _wall_list_length_ft(walls)
    before_count = len(walls)
    after = apply_fn(walls)
    dropped_ids = before_ids - {w["id"] for w in after}
    after_len = _wall_list_length_ft(after)
    dropped_len = max(0.0, before_len - after_len)

    if before_len > 0 and dropped_len / before_len > max_drop_frac:
        stats[f"{pass_name}_skipped"] = len(dropped_ids)
        audit.setdefault("skipped", []).append({
            "pass": pass_name,
            "would_drop": len(dropped_ids),
            "drop_frac": round(dropped_len / before_len, 4),
        })
        stats[pass_name] = 0
        return walls

    stats[pass_name] = before_count - len(after)
    if dropped_ids:
        audit.setdefault("drops", []).extend(
            {"id": wid, "pass": pass_name} for wid in sorted(dropped_ids)
        )
    return after


def cleanup_wall_list(
    walls: list[dict],
    axis_tol_px: int,
    px_per_unit: float,
    unit_label: str,
    *,
    audit: bool = False,
    max_drop_frac: float = 0.40,
) -> tuple[list[dict], dict[str, int]] | tuple[list[dict], dict[str, int], dict]:
    """Run post-merge wall cleanup passes; return walls and per-pass drop counts.

    When ``audit=True``, also return an audit dict with per-wall drop reasons
    and any passes skipped by the length-conservation guard.
    """
    stats: dict[str, int] = {}
    trail: dict = {"drops": [], "skipped": []}

    walls = _run_cleanup_pass(
        walls, "duplicate_exterior_strokes",
        lambda w: drop_duplicate_exterior_strokes(w, px_per_unit),
        stats, trail, max_drop_frac=max_drop_frac,
    )
    walls = _run_cleanup_pass(
        walls, "dimension_like",
        lambda w: drop_dimension_like_walls(w, axis_tol_px, px_per_unit),
        stats, trail, max_drop_frac=max_drop_frac,
    )
    walls = _run_cleanup_pass(
        walls, "spanning",
        lambda w: drop_spanning_coaxial_walls(w, axis_tol_px, px_per_unit=px_per_unit),
        stats, trail, max_drop_frac=max_drop_frac,
    )
    walls = _run_cleanup_pass(
        walls, "coaxial_merge",
        lambda w: consolidate_coaxial_wall_duplicates(
            w, axis_tol_px, px_per_unit, unit_label,
        ),
        stats, trail, max_drop_frac=max_drop_frac,
    )
    walls = _run_cleanup_pass(
        walls, "exterior_span",
        lambda w: drop_redundant_exterior_spans(w, px_per_unit, axis_tol_px),
        stats, trail, max_drop_frac=max_drop_frac,
    )
    walls = _run_cleanup_pass(
        walls, "spanning_final",
        lambda w: drop_spanning_coaxial_walls(w, axis_tol_px, px_per_unit=px_per_unit),
        stats, trail, max_drop_frac=max_drop_frac,
    )
    walls = _run_cleanup_pass(
        walls, "coaxial_merge_final",
        lambda w: consolidate_coaxial_wall_duplicates(
            w, axis_tol_px, px_per_unit, unit_label,
        ),
        stats, trail, max_drop_frac=max_drop_frac,
    )

    if audit:
        return walls, stats, trail
    return walls, stats


def snap_wall_endpoints(
    walls: list[dict],
    px_per_unit: float,
    unit_label: str,
) -> tuple[list[dict], int]:
    """Close dangling wall endpoints onto nearby perpendicular walls.

    Wall segments are extracted independently (polygon splits, Hough runs),
    so endpoints routinely stop a few pixels short of the wall they visually
    meet — the dominant cause of the low wall-network closure rate. For each
    endpoint of an axis-aligned wall, find the nearest perpendicular wall
    whose axis lies within snap_tol and whose span covers the endpoint (with
    snap_tol slack, which also covers L-corners), then move the endpoint onto
    that axis — extending short stops and trimming overshoots alike.

    Only perpendicular targets are considered: two collinear endpoints facing
    each other across a gap are a doorway and must never be bridged. Movement
    is bounded by snap_tol (~2 ft — sub-segment trims leave corner gaps up to
    ~1.6 ft) and skipped when it would degenerate the segment. Diagonal walls
    are left untouched.

    Returns the updated wall list and the number of endpoints moved.
    """
    snap_tol = max(12, int(round(2.0 * px_per_unit)))
    min_len = max(4, int(round(0.25 * px_per_unit)))

    def _orient(coords: list) -> Optional[bool]:
        x1, y1, x2, y2 = coords
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dy <= max(2, 0.05 * dx):
            return True          # horizontal
        if dx <= max(2, 0.05 * dy):
            return False         # vertical
        return None              # diagonal: leave untouched

    # Snapshot targets so results don't depend on processing order. Moving an
    # endpoint slides it along its own wall's axis, so target axes are stable
    # by construction; spans are taken from the snapshot.
    targets = []  # (is_horiz, axis, span_lo, span_hi)
    for w in walls:
        x1, y1, x2, y2 = w["px_coords"]
        horiz = _orient(w["px_coords"])
        if horiz is None:
            targets.append(None)
        elif horiz:
            targets.append((True, (y1 + y2) // 2, min(x1, x2), max(x1, x2)))
        else:
            targets.append((False, (x1 + x2) // 2, min(y1, y2), max(y1, y2)))

    moved = 0
    for wi, w in enumerate(walls):
        horiz = _orient(w["px_coords"])
        if horiz is None:
            continue
        coords = list(w["px_coords"])
        for end in (0, 1):
            ex, ey = coords[2 * end], coords[2 * end + 1]
            ox, oy = coords[2 * (1 - end)], coords[2 * (1 - end) + 1]
            along, other_along = (ex, ox) if horiz else (ey, oy)
            perp = ey if horiz else ex

            best = None
            for ti, tgt in enumerate(targets):
                if ti == wi or tgt is None or tgt[0] == horiz:
                    continue
                _, axis, lo, hi = tgt
                d = abs(axis - along)
                if d > snap_tol or d == 0:
                    continue
                if not (lo - snap_tol <= perp <= hi + snap_tol):
                    continue
                if best is None or d < best[1]:
                    best = (axis, d)
            if best is None:
                continue
            new_along = best[0]
            # Keep direction and a non-degenerate length.
            if (other_along - new_along) * (other_along - along) <= 0:
                continue
            if abs(other_along - new_along) < min_len:
                continue
            coords[2 * end] = new_along if horiz else ex
            coords[2 * end + 1] = ey if horiz else new_along
            moved += 1
        if coords != list(w["px_coords"]):
            x1, y1, x2, y2 = coords
            real_len = pixel_length(x1, y1, x2, y2) / px_per_unit
            w["px_coords"] = [int(x1), int(y1), int(x2), int(y2)]
            w["length"] = f"{real_len:.2f} {unit_label}"
            w["length_raw"] = round(real_len, 2)
    return walls, moved


def _has_parallel_partner(
    wall_mask: np.ndarray,
    seg: tuple,
    min_gap_px: int,
    max_gap_px: int,
    n_samples: int = 20,
    min_coverage: float = 0.5,
) -> bool:
    """
    True if seg looks like one face of a double-line wall: sampling points
    along the segment, a parallel mask stroke must exist within
    [min_gap_px, max_gap_px] perpendicular to the segment at >= min_coverage
    of samples. Single-stroke annotation lines (dimension strings, leaders,
    grid lines) have no such partner and fail this check.
    """
    h, w = wall_mask.shape
    x1, y1, x2, y2 = seg
    is_horiz = abs(x2 - x1) >= abs(y2 - y1)

    hits = 0
    for k in range(n_samples):
        t = (k + 0.5) / n_samples
        px = int(round(x1 + t * (x2 - x1)))
        py = int(round(y1 + t * (y2 - y1)))
        if is_horiz:
            lo_a = max(0, py - max_gap_px)
            hi_a = max(0, py - min_gap_px)
            lo_b = min(h - 1, py + min_gap_px)
            hi_b = min(h - 1, py + max_gap_px)
            above = wall_mask[lo_a:hi_a + 1, px]
            below = wall_mask[lo_b:hi_b + 1, px]
        else:
            lo_a = max(0, px - max_gap_px)
            hi_a = max(0, px - min_gap_px)
            lo_b = min(w - 1, px + min_gap_px)
            hi_b = min(w - 1, px + max_gap_px)
            above = wall_mask[py, lo_a:hi_a + 1]
            below = wall_mask[py, lo_b:hi_b + 1]
        if (above > 0).any() or (below > 0).any():
            hits += 1

    return hits >= n_samples * min_coverage


def _short_run_supplement(
    wall_mask: np.ndarray,
    existing_segments: list[tuple],
    min_length_px: int,
    max_length_px: int,
    dedup_tol_px: int,
    pair_gap_range: tuple[int, int],
) -> list[tuple]:
    """Find short axis-aligned wall runs that probabilistic Hough misses.

    HoughLinesP reliably skips sub-6-unit runs in a full sheet (long lines
    dominate the random sampling), so closet/alcove partitions never become
    candidates. This detects them directly: directional morphological opening
    keeps only runs >= min_length_px, connected components give one candidate
    per stroke, and the same wall-pair validation as the Hough pass rejects
    single-stroke annotation ink. Runs longer than max_length_px are skipped —
    the main Hough pass already covers those.
    """
    def _is_dup(seg: tuple) -> bool:
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
            else:
                if abs((nx1 + nx2) / 2 - (ex1 + ex2) / 2) > dedup_tol_px:
                    continue
                n_lo, n_hi = min(ny1, ny2), max(ny1, ny2)
                e_lo, e_hi = min(ey1, ey2), max(ey1, ey2)
            overlap = min(n_hi, e_hi) - max(n_lo, e_lo)
            if n_hi > n_lo and overlap / (n_hi - n_lo) >= 0.30:
                return True
        return False

    out: list[tuple] = []
    for horiz in (True, False):
        ksize = (int(min_length_px), 1) if horiz else (1, int(min_length_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        opened = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        for i in range(1, n):
            x, y, w, h = (int(stats[i][k]) for k in range(4))
            length = w if horiz else h
            if length < min_length_px or length > max_length_px:
                continue
            if horiz:
                seg = (x, y + h // 2, x + w - 1, y + h // 2)
            else:
                seg = (x + w // 2, y, x + w // 2, y + h - 1)
            if _is_dup(seg):
                continue
            if not _has_parallel_partner(
                wall_mask, seg, pair_gap_range[0], pair_gap_range[1]
            ):
                continue
            out.append(seg)
    return out


def _short_run_annotation_like(
    seg: tuple,
    others: list[tuple],
    axis_tol_px: int,
    offset_max_px: int,
) -> bool:
    """True when seg is parallel to a nearby segment at dimension-line offset.

    Fixture outlines and dimension extension ticks ride 1–2.5 units off a real
    wall; a recovered short run in that band is annotation, not structure —
    and if kept it would make cleanup's dimension-like pass misclassify the
    real wall next to it.
    """
    x1, y1, x2, y2 = seg
    horiz = abs(x2 - x1) >= abs(y2 - y1)
    if horiz:
        perp, lo, hi = (y1 + y2) / 2.0, min(x1, x2), max(x1, x2)
    else:
        perp, lo, hi = (x1 + x2) / 2.0, min(y1, y2), max(y1, y2)
    for o in others:
        if o == seg:
            continue
        ox1, oy1, ox2, oy2 = o
        o_horiz = abs(ox2 - ox1) >= abs(oy2 - oy1)
        if o_horiz != horiz:
            continue
        if horiz:
            o_perp, o_lo, o_hi = (oy1 + oy2) / 2.0, min(ox1, ox2), max(ox1, ox2)
        else:
            o_perp, o_lo, o_hi = (ox1 + ox2) / 2.0, min(oy1, oy2), max(oy1, oy2)
        d = abs(perp - o_perp)
        if d <= axis_tol_px or d > offset_max_px:
            continue
        if min(hi, o_hi) - max(lo, o_lo) > 0:
            return True
    return False


def _t_junctions_into(seg: tuple, accepted: list[tuple], near_tol: float) -> bool:
    """True when either endpoint of seg lies on a perpendicular accepted segment.

    Used by the short-partition recovery pass: a closet/alcove wall is only
    believable when it terminates on a wall the pipeline already trusts.
    """
    x1, y1, x2, y2 = seg
    horiz = abs(x2 - x1) >= abs(y2 - y1)
    for ax1, ay1, ax2, ay2 in accepted:
        a_horiz = abs(ax2 - ax1) >= abs(ay2 - ay1)
        if a_horiz == horiz:
            continue
        if a_horiz:
            axis = (ay1 + ay2) / 2.0
            lo, hi = min(ax1, ax2), max(ax1, ax2)
            for ex, ey in ((x1, y1), (x2, y2)):
                if abs(ey - axis) <= near_tol and lo - near_tol <= ex <= hi + near_tol:
                    return True
        else:
            axis = (ax1 + ax2) / 2.0
            lo, hi = min(ay1, ay2), max(ay1, ay2)
            for ex, ey in ((x1, y1), (x2, y2)):
                if abs(ex - axis) <= near_tol and lo - near_tol <= ey <= hi + near_tol:
                    return True
    return False


def _hough_supplement(
    wall_mask: np.ndarray,
    existing_segments: list[tuple],
    min_length_px: float = 60,
    max_gap_px: int = 20,
    dedup_tol_px: int = 22,
    pair_gap_range: Optional[tuple[int, int]] = None,
    fates: Optional[list] = None,
) -> list[tuple]:
    """
    Run HoughLinesP on wall_mask to find wall segments missed by the polygon approach.

    The polygon-based pipeline only extracts the building footprint's outer edges; any
    interior walls and recesses smoothed away by approxPolyDP are invisible to it.
    This supplement detects all significant H/V runs in the wall_pair_mask and adds any
    that are not already represented in existing_segments.

    pair_gap_range, when given as (min_gap_px, max_gap_px), enables a wall-pair
    validation: each candidate must have a parallel partner stroke in the
    undilated wall_mask within that perpendicular gap range along most of its
    length, rejecting single-stroke dimension/annotation lines.

    fates, when given an empty list, is filled with (segment, fate) tuples for
    every raw Hough candidate: 'accepted', 'rejected-non-orthogonal',
    'rejected-duplicate', or 'rejected-no-pair'. Used by debug tooling.

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
    rejected_no_pair = 0

    for line in lines:
        # Cast to Python int immediately — numpy.intc from HoughLinesP is not JSON-serializable.
        x1, y1, x2, y2 = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])

        # Orthogonal check: within 10° of H or V axis.
        angle = abs(math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1))))
        is_horiz = angle < 10
        is_vert = angle > 80
        if not (is_horiz or is_vert):
            if fates is not None:
                fates.append(((x1, y1, x2, y2), "rejected-non-orthogonal"))
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
        if _is_duplicate(seg):
            if fates is not None:
                fates.append((seg, "rejected-duplicate"))
            continue
        if pair_gap_range is not None and not _has_parallel_partner(
            wall_mask, seg, pair_gap_range[0], pair_gap_range[1]
        ):
            rejected_no_pair += 1
            if fates is not None:
                fates.append((seg, "rejected-no-pair"))
            continue
        accepted.append(seg)
        all_check.append(seg)
        if fates is not None:
            fates.append((seg, "accepted"))

    if rejected_no_pair:
        print(f"  [hough] rejected {rejected_no_pair} single-stroke segments "
              f"(no parallel wall partner)", file=sys.stderr)
    return accepted


def detect_wall_at_point(
    wall_pair_mask: np.ndarray,
    x_px: int,
    y_px: int,
    search_radius: int = 150,
    min_run_px: int = 15,
    contour: Optional[np.ndarray] = None,
    footprint_bbox: Optional[list[int]] = None,
    px_per_unit: float = 18.0,
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
        px_coords = [x1, wall_row, x2, wall_row]
        if contour is not None and footprint_bbox is not None:
            facing = outward_facing(x1, wall_row, x2, wall_row, contour) or bbox_edge_facing(
                x1, wall_row, x2, wall_row, footprint_bbox,
                max(12, int(1.0 * px_per_unit)),
            ) or ("North" if wall_row < h // 2 else "South")
        else:
            facing = "North" if wall_row < h // 2 else "South"
        return {"px_coords": px_coords, "facing": facing}
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
        px_coords = [wall_col, y1, wall_col, y2]
        if contour is not None and footprint_bbox is not None:
            facing = outward_facing(wall_col, y1, wall_col, y2, contour) or bbox_edge_facing(
                wall_col, y1, wall_col, y2, footprint_bbox,
                max(12, int(1.0 * px_per_unit)),
            ) or ("West" if wall_col < w // 2 else "East")
        else:
            facing = "West" if wall_col < w // 2 else "East"
        return {"px_coords": px_coords, "facing": facing}


# ─── Main pipeline ──────────────────────────────────────────────────────────

def _pad_roi(roi: dict, pad_frac: float = 0.04) -> dict:
    """Symmetric fractional pad, clamped to [0, 1]."""
    return {
        "x0_pct": max(0.0, roi["x0_pct"] - pad_frac),
        "y0_pct": max(0.0, roi["y0_pct"] - pad_frac),
        "x1_pct": min(1.0, roi["x1_pct"] + pad_frac),
        "y1_pct": min(1.0, roi["y1_pct"] + pad_frac),
        "method": roi.get("method", "user-roi"),
    }


def _union_roi(a: dict, b: dict) -> dict:
    """Axis-aligned union of two fractional ROIs."""
    return {
        "x0_pct": min(a["x0_pct"], b["x0_pct"]),
        "y0_pct": min(a["y0_pct"], b["y0_pct"]),
        "x1_pct": max(a["x1_pct"], b["x1_pct"]),
        "y1_pct": max(a["y1_pct"], b["y1_pct"]),
    }


def _expand_roi_from_hint(
    image: np.ndarray,
    roi_hint: dict,
    px_per_unit: float,
    pad_frac: float = 0.04,
) -> dict:
    """
    Expand a user-drawn ROI hint to the full building ink envelope.

    Scans wall-pair ink in the hint's horizontal band so a box drawn on the
    office block still includes the warehouse below (TRDI). Title block ink is
    suppressed via apply_margins=True on the scout preprocess pass.
    """
    full_h, full_w = image.shape[:2]
    # Do not blank sheet margins here — the south warehouse wall often sits in
    # the bottom margin band and apply_margins would clip it before we crop.
    _, wall_mask = preprocess(image, px_per_unit=px_per_unit, apply_margins=False)

    pad_x = int(full_w * pad_frac)
    pad_y = int(full_h * pad_frac)
    sx0 = max(0, int(roi_hint["x0_pct"] * full_w) - pad_x)
    sx1 = min(full_w, int(roi_hint["x1_pct"] * full_w) + pad_x)

    band = wall_mask[:, sx0:sx1]
    ink_rows = np.where((band > 0).any(axis=1))[0]
    if len(ink_rows) == 0:
        print(
            "  [pipeline] roi expand: no wall ink in hint x-band — using padded hint",
            file=sys.stderr,
        )
        return _pad_roi(roi_hint, pad_frac)

    by0 = max(0, int(ink_rows[0]) - pad_y)
    by1 = min(full_h, int(ink_rows[-1]) + 1 + pad_y)

    sub = wall_mask[by0:by1, sx0:sx1]
    ink_cols = np.where((sub > 0).any(axis=0))[0]
    if len(ink_cols) == 0:
        return _pad_roi(roi_hint, pad_frac)

    bx0 = max(0, sx0 + int(ink_cols[0]) - pad_x)
    bx1 = min(full_w, sx0 + int(ink_cols[-1]) + 1 + pad_x)

    expanded = {
        "x0_pct": bx0 / full_w,
        "y0_pct": by0 / full_h,
        "x1_pct": bx1 / full_w,
        "y1_pct": by1 / full_h,
        "method": "auto-expanded",
    }
    print(
        f"  [pipeline] roi expand: hint "
        f"({roi_hint['x0_pct']:.0%}–{roi_hint['x1_pct']:.0%} × "
        f"{roi_hint['y0_pct']:.0%}–{roi_hint['y1_pct']:.0%}) → "
        f"({expanded['x0_pct']:.0%}–{expanded['x1_pct']:.0%} × "
        f"{expanded['y0_pct']:.0%}–{expanded['y1_pct']:.0%})",
        file=sys.stderr,
    )
    return expanded


def _roi_full_to_crop(
    roi_full: dict,
    roi_offset: tuple[int, int],
    full_w: int,
    full_h: int,
    crop_w: int,
    crop_h: int,
) -> dict:
    """Map a full-image fractional ROI into crop-image fractional coords."""
    ox, oy = roi_offset
    return {
        "x0_pct": max(0.0, (roi_full["x0_pct"] * full_w - ox) / crop_w),
        "y0_pct": max(0.0, (roi_full["y0_pct"] * full_h - oy) / crop_h),
        "x1_pct": min(1.0, (roi_full["x1_pct"] * full_w - ox) / crop_w),
        "y1_pct": min(1.0, (roi_full["y1_pct"] * full_h - oy) / crop_h),
    }


def _segment_overlap_frac(
    x1: float, y1: float, x2: float, y2: float,
    roi: dict, img_w: int, img_h: int,
) -> float:
    """Fraction of segment length inside an axis-aligned fractional ROI."""
    rx0 = roi["x0_pct"] * img_w
    rx1 = roi["x1_pct"] * img_w
    ry0 = roi["y0_pct"] * img_h
    ry1 = roi["y1_pct"] * img_h
    seg_len = pixel_length(x1, y1, x2, y2)
    if seg_len < 1:
        return 0.0
    if abs(x2 - x1) >= abs(y2 - y1):
        sy = (y1 + y2) / 2
        if sy < ry0 or sy > ry1:
            return 0.0
        sx0, sx1 = min(x1, x2), max(x1, x2)
        overlap = max(0.0, min(sx1, rx1) - max(sx0, rx0))
        return overlap / seg_len
    sx = (x1 + x2) / 2
    if sx < rx0 or sx > rx1:
        return 0.0
    sy0, sy1 = min(y1, y2), max(y1, y2)
    overlap = max(0.0, min(sy1, ry1) - max(sy0, ry0))
    return overlap / seg_len


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


def _ink_bbox_area(mask: Optional[np.ndarray]) -> int:
    if mask is None or cv2.countNonZero(mask) == 0:
        return 0
    _, _, bw, bh = cv2.boundingRect(mask)
    return bw * bh


def _retry_footprint_if_sparse(
    component_mask: Optional[np.ndarray],
    binary: np.ndarray,
    wall_pair_mask: np.ndarray,
    px_per_unit: float,
    user_roi: bool,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Retry footprint detection with the any-ink binary when the strict one
    misses most of the wall ink.

    At low working resolutions (large sheets downscaled by MAX_ANALYSIS_PX)
    the pair-mask strokes are 1-2 px wide and the strict 127 threshold in
    footprint_binary erases them, so the footprint collapses to the thickest
    wall (e.g. a fire-wall poche) and every wall outside it — typically the
    whole warehouse/exterior perimeter — is discarded downstream. Detect that
    by comparing the footprint's bounding box against the wall ink's bounding
    box and rebuild the binary keeping any inked cell.

    Returns (component_mask, binary), updated only when the retry found a
    larger footprint.
    """
    ink_mask = wall_pair_mask
    if not user_roi:
        # Full-sheet mode: ignore title-block/margin ink when judging coverage
        # and rebuilding, mirroring the margin-blanked mask preprocess used.
        h, w = wall_pair_mask.shape
        excl = _build_exclusion_mask(h, w)
        ink_mask = cv2.bitwise_and(wall_pair_mask, cv2.bitwise_not(excl))

    ink_area = _ink_bbox_area(ink_mask)
    fp_area = _ink_bbox_area(component_mask)
    if not ink_area or fp_area >= 0.5 * ink_area:
        return component_mask, binary

    retry_binary = footprint_binary(ink_mask, px_per_unit, any_ink=True)
    retry_mask = find_footprint(retry_binary, use_exclusion=not user_roi)
    retry_area = _ink_bbox_area(retry_mask)
    if retry_area > fp_area:
        print(
            f"  [pipeline] footprint retry (thin strokes): bbox coverage "
            f"{fp_area / ink_area:.0%} → {retry_area / ink_area:.0%} of wall ink",
            file=sys.stderr,
        )
        return retry_mask, retry_binary
    return component_mask, binary


def analyze_page(
    image: np.ndarray,
    scale_str: str,
    dpi: int,
    roi: Optional[dict] = None,
    doorway_close_ft: float = 2.5,
    room_debug_dir: Optional[str] = None,
    crop_mode: bool = False,
    px_per_unit_override: Optional[float] = None,
) -> dict:
    sheet_w = image.shape[1]
    sheet_h = image.shape[0]
    roi_offset = (0, 0)
    roi_hint_pct = None
    analysis_roi_pct = None
    effective_roi_crop = None

    # Parse scale on the full sheet so px_per_unit is available for ROI expansion.
    cal = parse_scale(scale_str, dpi, output_unit="ft")
    px_per_unit = float(px_per_unit_override) if px_per_unit_override else cal["px_per_unit"]
    unit_label = cal["unit_label"]
    is_metric = unit_label == "m"
    area_unit = f"{unit_label}²"

    if roi:
        roi_hint_pct = {
            "x0_pct": roi["x0_pct"],
            "y0_pct": roi["y0_pct"],
            "x1_pct": roi["x1_pct"],
            "y1_pct": roi["y1_pct"],
            "method": roi.get("method", "user-roi"),
        }
        roi = _expand_roi_from_hint(image, roi_hint_pct, px_per_unit)
        analysis_roi_pct = dict(roi)
        image, roi_offset, full_w, full_h = _crop_to_roi(image, roi)
        crop_h, crop_w = image.shape[:2]
        effective_roi_crop = _roi_full_to_crop(
            analysis_roi_pct, roi_offset, full_w, full_h, crop_w, crop_h,
        )
    else:
        full_w, full_h = sheet_w, sheet_h

    calibration_issues = []
    if not crop_mode:
        calibration_issues.extend(validate_dpi(dpi))
    calibration_issues.extend(validate_px_per_unit(px_per_unit, unit_label))
    for issue in calibration_issues:
        print(f"  {issue_to_log_line(issue)}", file=sys.stderr)

    # With a user-drawn ROI the title block/margins are already cropped away;
    # the sheet-fraction exclusion zones would erase real walls (bottom 18 %,
    # top 12 %, bottom-right quadrant of the crop), clip the footprint, and
    # cause every wall segment in those areas to be dropped downstream.
    user_roi = roi is not None
    t0 = time.time()
    binary, wall_pair_mask = preprocess(
        image, px_per_unit=px_per_unit, apply_margins=not user_roi
    )
    print(f"  [pipeline] preprocess total: {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    component_mask = find_footprint(binary, use_exclusion=not user_roi)
    print(f"  [pipeline] find_footprint: {time.time()-t0:.1f}s", file=sys.stderr)

    component_mask, binary = _retry_footprint_if_sparse(
        component_mask, binary, wall_pair_mask, px_per_unit, user_roi,
    )

    if crop_mode and component_mask is not None:
        img_area = image.shape[0] * image.shape[1]
        fp_frac = cv2.countNonZero(component_mask) / max(img_area, 1)
        if fp_frac < 0.25:
            # Tight LabelMe crops often touch all edges; morphology can pick a
            # small interior island. Fall back to the wall-ink envelope.
            k = max(15, int(2.5 * px_per_unit))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            dil = cv2.dilate(wall_pair_mask, kernel, iterations=2)
            alt = flood_fill_interior(dil)
            alt_frac = cv2.countNonZero(alt) / max(img_area, 1)
            if 0.25 <= alt_frac <= 0.98:
                print(
                    f"  [pipeline] crop footprint fallback: {fp_frac:.2f} → {alt_frac:.2f} "
                    f"of image (wall-ink envelope)",
                    file=sys.stderr,
                )
                component_mask = alt

    if component_mask is None:
        return {"error": "No building footprint found"}

    contour = find_footprint_contour(component_mask)
    if contour is None:
        return {"error": "No building footprint found"}

    eps = 0.002 if roi else 0.001
    polygon = simplify_polygon(contour, epsilon_factor=eps)
    analyze_page._last_polygon = polygon
    img_h, img_w = image.shape[:2]
    # Scale-adaptive minimum: walls shorter than 12 real-world units are polygon
    # artifacts on full sheets. LabelMe crops are already tight — use 6 ft so
    # partial exterior edges on large dorm crops are not discarded.
    if crop_mode:
        min_seg_px = max(36, int(6 * px_per_unit))
    else:
        min_seg_px = max(60, int(12 * px_per_unit))
    print(
        f"  [pipeline] min_seg_px={min_seg_px} (px_per_unit={px_per_unit:.1f}, "
        f"crop_mode={crop_mode})",
        file=sys.stderr,
    )
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
    filter_roi = _expand_poly_roi(poly_roi, wall_pair_mask, img_w, img_h) if user_roi else poly_roi
    if user_roi and effective_roi_crop is not None:
        filter_roi = _union_roi(filter_roi, effective_roi_crop)
    # ROI crops often fill the box: north exterior walls legitimately span ~100 %
    # of crop width and were dropped by the old 0.95 × min(w,h) cap.
    max_span_frac = 1.0 if user_roi else 0.95
    segments = _filter_wall_segments(segments, img_w, img_h, roi=filter_roi, max_span_frac=max_span_frac)
    print(f"  [pipeline] segments: raw={raw_count} after_filter={len(segments)}", file=sys.stderr)
    segments = snap_segments_to_walls(
        segments, wall_pair_mask, px_per_unit=px_per_unit,
        validate_pairs=True, allow_hop=True,
    )
    print(f"  [pipeline] segments after snap={len(segments)}", file=sys.stderr)
    # Cut polygon-edge tails that ride an annotation strip past the building.
    segments = clamp_segments_to_envelope(
        segments, pad_px=max(2, wall_pair_gap_range(px_per_unit)[1] // 2),
    )

    exterior_segs = list(segments)

    xs_poly = polygon[:, 0]
    ys_poly = polygon[:, 1]
    fp_bbox_for_facing = [
        int(xs_poly.min()), int(ys_poly.min()),
        int(xs_poly.max()), int(ys_poly.max()),
    ]

    # Split exterior walls into per-room sub-segments using a geometric room map.
    t0 = time.time()
    rooms, exterior_walls = split_exterior_walls_by_room(
        exterior_segs,
        wall_pair_mask=wall_pair_mask,
        contour=contour,
        footprint_bbox=fp_bbox_for_facing,
        image_shape=image.shape,
        px_per_unit=px_per_unit,
        unit_label=unit_label,
        doorway_close_ft=doorway_close_ft,
        debug_dir=room_debug_dir,
        crop_mode=crop_mode,
    )
    print(
        f"  [pipeline] room split: {time.time()-t0:.1f}s "
        f"({len(rooms)} rooms, {len(exterior_walls)} exterior sub-segments)",
        file=sys.stderr,
    )

    # Supplement with Hough lines for interior walls missed by the polygon.
    ext_sub_segs = [tuple(w["px_coords"]) for w in exterior_walls]
    dedup_ref = ext_sub_segs + exterior_segs
    near_tol = max(15, int(0.5 * px_per_unit))
    pair_gap_range = wall_pair_gap_range(px_per_unit)
    if crop_mode:
        hough_min_px = max(36, int(2.5 * px_per_unit))
    else:
        hough_min_px = max(60, int(6 * px_per_unit))
    hough_dedup_tol = dedup_axis_tol_px(px_per_unit)
    hough_segs = _hough_supplement(
        wall_pair_mask, dedup_ref,
        min_length_px=hough_min_px,
        dedup_tol_px=hough_dedup_tol,
        pair_gap_range=pair_gap_range,
    )
    if hough_segs:
        hough_segs = _filter_wall_segments(
            hough_segs, img_w, img_h, roi=filter_roi, max_span_frac=max_span_frac
        )
        hough_segs = snap_segments_to_walls(
            hough_segs, wall_pair_mask, px_per_unit=px_per_unit, validate_pairs=True,
        )
        before_trace = len(hough_segs)
        hough_segs = [
            s for s in hough_segs
            if not segment_traces_exterior(s, dedup_ref, near_tol)
        ]
        if before_trace > len(hough_segs):
            print(
                f"  [pipeline] rejected {before_trace - len(hough_segs)} "
                f"exterior-tracing Hough segments",
                file=sys.stderr,
            )
        # Annotation ink (dimension strings) sits outside the building; with
        # the exteriors pair-validated onto true wall ink, anything beyond
        # their axis envelope cannot be an interior wall.
        before_env = len(hough_segs)
        hough_segs = drop_segments_outside_exterior(hough_segs, exterior_segs)
        if before_env > len(hough_segs):
            print(
                f"  [pipeline] rejected {before_env - len(hough_segs)} "
                f"Hough segments outside the exterior envelope",
                file=sys.stderr,
            )
        print(f"  [pipeline] Hough supplement: +{len(hough_segs)} interior segments",
              file=sys.stderr)

    axis_tol_px = dedup_axis_tol_px(px_per_unit)
    gap_tol_px = max(5, int(0.3 * px_per_unit))
    if hough_segs:
        before_dedup = len(hough_segs)
        hough_segs = merge_and_deduplicate_segments(
            hough_segs, axis_tol_px=axis_tol_px, gap_tol_px=gap_tol_px
        )
        print(f"  [pipeline] interior dedup: {before_dedup} → {len(hough_segs)} segments",
              file=sys.stderr)

    # Short-partition recovery: closet/alcove walls are real wall pairs but
    # shorter than the 6-unit Hough minimum (a door gap splits them further).
    # A second pass at ~2.5 units keeps only candidates that T-junction into
    # an already-accepted wall (iterated, so a stub chaining off a recovered
    # partition is also accepted) — free-floating short runs stay rejected.
    short_min_px = max(36, int(2.5 * px_per_unit))
    if short_min_px < hough_min_px:
        accepted_ref = dedup_ref + hough_segs
        short_segs = _short_run_supplement(
            wall_pair_mask, accepted_ref,
            min_length_px=short_min_px,
            max_length_px=2 * hough_min_px,
            dedup_tol_px=hough_dedup_tol,
            pair_gap_range=pair_gap_range,
        )
        if short_segs:
            short_segs = _filter_wall_segments(
                short_segs, img_w, img_h, roi=filter_roi, max_span_frac=max_span_frac
            )
            # No snap_segments_to_walls here: a short pair next to a long wall
            # gets dragged onto the stronger centerline (e.g. closet partition
            # onto the exterior, 60+ px away) and then rejected as tracing it.
            # The candidates are already pair-validated by _hough_supplement.
            short_segs = [
                s for s in short_segs
                if not segment_traces_exterior(s, dedup_ref, near_tol)
            ]
            short_segs = drop_segments_outside_exterior(short_segs, exterior_segs)
            # Reject fixture/dimension scraps riding 1–2.5 units off a wall
            # (or off each other) — keeping them would also poison cleanup's
            # dimension-like classification of the real walls beside them.
            offset_max_px = max(int(2.5 * px_per_unit), axis_tol_px * 2)
            short_segs = [
                s for s in short_segs
                if not _short_run_annotation_like(
                    s, accepted_ref + short_segs, axis_tol_px, offset_max_px
                )
            ]
        if short_segs:
            accepted = list(accepted_ref)
            kept_short: list[tuple] = []
            changed = True
            while changed and short_segs:
                changed = False
                remaining = []
                for s in short_segs:
                    if _t_junctions_into(s, accepted, near_tol):
                        kept_short.append(s)
                        accepted.append(s)
                        changed = True
                    else:
                        remaining.append(s)
                short_segs = remaining
            if kept_short:
                # Dedup the shorts among themselves only: re-merging the main
                # interior set would extend/absorb already-stable walls and
                # destabilize real plans. This pass is purely additive.
                kept_short = merge_and_deduplicate_segments(
                    kept_short, axis_tol_px=axis_tol_px, gap_tol_px=gap_tol_px,
                )
                hough_segs = hough_segs + kept_short
                print(
                    f"  [pipeline] short-partition recovery: +{len(kept_short)} "
                    f"T-junctioned segments ({len(hough_segs)} interior total)",
                    file=sys.stderr,
                )

    interior_walls = measure_walls(
        hough_segs, px_per_unit, unit_label,
        contour=contour, footprint_bbox=fp_bbox_for_facing,
    )
    interior_id_base = len(exterior_segs)
    for i, w in enumerate(interior_walls):
        w["id"] = f"w{interior_id_base + i + 1}"
        w["name"] = f"{w['facing']} Wall {interior_id_base + i + 1}"
        w["is_exterior"] = False

    walls = exterior_walls + interior_walls
    before_cleanup = len(walls)
    walls, cleanup_stats = cleanup_wall_list(
        walls, axis_tol_px, px_per_unit, unit_label,
    )
    dropped = before_cleanup - len(walls)
    if dropped:
        parts = ", ".join(
            f"{k}={v}" for k, v in cleanup_stats.items() if v
        )
        print(
            f"  [pipeline] wall cleanup: {before_cleanup} → {len(walls)} "
            f"({parts or 'no change'})",
            file=sys.stderr,
        )
    min_real = 8.0 if unit_label == "ft" else 2.5
    walls_before_len = list(walls)
    if roi:
        # Keep all exterior sub-segments — room split already merges slivers.
        exterior_kept = [w for w in walls if w.get("is_exterior")]
        interior_kept = [w for w in walls if not w.get("is_exterior")
                         and w["length_raw"] >= min_real]
        walls = exterior_kept + interior_kept
        if len(walls) < 8:
            walls = sorted(walls_before_len, key=lambda w: -w["length_raw"])[:20]

    walls, snapped_ends = snap_wall_endpoints(walls, px_per_unit, unit_label)
    if snapped_ends:
        print(f"  [pipeline] endpoint snap: closed {snapped_ends} endpoint(s)",
              file=sys.stderr)

    # Doors: collinear wall gaps in the same crop frame as walls/mask.
    t0 = time.time()
    ink_mask = ink_mask_from_image(image)
    doors = detect_doors(
        walls, wall_pair_mask, ink_mask,
        px_per_unit, unit_label,
        axis_tol_px=dedup_axis_tol_px(px_per_unit),
        crop_mode=crop_mode,
    )
    print(f"  [pipeline] door detect: {time.time()-t0:.1f}s ({len(doors)} doors)",
          file=sys.stderr)

    t0 = time.time()
    windows = detect_windows(
        walls, wall_pair_mask, ink_mask,
        px_per_unit, unit_label,
        axis_tol_px=dedup_axis_tol_px(px_per_unit),
        crop_mode=crop_mode,
        doors=doors,
    )
    print(f"  [pipeline] window detect: {time.time()-t0:.1f}s ({len(windows)} windows)",
          file=sys.stderr)

    if roi_offset != (0, 0):
        ox, oy = roi_offset
        for w in walls:
            w["px_coords"] = _shift_px_coords(w["px_coords"], roi_offset)
        for d in doors:
            x0, y0, x1, y1 = d["bbox_px"]
            d["bbox_px"] = [x0 + ox, y0 + oy, x1 + ox, y1 + oy]
            cx, cy = d["center_px"]
            d["center_px"] = [cx + ox, cy + oy]
        for win in windows:
            x0, y0, x1, y1 = win["bbox_px"]
            win["bbox_px"] = [x0 + ox, y0 + oy, x1 + ox, y1 + oy]
            cx, cy = win["center_px"]
            win["center_px"] = [cx + ox, cy + oy]
        for r in rooms:
            cx, cy = r["centroid_px"]
            r["centroid_px"] = [cx + ox, cy + oy]
            x0, y0, x1, y1 = r["bbox_px"]
            r["bbox_px"] = [x0 + ox, y0 + oy, x1 + ox, y1 + oy]
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

    width_ft = abs(fp_bbox[2] - fp_bbox[0]) / px_per_unit
    height_ft = abs(fp_bbox[3] - fp_bbox[1]) / px_per_unit
    footprint_span_ft = (width_ft, height_ft)

    late_issues = []
    late_issues.extend(validate_footprint_span(fp_bbox, px_per_unit, unit_label))
    late_issues.extend(validate_total_area(total_area_real, unit_label))
    late_issues.extend(check_dpi_alternatives(scale_str, dpi, footprint_span_ft))
    calibration_issues.extend(late_issues)
    for issue in late_issues:
        print(f"  {issue_to_log_line(issue)}", file=sys.stderr)

    calibration = summarize_calibration(
        calibration_issues,
        dpi=dpi,
        px_per_unit=px_per_unit,
        footprint_span_ft=footprint_span_ft,
        total_area_raw=total_area_real,
        unit_label=unit_label,
    )
    if crop_mode:
        calibration["crop_mode"] = True

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
        "rooms": rooms,
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "mask_cache_path": mask_cache_path,
        # roi_offset is the pixel offset of the cropped image within the full image.
        # detect_wall_at_point needs this to translate full-image click coords into
        # crop-image coords before querying the mask.
        "mask_roi_offset": list(roi_offset),
        "calibration": calibration,
        "roi_hint_pct": roi_hint_pct,
        "analysis_roi_pct": analysis_roi_pct,
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