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
import time
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


def _strip_spanning_grid_lines(mask: np.ndarray) -> np.ndarray:
    """
    Remove long H/V runs that span most of the sheet (structural grid, borders).
    Building wall strokes are shorter than ~40% of the sheet span.
    """
    h, w = mask.shape
    out = mask.copy()
    kw = max(40, int(w * 0.42))
    kh = max(40, int(h * 0.42))
    long_h = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    )
    long_v = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    )
    out = cv2.bitwise_and(out, cv2.bitwise_not(long_h))
    out = cv2.bitwise_and(out, cv2.bitwise_not(long_v))
    return out


def _extract_wall_lines(image: np.ndarray, blank_right_frac: Optional[float] = None) -> np.ndarray:
    """Threshold → clean noise → extract H/V wall lines via double-line pairing."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = image.shape
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
    # Wall thickness gap (wider than typical grid-line spacing).
    horiz_walls = _find_wall_pairs(horiz, scan_rows=True, min_gap_px=6, max_gap_px=55)
    vert_walls = _find_wall_pairs(vert, scan_rows=False, min_gap_px=6, max_gap_px=55)
    combined = cv2.bitwise_or(horiz_walls, vert_walls)
    return _strip_spanning_grid_lines(combined)


def preprocess(image: np.ndarray, blank_right_frac: Optional[float] = None) -> np.ndarray:
    """
    Two-pass preprocessing:
      1. Extract H/V wall lines at full resolution.
      2. Downscale → heavy dilation+closing to bridge doorways →
         upscale the solid footprint mask back to full resolution.
    """
    t0 = time.time()
    walls = _extract_wall_lines(image, blank_right_frac=blank_right_frac)
    print(f"  [preprocess] wall-line extraction: {time.time()-t0:.1f}s", file=sys.stderr)

    h, w = walls.shape
    small_h, small_w = h // DOWNSCALE, w // DOWNSCALE

    t0 = time.time()
    small = cv2.resize(walls, (small_w, small_h), interpolation=cv2.INTER_AREA)
    _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)

    small = cv2.dilate(
        small,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2,
    )

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, close_k, iterations=1)

    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    print(f"  [preprocess] downscale morphology: {time.time()-t0:.1f}s", file=sys.stderr)

    return big

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
      - Reject components that touch the image edge (sheet border artifacts).
      - Reject components that fill most of the image (the border itself).
      - Reject components with extreme aspect ratios (title blocks, scale bars).
      - Among survivors, pick the largest by area.
    """
    filled = flood_fill_interior(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled)

    h, w = filled.shape
    image_area = h * w

    edge_margin_x = w * 0.01
    edge_margin_y = h * 0.01

    excl = _build_exclusion_mask(h, w)
    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < image_area * 0.005:
            continue

        if area > image_area * 0.85:
            continue

        aspect = width / max(height, 1)
        if aspect > 12 or aspect < 1 / 12:
            continue

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
    t0 = time.time()
    binary = preprocess(image)
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
    min_seg_px = max(70.0, min(img_w, img_h) * 0.03) if roi else max(60.0, min(img_w, img_h) * 0.025)
    segments = extract_wall_segments(polygon, min_length_px=min_seg_px)
    raw_count = len(segments)
    segments = _filter_wall_segments(segments, img_w, img_h, roi=None)
        
    cal = parse_scale(scale_str, dpi, output_unit="ft") #see scale_parse.py
    px_per_unit = cal["px_per_unit"]
    unit_label = cal["unit_label"]
    is_metric = unit_label == "m"
    area_unit = f"{unit_label}²"

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