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


def _extract_wall_lines(
    image: np.ndarray,
    blank_right_frac: float = 0.78,
    blank_bottom_frac: float = 0.91,
    blank_left_px: int = 60,
) -> np.ndarray:
    """Threshold → clean noise → extract H/V wall lines via double-line pairing."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = image.shape
    image = image.copy()
    image[:int(h * 0.12), :] = 255                # blank top notes/header zone
    image[int(h * blank_bottom_frac):, :] = 255   # blank bottom title-block / notes strip
    image[:, :blank_left_px] = 255                 # blank left border/grid line
    image[:, int(w * blank_right_frac):] = 255     # blank right-side title block strip

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
    horiz_walls = _find_wall_pairs(horiz, scan_rows=True,  min_gap_px=1, max_gap_px=60)
    vert_walls  = _find_wall_pairs(vert,  scan_rows=False, min_gap_px=1, max_gap_px=60)
    return cv2.bitwise_or(horiz_walls, vert_walls)


def preprocess(
    image: np.ndarray,
    blank_right_frac: float = 0.78,
    blank_bottom_frac: float = 0.91,
) -> np.ndarray:
    """
    Two-pass preprocessing:
      1. Extract H/V wall lines at full resolution.
      2. Downscale → heavy dilation+closing to bridge doorways →
         upscale the solid footprint mask back to full resolution.
    """
    t0 = time.time()
    walls = _extract_wall_lines(
        image, blank_right_frac=blank_right_frac, blank_bottom_frac=blank_bottom_frac
    )
    print(f"  [preprocess] wall-line extraction: {time.time()-t0:.1f}s", file=sys.stderr)

    h, w = walls.shape
    small_h, small_w = h // DOWNSCALE, w // DOWNSCALE

    t0 = time.time()
    # Max-pool: a block is "on" if ANY pixel in it is a wall pixel.
    # INTER_AREA averaging fails on sparse wall masks (value stays < 127),
    # so we reshape-and-max instead.
    small = (
        walls[:small_h * DOWNSCALE, :small_w * DOWNSCALE]
        .reshape(small_h, DOWNSCALE, small_w, DOWNSCALE)
        .max(axis=(1, 3))
        .astype(np.uint8)
    )
    _, small = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY)

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

    best_label = None
    best_area = -1

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

        if area > best_area:
            best_area = area
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


def _point_on_segment(px: float, py: float, seg: tuple, tol: float) -> bool:
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
    x1, y1, x2, y2 = seg
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x_min, y_min, x_max, y_max = footprint_bbox
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    is_horiz = _segment_is_horizontal(x1, y1, x2, y2)
    neighbor_facings = [facings[j] for j in adjacency[seg_idx] if facings[j] is not None]

    if is_horiz:
        for f in neighbor_facings:
            if f in _HORIZ_FACINGS:
                return f
        for j in adjacency[seg_idx]:
            if facings[j] in _VERT_FACINGS:
                return "North" if my < cy else "South"
        return "North" if my < cy else "South"

    for f in neighbor_facings:
        if f in _VERT_FACINGS:
            return f
    for j in adjacency[seg_idx]:
        if facings[j] in _HORIZ_FACINGS:
            return "East" if mx > cx else "West"
    return "West" if mx < cx else "East"


def assign_segment_facings(
    segments: list[tuple],
    contour: np.ndarray,
    footprint_bbox: list[int],
    px_per_unit: float,
) -> list[str]:
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


# ─── Main pipeline ──────────────────────────────────────────────────────────

def analyze_page(image: np.ndarray, scale_str: str, dpi: int) -> dict:
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

    polygon = simplify_polygon(contour)
    # Stash intermediates so --visualize doesn't re-run the pipeline
    analyze_page._last_polygon = polygon
    segments = extract_wall_segments(polygon) #see extract_wall_segments_class.py
        
    cal = parse_scale(scale_str, dpi, output_unit="ft") #see scale_parse.py
    px_per_unit = cal["px_per_unit"]
    unit_label = cal["unit_label"]
    is_metric = unit_label == "m"
    area_unit = f"{unit_label}²"

    xs = polygon[:, 0]
    ys = polygon[:, 1]
    fp_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    walls = measure_walls(
        segments, px_per_unit, unit_label,
        contour=contour, footprint_bbox=fp_bbox,
    )

    total_area_px = cv2.contourArea(contour)
    total_area_real = total_area_px / (px_per_unit ** 2)

    img_h, img_w = image.shape[:2]

    return {
        "detected_scale": scale_str,
        "total_area": f"{total_area_real:.1f} {area_unit}",
        "units": "metric" if is_metric else "imperial",
        "polygon_vertices": len(polygon),
        "image_size_px": [img_w, img_h],
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