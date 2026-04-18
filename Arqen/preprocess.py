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


def _extract_wall_lines(image: np.ndarray) -> np.ndarray:
    """Threshold → clean noise → extract horizontal/vertical structure."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = image.shape
    image = image.copy()
    image[:, int(w * 0.78):] = 255  # 255 = white (background) in grayscale

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = 40
    keep = np.zeros(num_labels, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[label] = 255
    cleaned = keep[labels]

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horiz = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vert_kernel)
    return cv2.bitwise_or(horiz, vert)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Two-pass preprocessing:
      1. Extract H/V wall lines at full resolution.
      2. Downscale → heavy dilation+closing to bridge doorways →
         upscale the solid footprint mask back to full resolution.
    """
    t0 = time.time()
    walls = _extract_wall_lines(image)
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
    filled = flood_fill_interior(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled)

    h, w = filled.shape
    image_area = h * w

    edge_margin_x = w * 0.03
    edge_margin_y = h * 0.03

    best_label = None
    best_score = -1

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < image_area * 0.001:
            continue

        aspect = width / max(height, 1)
        if aspect > 12 or aspect < 1/12:
            continue

        touches_all_edges = (
            x < edge_margin_x and
            y < edge_margin_y and
            (x + width) > (w - edge_margin_x) and
            (y + height) > (h - edge_margin_y) 
        )

        if not touches_all_edges:
            continue
        
        #or if this is the largest box, and it takes up a lot of the image
        if width > w * 0.8 and height > h * 0.8:
            continue

        score = height * width 
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None
    
    mask = np.zeros_like(filled)
    mask[labels == best_label] = 255
    return mask

def find_footprint_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    return largest


# ─── Step 4: Simplify contour → clean polygon ──────────────────────────────

#finding some parameteres to help us work with the shape of our building 
def simplify_polygon(contour: np.ndarray, epsilon_factor: float = 0.005) -> np.ndarray:
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

def analyze_page(image: np.ndarray, scale_str: str, dpi: int) -> dict:
    t0 = time.time()
    binary = preprocess(image)
    print(f"  [pipeline] preprocess total: {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    component_mask = find_footprint(binary)
    print(f"  [pipeline] find_footprint: {time.time()-t0:.1f}s", file=sys.stderr)

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

    walls = measure_walls(segments, px_per_unit, unit_label)

    total_area_px = cv2.contourArea(contour)
    total_area_real = total_area_px / (px_per_unit ** 2)

    img_h, img_w = image.shape[:2]
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    fp_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

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