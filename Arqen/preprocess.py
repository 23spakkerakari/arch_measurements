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


def preprocess(image: np.ndarray) -> np.ndarray:
    '''
    We sharpen the edges and lines on our images
    using threshold (line below)
    this helps cv2 better understand and pick out the walls 
    '''
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    min_area = 40 #to be tuned

    for label in range(1, num_labels):
        area = stats[labels, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    
    # Here we're preserving long horizontal and vertical structure
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    horiz = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vert_kernel)

    structure = cv2.bitwise_or(horiz, vert)

    # Bridge small gaps in wall lines
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(structure, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return closed

def find_footprint(binary: np.ndarry):
    '''
    New Plan:
    1. Finding connected components
    2. Filtering out junk
    3. Finding best contour from connected components
    4. Overall finding the plan component

    '''
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary) #

    h, w = binary.shape
    image_area = h*w

    best_label = None
    best_score = -1

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT] #leftmost x coordinate
        y = stats[label, cv2.CC_STAT_TOP] #topmost y coordinate
        width = stats[label, cv2.CC_STAT_WIDTH] #width of the component
        height = stats[label, cv2.CC_STAT_HEIGHT] #height of the component
        area = stats[label, cv2.CC_STAT_AREA] #area of the component

        if area < image_area * 0.002: # we use this to reject tiny, random squares and stuff
            continue
        
        bbox_area = width * height
        fill_ratio = area / max(bbox_area, 1)

        # Reject extremely thin note-like / line-like regions
        aspect = width / max(height, 1)
        if aspect > 12 or aspect < 1 / 12:
            continue
        
        score = area * (0.5 + min(fill_ratio, 1))

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None
    
    mask = np.zeros_like(binary) #empty mask
    mask[labels == best_label] = 255 #set the mask to the best label
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
    binary = preprocess(image)

    contour = find_footprint(binary)
    if contour is None:
        return {"error": "No building footprint found"}

    polygon = simplify_polygon(contour)
    segments = extract_wall_segments(polygon) #see extract_wall_segments_class.py
        
    cal = parse_scale(scale_str, dpi, output_unit="ft") #see scale_parse.py
    px_per_unit = cal["px_per_unit"]
    unit_label = cal["unit_label"]
    is_metric = unit_label == "m"
    area_unit = f"{unit_label}²"

    walls = measure_walls(segments, px_per_unit, unit_label)

    total_area_px = cv2.contourArea(contour)
    total_area_real = total_area_px / (px_per_unit ** 2)

    return {
        "detected_scale": scale_str,
        "total_area": f"{total_area_real:.1f} {area_unit}",
        "units": "metric" if is_metric else "imperial",
        "polygon_vertices": len(polygon),
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
        binary = preprocess(image)
        component_mask = find_footprint(binary)
        contour = find_footprint_contour(component_mask)
        polygon = simplify_polygon(contour)
        vis_path = str(pdf_path.with_suffix(".annotated.png"))
        visualize(image, polygon, vis_path)
        result["visualization"] = vis_path
        print(f"Saved annotated image to {vis_path}", file=sys.stderr)

    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()