"""
preprocess.py — Exterior wall measurement & facing from architectural plan PDFs.

Pipeline:
  1. Rasterize PDF pages at 300 DPI
  2. Preprocess: grayscale → blur → binary threshold → morphological cleanup
  3. Find the largest external contour (building footprint)
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
import cv2
import fitz  # PyMuPDF — no external Poppler binaries needed
import numpy as np


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
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close to bridge small gaps in wall lines
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


# ─── Step 3: Find the building footprint (largest external contour) ─────────

def find_footprint(binary: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found", file=sys.stderr)
        return None
    largest = max(contours, key=cv2.contourArea)

    image_area = binary.shape[0] * binary.shape[1]
    if cv2.contourArea(largest) < image_area * 0.01:
        return None

    return largest


# ─── Step 4: Simplify contour → clean polygon ──────────────────────────────

def simplify_polygon(contour: np.ndarray, epsilon_factor: float = 0.02) -> np.ndarray:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    return approx.reshape(-1, 2)


# ─── Step 5: Segment polygon into wall segments ────────────────────────────

def extract_wall_segments(polygon: np.ndarray) -> list[tuple]:
    """Returns list of (x1, y1, x2, y2) for each edge of the polygon."""
    segments = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        segments.append((int(x1), int(y1), int(x2), int(y2)))
    return segments


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


# ─── Step 7: Scale calibration & real-world measurement ────────────────────

def parse_scale(scale_str: str, dpi: int) -> float:
    """
    Parse a scale string and return pixels-per-foot.

    Supported formats:
      "1/4in=1ft"   → 0.25 inches on paper = 1 foot real
      "1:100"       → 1 unit on paper = 100 units real  (metric, returns px/m)
      "1/8in=1ft"
    """
    scale_str = scale_str.strip().lower().replace(" ", "").replace("\"", "in").replace("'", "ft")

    if "=" in scale_str:
        left, right = scale_str.split("=")
        paper_inches = _parse_length_inches(left)
        real_feet = _parse_length_feet(right)
        pixels_per_paper_inch = dpi
        pixels_per_foot = (paper_inches * pixels_per_paper_inch) / real_feet
        return pixels_per_foot

    if ":" in scale_str:
        parts = scale_str.split(":")
        ratio = float(parts[1]) / float(parts[0])
        pixels_per_unit = dpi / 25.4  # px per mm at this DPI
        pixels_per_meter = pixels_per_unit * 1000 / ratio
        return pixels_per_meter

    raise ValueError(f"Cannot parse scale: {scale_str}")


def _parse_length_inches(s: str) -> float:
    s = s.replace("in", "").replace("inch", "")
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)
    return float(s)


def _parse_length_feet(s: str) -> float:
    s = s.replace("ft", "").replace("foot", "").replace("feet", "")
    return float(s)


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
    segments = extract_wall_segments(polygon)

    is_metric = ":" in scale_str
    unit_label = "m" if is_metric else "ft"
    px_per_unit = parse_scale(scale_str, dpi)

    walls = measure_walls(segments, px_per_unit, unit_label)

    total_area_px = cv2.contourArea(contour)
    total_area_real = total_area_px / (px_per_unit ** 2)
    area_unit = "m²" if is_metric else "ft²"

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
        contour = find_footprint(binary)
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