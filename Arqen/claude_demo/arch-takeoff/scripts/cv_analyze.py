#!/usr/bin/env python3
"""Run Arqen preprocess.py CV pipeline on a raster image; output JSON to stdout."""

import argparse
import json
import sys
from pathlib import Path

import cv2

ARQEN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ARQEN_ROOT))

from preprocess import analyze_page  # noqa: E402


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to plan PNG/JPG")
    parser.add_argument("--scale", required=True, help='e.g. "1:50" or "1/4in=1ft"')
    parser.add_argument("--dpi", type=int, default=150, help="DPI hint for scale calibration")
    parser.add_argument(
        "--roi",
        default=None,
        help="Building region as fractions: x0,y0,x1,y1 (0-1)",
    )
    args = parser.parse_args()

    roi = None
    if args.roi:
        parts = [float(x) for x in args.roi.split(",")]
        if len(parts) == 4:
            roi = {
                "x0_pct": parts[0],
                "y0_pct": parts[1],
                "x1_pct": parts[2],
                "y1_pct": parts[3],
            }

    image = load_image(args.image)
    result = analyze_page(image, args.scale, args.dpi, roi=roi)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
