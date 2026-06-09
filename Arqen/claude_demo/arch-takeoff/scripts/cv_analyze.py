#!/usr/bin/env python3
"""Run Arqen preprocess.py CV pipeline on a raster image; output JSON to stdout."""

import argparse
import json
import sys
from pathlib import Path

import cv2

ARQEN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ARQEN_ROOT))

from preprocess import analyze_page, detect_wall_at_point, preprocess  # noqa: E402
from scale_parse import parse_scale  # noqa: E402


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
    parser.add_argument(
        "--mode",
        default="analyze",
        choices=["analyze", "detect-wall"],
        help="analyze: full pipeline; detect-wall: find wall near a click point",
    )
    parser.add_argument("--x_pct", type=float, default=None, help="Click X as image fraction (0-1)")
    parser.add_argument("--y_pct", type=float, default=None, help="Click Y as image fraction (0-1)")
    parser.add_argument("--mask_cache_path", default=None, help="Path to cached wall_pair_mask PNG (skips preprocessing)")
    parser.add_argument("--mask_roi_offset", default="0,0", help="Pixel offset of the mask's crop origin within the full image: ox,oy")
    args = parser.parse_args()

    image = load_image(args.image)

    if args.mode == "detect-wall":
        if args.x_pct is None or args.y_pct is None:
            print(json.dumps({"error": "--x_pct and --y_pct required for detect-wall mode"}))
            sys.exit(1)
        h, w = image.shape[:2]
        x_px = int(args.x_pct * w)
        y_px = int(args.y_pct * h)
        cal = parse_scale(args.scale, args.dpi, output_unit="ft")
        px_per_unit = cal["px_per_unit"]

        # Use the cached mask from the initial analysis to avoid re-running
        # the expensive _extract_wall_lines pass on every click.
        if args.mask_cache_path:
            wall_mask = cv2.imread(args.mask_cache_path, cv2.IMREAD_GRAYSCALE)
            if wall_mask is None:
                print(json.dumps({"error": f"Could not load mask cache: {args.mask_cache_path}"}))
                sys.exit(1)
            mh, mw = wall_mask.shape[:2]
            # #region agent log
            import os as _os
            _log_path = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..', 'debug-7104c9.log'))
            with open(_log_path, 'a') as _lf:
                import json as _json
                _lf.write(_json.dumps({'sessionId':'7104c9','location':'cv_analyze.py:mask-loaded','message':'mask loaded','data':{'mask_wh':[mw,mh],'image_wh':[w,h],'mask_nonzero':int(cv2.countNonZero(wall_mask)),'x_px':x_px,'y_px':y_px},'timestamp':int(__import__('time').time()*1000),'hypothesisId':'WALL-MASK'}) + '\n')
            # #endregion
            # Translate full-image click coords into mask/crop-image coords.
            # The mask was saved from a ROI-cropped image; using full-image coords
            # directly would land at the wrong pixel position in the mask.
            ox, oy = [int(v) for v in args.mask_roi_offset.split(",")]
            x_px_mask = max(0, x_px - ox)
            y_px_mask = max(0, y_px - oy)
        else:
            _, wall_mask = preprocess(image, px_per_unit=px_per_unit)
            ox, oy = 0, 0
            x_px_mask = x_px
            y_px_mask = y_px

        seg = detect_wall_at_point(wall_mask, x_px_mask, y_px_mask)
        if seg is None:
            print(json.dumps({"wall": None}))
        else:
            from preprocess import pixel_length, wall_angle_deg  # noqa
            # Translate segment coords from crop space back to full-image space.
            cx1, cy1, cx2, cy2 = seg["px_coords"]
            x1, y1, x2, y2 = cx1 + ox, cy1 + oy, cx2 + ox, cy2 + oy
            real_len = pixel_length(x1, y1, x2, y2) / px_per_unit
            angle = wall_angle_deg(x1, y1, x2, y2)
            facing = seg.get("facing") or (
                "North" if (y1 + y2) / 2 < h // 2 else "South"
                if abs(x2 - x1) >= abs(y2 - y1)
                else "West" if (x1 + x2) / 2 < w // 2 else "East"
            )
            result = {
                "wall": {
                    "px_coords": [x1, y1, x2, y2],
                    "x1_pct": x1 / w,
                    "y1_pct": y1 / h,
                    "x2_pct": x2 / w,
                    "y2_pct": y2 / h,
                    "facing": facing,
                    "length_raw": round(real_len, 2),
                    "length": f"{real_len:.2f} {cal['unit_label']}",
                    "image_size_px": [w, h],
                }
            }
            print(json.dumps(result))
        return

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

    # Debug capture: when ARQEN_DEBUG_DUMP=1 or a debug_runs/.capture marker
    # exists, save the exact request (image + scale + dpi + roi) so the run can
    # be replayed offline with Arqen/debug_pipeline.py.
    import os
    import shutil
    import time
    capture_marker = ARQEN_ROOT / "debug_runs" / ".capture"
    if os.environ.get("ARQEN_DEBUG_DUMP") == "1" or capture_marker.exists():
        try:
            run_dir = ARQEN_ROOT / "debug_runs" / time.strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(args.image, run_dir / "image.png")
            with open(run_dir / "request.json", "w") as f:
                json.dump({"scale": args.scale, "dpi": args.dpi, "roi": roi}, f, indent=2)
            print(f"  [debug-dump] saved request to {run_dir}", file=sys.stderr)
        except Exception as e:
            print(f"  [debug-dump] failed: {e}", file=sys.stderr)

    result = analyze_page(image, args.scale, args.dpi, roi=roi)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
