"""
debug_windows.py — Window detection debug harness.

Runs the CV pipeline through door detection, then evaluates every window
candidate with acceptance/rejection reasons and saves an annotated overlay.

Usage:
  python debug_windows.py --image plan.png --scale "3/8in=1ft" --dpi 150 \\
      --out debug_runs/windows

  python debug_windows.py --image plan.png --scale "1in=16ft" --dpi 150 \\
      --roi 0,0.05,1,0.95 --out debug_runs/windows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import preprocess as pp  # noqa: E402
from door_detect import detect_doors, ink_mask_from_image  # noqa: E402
from window_detect import detect_window_candidates, detect_windows  # noqa: E402

ACCEPT_COLOR = (0, 200, 0)
REJECT_COLOR = (0, 0, 255)
WALL_COLOR = (180, 180, 180)
CANDIDATE_COLOR = (255, 180, 0)


def _parse_roi(s: str) -> dict:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x0,y0,x1,y1 fractions")
    return {"x0": parts[0], "y0": parts[1], "x1": parts[2], "y1": parts[3]}


def _draw_bbox(canvas: np.ndarray, bbox: list, color: tuple, label: str) -> None:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        canvas, label, (x0 + 4, max(16, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    roi = _parse_roi(args.roi) if args.roi else None
    result = pp.analyze_page(
        image_rgb,
        args.scale,
        args.dpi,
        roi=roi,
        crop_mode=bool(roi),
    )
    if "error" in result:
        raise SystemExit(f"analyze_page failed: {result['error']}")

    walls = result.get("walls") or []
    windows = result.get("windows") or []

    # Re-run candidate detection on crop-frame geometry when ROI was used.
    h, w = image_rgb.shape[:2]
    if roi:
        x0 = int(roi["x0"] * w)
        y0 = int(roi["y0"] * h)
        x1 = int(roi["x1"] * w)
        y1 = int(roi["y1"] * h)
        crop = image_rgb[y0:y1, x0:x1]
        _, wall_pair_mask = pp.preprocess(crop, result["px_per_ft"], crop_mode=True)
        ink_mask = ink_mask_from_image(crop)
        px_per_unit = result["px_per_ft"]
        crop_walls = []
        for wall in walls:
            c = wall.get("px_coords")
            if not c:
                continue
            crop_walls.append({
                **wall,
                "px_coords": [c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0],
            })
        doors = detect_doors(
            crop_walls, wall_pair_mask, ink_mask, px_per_unit, crop_mode=True,
        )
        candidates = detect_window_candidates(
            crop_walls, wall_pair_mask, ink_mask, px_per_unit,
            crop_mode=True, doors=doors,
        )
        for c in candidates:
            if "bbox_px" in c:
                b = c["bbox_px"]
                c["bbox_px"] = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
            if c.get("window"):
                b = c["window"]["bbox_px"]
                c["window"]["bbox_px"] = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
        canvas = image_bgr.copy()
    else:
        _, wall_pair_mask = pp.preprocess(image_rgb, result["px_per_ft"])
        ink_mask = ink_mask_from_image(image_rgb)
        px_per_unit = result["px_per_ft"]
        doors = detect_doors(walls, wall_pair_mask, ink_mask, px_per_unit)
        candidates = detect_window_candidates(
            walls, wall_pair_mask, ink_mask, px_per_unit, doors=doors,
        )
        canvas = image_bgr.copy()

    exterior = [w for w in walls if w.get("is_exterior")]
    for wall in exterior:
        c = wall.get("px_coords") or []
        if len(c) >= 4:
            cv2.line(canvas, c[:2], c[2:], WALL_COLOR, 2)

    for i, cand in enumerate(candidates):
        bbox = cand.get("bbox_px") or (cand.get("window") or {}).get("bbox_px")
        if not bbox:
            continue
        if cand["status"] == "accepted":
            color = ACCEPT_COLOR
            tag = f"ok:{cand.get('strategy', '?')}"
        else:
            color = REJECT_COLOR
            tag = cand.get("reject_reason") or "rejected"
        _draw_bbox(canvas, bbox, color, f"{i}:{tag}")

    for w in windows:
        _draw_bbox(canvas, w["bbox_px"], CANDIDATE_COLOR, w.get("id", "win"))

    overlay_path = out_dir / "window_debug_overlay.png"
    cv2.imwrite(str(overlay_path), canvas)

    summary = {
        "image": args.image,
        "scale": args.scale,
        "dpi": args.dpi,
        "windows_emitted": len(windows),
        "candidates_total": len(candidates),
        "candidates_accepted": sum(1 for c in candidates if c["status"] == "accepted"),
        "reject_reasons": {},
        "windows": windows,
        "candidates": candidates,
    }
    for c in candidates:
        if c["status"] == "rejected":
            reason = c.get("reject_reason") or "unknown"
            summary["reject_reasons"][reason] = summary["reject_reasons"].get(reason, 0) + 1

    json_path = out_dir / "window_candidates.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pred_path = out_dir / "prediction_snippet.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_size_px": result.get("image_size_px"),
                "px_per_ft": result.get("px_per_ft"),
                "windows": windows,
            },
            f,
            indent=2,
        )

    print(f"Saved overlay: {overlay_path}")
    print(f"Saved candidates: {json_path}")
    print(
        f"Windows: {len(windows)} emitted, "
        f"{summary['candidates_accepted']}/{len(candidates)} candidates accepted"
    )
    if summary["reject_reasons"]:
        print("Reject reasons:", summary["reject_reasons"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="Plan image (PNG/JPG)")
    ap.add_argument("--scale", required=True, help='Scale string e.g. "3/8in=1ft"')
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--roi", help="Optional ROI fractions x0,y0,x1,y1")
    ap.add_argument("--out", default="debug_runs/windows")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
