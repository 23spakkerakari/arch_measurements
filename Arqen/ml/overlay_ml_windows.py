#!/usr/bin/env python3
"""Overlay raw ML window detections (and GT windows) on validation case images.

Purpose: sanity-check whether the false positives the scorer reports on
"fp_only" plans (0 ground-truth windows) are genuine errors or just windows the
annotator never labeled. We draw EVERY raw ML detection (no wall gate) with its
confidence, plus any GT window boxes, so a human can eyeball the difference.

Usage:
    python Arqen/ml/overlay_ml_windows.py
    python Arqen/ml/overlay_ml_windows.py --cases labelme_fp_1 labelme_fp_35_1
    python Arqen/ml/overlay_ml_windows.py --conf 0.5 --out debug_runs/ml_fp_probe

Outputs one PNG per case under the output dir. Requires ultralytics + the
trained weights (Arqen/ml/weights/window_yolo.pt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ARQEN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARQEN_DIR.parent
sys.path.insert(0, str(ARQEN_DIR))

import window_detect_ml as wml  # noqa: E402

CASES_ROOT = REPO_ROOT / "validation" / "cases"
DEFAULT_OUT = REPO_ROOT / "debug_runs" / "ml_fp_probe"
# A few zero-GT ("fp_only") plans that drove the precision hit, by FP count.
DEFAULT_CASES = ["labelme_fp_1", "labelme_fp_35_1", "labelme_fp_35_2", "labelme_fp_25_1"]


def _px_per_ft(manifest: dict) -> float:
    if manifest.get("inferred_px_per_ft"):
        return float(manifest["inferred_px_per_ft"])
    try:
        from scale_parse import parse_scale
        cal = parse_scale(manifest["scale"], int(manifest.get("dpi", 300)), output_unit="ft")
        return float(cal["px_per_unit"])
    except Exception:
        return 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if wml._load_model() is None:
        print("ERROR: model unavailable (need ultralytics + Arqen/ml/weights/window_yolo.pt)",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_id in args.cases:
        case_dir = CASES_ROOT / case_id
        manifest_path = case_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"  SKIP {case_id}: no manifest")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        img_path = case_dir / manifest.get("image", "image.png")
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  SKIP {case_id}: cannot read {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ppf = _px_per_ft(manifest)

        # Raw ML detections, no wall gate, so we see everything the model fires on.
        dets = wml.detect_windows_ml(rgb, ppf, walls=None, conf=args.conf)

        gt = {}
        gt_path = case_dir / "ground_truth.json"
        if gt_path.exists():
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_windows = gt.get("windows", [])

        vis = bgr.copy()
        for w in gt_windows:
            x0, y0, x1, y1 = (int(v) for v in w["bbox_px"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 3)  # GT: green
        for d in dets:
            x0, y0, x1, y1 = (int(v) for v in d["bbox_px"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)  # ML: red
            cv2.putText(vis, f"{d['confidence']:.2f}", (x0, max(0, y0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        out_path = out_dir / f"{case_id}.png"
        cv2.imwrite(str(out_path), vis)
        print(f"  {case_id}: {len(dets)} ML dets (conf>={args.conf}), "
              f"{len(gt_windows)} GT windows -> {out_path}")

    print(f"\nGreen=GT windows, Red=ML detections. Overlays in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
