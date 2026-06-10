#!/usr/bin/env python3
"""Generate synthetic validation cases (rendered plan + exact ground truth).

Writes validation/cases/<name>/ with image.png, manifest.json, ground_truth.json.
With --score, also runs the pipeline and prints the scoring report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Arqen"))

import cv2  # noqa: E402

from arqen_validation.synth import ALL_PLANS  # noqa: E402


def write_case(name: str, cases_root: Path) -> Path:
    plan = ALL_PLANS[name]()
    case_dir = cases_root / name
    case_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(case_dir / "image.png"),
                cv2.cvtColor(plan.image, cv2.COLOR_RGB2BGR))

    manifest = {
        "id": name,
        "description": "Synthetic rendered plan with exact ground truth.",
        "image": "image.png",
        "scale": plan.scale_str,
        "dpi": plan.dpi,
        "roi": None,
        "doorway_close_ft": 2.5,
        "image_size_px": plan.image_size_px,
        "generated_by": "validation/generate_synth_cases.py",
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    (case_dir / "ground_truth.json").write_text(
        json.dumps(plan.ground_truth, indent=2), encoding="utf-8")
    return case_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", default=str(ROOT / "cases"))
    parser.add_argument("--score", action="store_true",
                        help="Run the pipeline and print scores after generating")
    args = parser.parse_args()

    cases_root = Path(args.cases_root)
    for name in ALL_PLANS:
        case_dir = write_case(name, cases_root)
        print(f"Wrote {case_dir}")

    if args.score:
        from arqen_validation.runner import run_case_pipeline, structural_summary
        from arqen_validation.score import score_prediction

        for name in ALL_PLANS:
            case_dir = cases_root / name
            prediction = run_case_pipeline(case_dir)
            summary = structural_summary(prediction)
            print(f"\n=== {name}: {json.dumps(summary)}")
            if "error" in prediction:
                continue
            gt = json.loads((case_dir / "ground_truth.json").read_text(encoding="utf-8"))
            report = score_prediction(gt, prediction, case_id=name)
            for cat, data in report["categories"].items():
                counts = data["counts"]
                print(f"  {cat:11s} P={data['precision']:.2f} R={data['recall']:.2f} "
                      f"F1={data['f1']:.2f} (TP={counts['true_positives']} "
                      f"FP={counts['false_positives']} FN={counts['false_negatives']})")
            closure = report["closure"]
            print(f"  closure: net={closure['wall_network']['closure_rate']} "
                  f"room={closure['room_boundary']['closure_rate']} "
                  f"cov={(closure['interior_coverage'] or {}).get('coverage')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
