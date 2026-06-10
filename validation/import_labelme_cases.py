#!/usr/bin/env python3
"""Import LabelMe floor-plan annotations as Arqen validation cases.

Example (your Downloads folder):
  python validation/import_labelme_cases.py \\
    --labelme-dir "C:/Users/jakep/Downloads/arqen-labs-jun/arqen-labs-jun/floor-plan-annotated" \\
    --images-root "C:/Users/jakep/Downloads/arqen-labs-jun/arqen-labs-jun/floor-plan-cropped" \\
    --pilot FP_86_2 \\
    --run-score

Import all 20:
  python validation/import_labelme_cases.py --labelme-dir ... --images-root ... --all --run-score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Arqen"))

from arqen_validation.labelme import import_labelme_case, recalibrate_case  # noqa: E402


def _print_score_summary(report: dict) -> None:
    print(f"\n=== Score: {report['case_id']} ===")
    for cat, data in report["categories"].items():
        c = data["counts"]
        total = c["true_positives"] + c["false_positives"] + c["false_negatives"]
        if total == 0:
            continue
        print(
            f"  {cat:11s} P/R/F1={data['precision']:.3f}/"
            f"{data['recall']:.3f}/{data['f1']:.3f} "
            f"(TP={c['true_positives']} FP={c['false_positives']} "
            f"FN={c['false_negatives']})"
        )
        if data["missing_objects"]:
            ids = [o.get("id", "?") for o in data["missing_objects"][:8]]
            print(f"    missing: {ids}")
        if data["false_positives"]:
            ids = [o.get("id", "?") for o in data["false_positives"][:8]]
            print(f"    false +: {ids}")
    closure = report.get("closure") or {}
    cov = (closure.get("interior_coverage") or {}).get("coverage")
    rb = closure.get("room_boundary") or {}
    if cov is not None:
        print(f"  interior coverage: {cov:.3f}")
    if rb.get("closure_rate") is not None:
        print(
            f"  room boundary closure: {rb['closure_rate']:.3f} "
            f"(mean cov {rb.get('mean_boundary_coverage')})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labelme-dir", default=None,
        help="Folder containing LabelMe JSON files (FP_*.json)",
    )
    parser.add_argument(
        "--images-root",
        help="Folder with cropped PNGs (default: sibling floor-plan-cropped)",
    )
    parser.add_argument(
        "--cases-root", default=str(ROOT / "cases"),
        help="Output validation/cases root",
    )
    parser.add_argument(
        "--case-prefix", default="labelme_",
        help="Prefix for case folder names (default: labelme_)",
    )
    parser.add_argument(
        "--scale", default=None,
        help="Architectural scale (default: auto-infer for cropped PNGs)",
    )
    parser.add_argument(
        "--dpi", type=int, default=None,
        help="Raster DPI (default: auto-infer from annotation span)",
    )
    parser.add_argument(
        "--assumed-span-ft", type=float, default=50.0,
        help="Assumed building span in feet for auto calibration",
    )
    parser.add_argument("--pilot", help="Import one stem only, e.g. FP_86_2")
    parser.add_argument("--all", action="store_true", help="Import every FP_*.json")
    parser.add_argument(
        "--run-score", action="store_true",
        help="Run pipeline + scorer after import",
    )
    parser.add_argument(
        "--recalibrate", action="store_true",
        help="Refresh scale/dpi in existing labelme_* manifests (no re-import)",
    )
    args = parser.parse_args()

    if args.recalibrate:
        cases_root = Path(args.cases_root)
        case_dirs = sorted(
            p for p in cases_root.iterdir()
            if p.is_dir() and p.name.startswith(args.case_prefix)
        )
        if not case_dirs:
            print(f"No {args.case_prefix}* cases under {cases_root}")
            return 1
        for case_dir in case_dirs:
            try:
                info = recalibrate_case(case_dir, assumed_span_ft=args.assumed_span_ft)
            except FileNotFoundError as exc:
                print(f"  SKIP {case_dir.name}: {exc}")
                continue
            print(
                f"[recalibrate] {info['case_id']}: "
                f"px/ft={info['inferred_px_per_ft']} "
                f"(hyp={info['calibration_hypothesis_ft']} ft, dpi={info['dpi']})"
            )
        if args.run_score:
            from arqen_validation.runner import run_case_pipeline  # noqa: E402
            from arqen_validation.score import score_prediction  # noqa: E402

            for case_dir in case_dirs:
                print(f"[pipeline] {case_dir.name} ...")
                prediction = run_case_pipeline(case_dir)
                if prediction.get("error"):
                    print(f"  ERROR: {prediction['error']}")
                    continue
                gt = json.loads((case_dir / "ground_truth.json").read_text(encoding="utf-8"))
                report = score_prediction(gt, prediction, case_id=case_dir.name)
                (case_dir / "report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8",
                )
                _print_score_summary(report)
        return 0

    if not args.labelme_dir:
        parser.error("--labelme-dir is required unless --recalibrate")
    labelme_dir = Path(args.labelme_dir)
    if not labelme_dir.is_dir():
        parser.error(f"Not a directory: {labelme_dir}")

    images_root = Path(args.images_root) if args.images_root else (
        labelme_dir.parent / "floor-plan-cropped"
    )
    cases_root = Path(args.cases_root)

    json_files = sorted(labelme_dir.glob("FP_*.json"))
    if args.pilot:
        json_files = [labelme_dir / f"{args.pilot}.json"]
        if not json_files[0].exists():
            json_files = [labelme_dir / args.pilot]
        if not json_files[0].exists():
            parser.error(f"LabelMe file not found for pilot: {args.pilot}")
    elif not args.all:
        parser.error("Provide --pilot NAME or --all")

    imported = []
    for jpath in json_files:
        case_id = f"{args.case_prefix}{jpath.stem.lower()}"
        case_dir = cases_root / case_id
        print(f"[import] {jpath.name} -> {case_dir.name}")
        try:
            summary = import_labelme_case(
                jpath,
                case_dir,
                images_root=images_root,
                scale=args.scale,
                dpi=args.dpi,
                assumed_span_ft=args.assumed_span_ft,
            )
        except FileNotFoundError as exc:
            print(f"  SKIP {exc}")
            continue
        print(f"  rooms={summary['counts']['rooms']} walls={summary['counts']['walls']} "
              f"doors={summary['counts']['doors']} windows={summary['counts']['windows']}")
        skipped = summary["report"].get("skipped_labels") or {}
        if skipped:
            print(f"  skipped labels: {skipped}")
        wall_warn = len(summary["report"].get("wall_centerline_warnings") or [])
        if wall_warn:
            print(f"  wall polygons approximated to centerlines: {wall_warn}")
        if summary["report"].get("inferred_px_per_ft"):
            print(f"  inferred px/ft: {summary['report']['inferred_px_per_ft']} "
                  f"(dpi={summary['report'].get('inferred_dpi')})")
        imported.append(case_dir)

    if args.run_score and imported:
        from arqen_validation.runner import run_case_pipeline  # noqa: E402
        from arqen_validation.score import score_prediction  # noqa: E402

        for case_dir in imported:
            print(f"[pipeline] {case_dir.name} ...")
            prediction = run_case_pipeline(case_dir)
            if prediction.get("error"):
                print(f"  ERROR: {prediction['error']}")
                continue
            gt = json.loads((case_dir / "ground_truth.json").read_text(encoding="utf-8"))
            report = score_prediction(gt, prediction, case_id=case_dir.name)
            (case_dir / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8",
            )
            _print_score_summary(report)

    print(f"\nImported {len(imported)} case(s) under {cases_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
