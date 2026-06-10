#!/usr/bin/env python3
"""CLI for scoring Arqen extraction output against validation ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arqen_validation.score import score_all_cases, score_case, score_prediction  # noqa: E402


def _print_report(report: dict) -> None:
    summary = report["summary"]
    print(f"\nCase: {report['case_id']}")
    print(f"  Macro  P/R/F1: {summary['macro']['precision']:.3f} / "
          f"{summary['macro']['recall']:.3f} / {summary['macro']['f1']:.3f}")
    print(f"  Micro  P/R/F1: {summary['micro']['precision']:.3f} / "
          f"{summary['micro']['recall']:.3f} / {summary['micro']['f1']:.3f}")
    if summary.get("mean_iou") is not None:
        print(f"  Mean IoU: {summary['mean_iou']:.3f}")

    for category, data in report["categories"].items():
        counts = data["counts"]
        if counts["true_positives"] + counts["false_positives"] + counts["false_negatives"] == 0:
            continue
        iou = data.get("mean_iou")
        iou_str = f", IoU={iou:.3f}" if iou is not None else ""
        print(
            f"  {category:11s} P/R/F1={data['precision']:.3f}/"
            f"{data['recall']:.3f}/{data['f1']:.3f}{iou_str} "
            f"(TP={counts['true_positives']} FP={counts['false_positives']} "
            f"FN={counts['false_negatives']})"
        )
        if data["missing_objects"]:
            ids = [o.get("id", "?") for o in data["missing_objects"][:5]]
            print(f"    missing: {ids}{' …' if len(data['missing_objects']) > 5 else ''}")
        if data["false_positives"]:
            ids = [o.get("id", "?") for o in data["false_positives"][:5]]
            print(f"    false +: {ids}{' …' if len(data['false_positives']) > 5 else ''}")

    closure = report.get("closure")
    if closure:
        print(f"  Closure (tol={closure['tolerance_px']}px):")
        net = closure["wall_network"]
        if net["closure_rate"] is not None:
            print(
                f"    wall network: {net['closure_rate']:.3f} closed "
                f"({net['dangling_endpoints']}/{net['endpoint_count']} dangling endpoints)"
            )
        rb = closure["room_boundary"]
        if rb["closure_rate"] is not None:
            print(
                f"    room boundary: {rb['closure_rate']:.3f} closed "
                f"({rb['closed_rooms']}/{rb['room_count']} rooms, "
                f"mean coverage {rb['mean_boundary_coverage']:.3f})"
            )
        cov = closure.get("interior_coverage")
        if cov:
            print(f"    interior coverage: {cov['coverage']:.3f} of footprint area in rooms")


def _run_pipeline(case_dir: Path, manifest: dict) -> Path:
    arqen_dir = ROOT.parent / "Arqen"
    sys.path.insert(0, str(arqen_dir))

    import cv2  # noqa: WPS433
    from preprocess import analyze_page  # noqa: WPS433

    pdf_path = case_dir / manifest.get("pdf", "plan.pdf")
    if not pdf_path.exists() and manifest.get("pdf_path"):
        pdf_path = Path(manifest["pdf_path"])
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found for case: {pdf_path}")

    import fitz  # PyMuPDF  # noqa: WPS433

    dpi = int(manifest.get("dpi", 300))
    scale = manifest["scale"]
    roi = manifest.get("roi")

    doc = fitz.open(str(pdf_path))
    page = doc.load_page(manifest.get("page", 0))
    pix = page.get_pixmap(dpi=dpi)
    image = cv2.cvtColor(
        cv2.imdecode(
            __import__("numpy").frombuffer(pix.tobytes("png"), dtype=__import__("numpy").uint8),
            cv2.IMREAD_COLOR,
        ),
        cv2.COLOR_BGR2RGB,
    )

    result = analyze_page(
        image,
        scale,
        dpi,
        roi=roi,
        doorway_close_ft=manifest.get("doorway_close_ft", 2.5),
    )
    if "error" in result:
        raise RuntimeError(result["error"])

    out_path = case_dir / "prediction.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Case folder name under validation/cases/")
    parser.add_argument("--cases-root", default=str(ROOT / "cases"))
    parser.add_argument("--prediction", help="Path to prediction JSON")
    parser.add_argument("--ground-truth", help="Path to ground truth JSON (ad-hoc mode)")
    parser.add_argument("--output", help="Write report JSON to this path")
    parser.add_argument("--all", action="store_true", help="Score every case with prediction.json")
    parser.add_argument("--run-pipeline", action="store_true", help="Run preprocess.py before scoring")
    parser.add_argument("--threshold", action="append", metavar="CATEGORY=VALUE")
    args = parser.parse_args()

    thresholds = {}
    if args.threshold:
        for item in args.threshold:
            key, val = item.split("=", 1)
            thresholds[key.strip()] = float(val)

    if args.ground_truth and args.prediction:
        gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
        pred = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
        report = score_prediction(gt, pred, thresholds=thresholds or None)
        _print_report(report)
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    cases_root = Path(args.cases_root)

    if args.all:
        bundle = score_all_cases(
            cases_root,
            output_dir=Path(args.output) if args.output else ROOT / "reports",
            thresholds=thresholds or None,
        )
        print(f"Scored {bundle['case_count']} case(s)")
        for report in bundle["reports"]:
            _print_report(report)
        return 0

    if not args.case:
        parser.error("Provide --case, --all, or both --ground-truth and --prediction")

    case_dir = cases_root / args.case
    if not case_dir.exists():
        parser.error(f"Case not found: {case_dir}")

    prediction_path = Path(args.prediction) if args.prediction else None

    if args.run_pipeline:
        manifest_path = case_dir / "manifest.json"
        if not manifest_path.exists():
            parser.error(f"Missing manifest.json in {case_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prediction_path = _run_pipeline(case_dir, manifest)
        print(f"Wrote prediction: {prediction_path}")

    report = score_case(
        case_dir,
        prediction_path,
        output_path=Path(args.output) if args.output else case_dir / "report.json",
        thresholds=thresholds or None,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
