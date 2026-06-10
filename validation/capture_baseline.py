#!/usr/bin/env python3
"""Capture baseline metrics for the current pipeline implementation.

For every case under validation/cases/ with a manifest.json:
  - run the pipeline (or reuse a static prediction for scorer-only cases)
  - record ground-truth-free structural metrics (counts, area, closure)
  - score against ground_truth.json where it exists (P/R/F1 per category)
  - write per-case report.json and a combined baselines/baseline.json

Usage:
  python validation/capture_baseline.py            # all cases
  python validation/capture_baseline.py --case synth_two_room --case mcginnies_pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Arqen"))

from arqen_validation.runner import (  # noqa: E402
    environment_info,
    load_manifest,
    run_case_pipeline,
    structural_summary,
)
from arqen_validation.score import score_prediction  # noqa: E402


def _score_summary(report: dict) -> dict:
    return {
        category: {
            "precision": data["precision"],
            "recall": data["recall"],
            "f1": data["f1"],
            "true_positives": data["counts"]["true_positives"],
            "false_positives": data["counts"]["false_positives"],
            "false_negatives": data["counts"]["false_negatives"],
        }
        for category, data in report["categories"].items()
    }


def _closure_summary(report: dict) -> dict | None:
    closure = report.get("closure")
    if not closure:
        return None
    return {
        "tolerance_px": closure["tolerance_px"],
        "wall_network_closure_rate": closure["wall_network"]["closure_rate"],
        "room_boundary_closure_rate": closure["room_boundary"]["closure_rate"],
        "mean_boundary_coverage": closure["room_boundary"]["mean_boundary_coverage"],
        "interior_coverage": (closure.get("interior_coverage") or {}).get("coverage"),
    }


def capture_case(case_dir: Path) -> dict | None:
    case_id = case_dir.name
    manifest = load_manifest(case_dir)

    if manifest.get("static_prediction"):
        pred_path = case_dir / "prediction.json"
        if not pred_path.exists():
            print(f"  [skip] {case_id}: static case without prediction.json")
            return None
        prediction = json.loads(pred_path.read_text(encoding="utf-8"))
    else:
        try:
            prediction = run_case_pipeline(case_dir, manifest)
        except FileNotFoundError as exc:
            print(f"  [skip] {case_id}: input missing ({exc})")
            return None

    entry: dict = {"structural": structural_summary(prediction), "scores": None,
                   "closure": None}

    gt_path = case_dir / "ground_truth.json"
    if gt_path.exists() and "error" not in prediction:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        report = score_prediction(gt, prediction, case_id=case_id)
        entry["scores"] = _score_summary(report)
        entry["closure"] = _closure_summary(report)
        with (case_dir / "report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", default=str(ROOT / "cases"))
    parser.add_argument("--out", default=str(ROOT / "baselines" / "baseline.json"))
    parser.add_argument("--case", action="append",
                        help="Limit to specific case id(s)")
    args = parser.parse_args()

    cases_root = Path(args.cases_root)
    case_dirs = sorted(
        p for p in cases_root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "manifest.json").exists()
    )
    if args.case:
        wanted = set(args.case)
        case_dirs = [p for p in case_dirs if p.name in wanted]

    baseline = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": environment_info(),
        "cases": {},
    }

    for case_dir in case_dirs:
        print(f"[capture] {case_dir.name} ...")
        t0 = time.time()
        entry = capture_case(case_dir)
        if entry is None:
            continue
        baseline["cases"][case_dir.name] = entry
        structural = entry["structural"]
        if structural.get("error"):
            print(f"  -> ERROR: {structural['error']} ({time.time() - t0:.1f}s)")
        else:
            line = (f"  -> walls={structural['wall_count']} "
                    f"rooms={structural['room_count']} "
                    f"area={structural['total_area_raw']} "
                    f"net_closure={structural['wall_network_closure_rate']} "
                    f"({time.time() - t0:.1f}s)")
            print(line)
            if entry["scores"]:
                for cat in ("rooms", "walls", "doors", "windows", "dimensions"):
                    s = entry["scores"].get(cat)
                    if s:
                        print(f"     {cat:10s} P/R/F1 = "
                              f"{s['precision']:.3f}/{s['recall']:.3f}/{s['f1']:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nBaseline written: {out_path} ({len(baseline['cases'])} case(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
