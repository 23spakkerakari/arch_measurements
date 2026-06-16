#!/usr/bin/env python3
"""Window-only validation metrics: per-case and aggregate P/R/F1.

Single source of truth for measuring window-detection accuracy across every
phase of the "Window Accuracy V2" work. Runs identically whether or not the
pipeline is re-run, so before/after comparisons stay honest.

Usage:
    # Score existing prediction.json files
    python validation/window_metrics.py

    # Re-run the pipeline first, then score
    python validation/window_metrics.py --run-pipeline

    # Restrict to specific cases
    python validation/window_metrics.py --cases synth_two_room labelme_fp_27

Case taxonomy (printed in the summary):
    - synth:   synthetic plans with exact GT (regression guardrail, must stay ~1.0)
    - labeled: real plans with >0 GT windows (recall + precision target)
    - fp_only: real plans with 0 GT windows (precision guardrail, FPs must stay low)
    - empty:   no GT windows and no predictions (vacuous, ignored in aggregates)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arqen_validation.runner import load_manifest, run_case_pipeline  # noqa: E402
from arqen_validation.score import score_case  # noqa: E402

CASES_ROOT = ROOT / "cases"


def _classify(case_id: str, counts: dict) -> str:
    gt = counts["true_positives"] + counts["false_negatives"]
    pred = counts["true_positives"] + counts["false_positives"]
    if case_id.startswith("synth"):
        return "synth"
    if gt > 0:
        return "labeled"
    if pred > 0:
        return "fp_only"
    return "empty"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def collect(case_dirs: list[Path], run_pipeline: bool) -> list[dict]:
    rows: list[dict] = []
    for case_dir in case_dirs:
        gt_path = case_dir / "ground_truth.json"
        if not gt_path.exists():
            continue
        if run_pipeline:
            try:
                run_case_pipeline(case_dir, load_manifest(case_dir), write_prediction=True)
            except Exception as e:  # noqa: BLE001
                rows.append({"case_id": case_dir.name, "error": str(e)})
                continue
        if not (case_dir / "prediction.json").exists():
            continue
        try:
            report = score_case(case_dir)
        except Exception as e:  # noqa: BLE001
            rows.append({"case_id": case_dir.name, "error": str(e)})
            continue
        win = report["categories"].get("windows", {})
        counts = win.get("counts", {"true_positives": 0, "false_positives": 0, "false_negatives": 0})
        rows.append({
            "case_id": report["case_id"],
            "kind": _classify(report["case_id"], counts),
            "precision": win.get("precision", 0.0),
            "recall": win.get("recall", 0.0),
            "f1": win.get("f1", 0.0),
            "tp": counts["true_positives"],
            "fp": counts["false_positives"],
            "fn": counts["false_negatives"],
        })
    return rows


def summarize(rows: list[dict]) -> None:
    errors = [r for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    rows.sort(key=lambda r: (r["kind"], -r["f1"], r["case_id"]))

    print(f"\n{'case':28s} {'kind':8s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    print("-" * 78)
    for r in rows:
        print(
            f"{r['case_id']:28s} {r['kind']:8s} "
            f"{r['precision']:6.2f} {r['recall']:6.2f} {r['f1']:6.2f} "
            f"{r['tp']:4d} {r['fp']:4d} {r['fn']:4d}"
        )

    def agg(kinds: set[str]) -> dict:
        sub = [r for r in rows if r["kind"] in kinds]
        tp = sum(r["tp"] for r in sub)
        fp = sum(r["fp"] for r in sub)
        fn = sum(r["fn"] for r in sub)
        p, rec, f1 = _prf(tp, fp, fn)
        return {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": rec, "f1": f1, "n": len(sub)}

    print("\nAggregates (micro):")
    for label, kinds in (
        ("labeled (real, GT>0)", {"labeled"}),
        ("fp_only (real, GT=0)", {"fp_only"}),
        ("all real (labeled+fp_only)", {"labeled", "fp_only"}),
        ("synth", {"synth"}),
    ):
        a = agg(kinds)
        print(
            f"  {label:30s} n={a['n']:2d}  "
            f"P={a['p']:.3f} R={a['r']:.3f} F1={a['f1']:.3f}  "
            f"TP={a['tp']} FP={a['fp']} FN={a['fn']}"
        )

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e['case_id']}: {e['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pipeline", action="store_true", help="Re-run analyze_page before scoring")
    parser.add_argument("--cases", nargs="*", help="Restrict to these case folder names")
    parser.add_argument("--cases-root", default=str(CASES_ROOT))
    args = parser.parse_args()

    cases_root = Path(args.cases_root)
    if args.cases:
        case_dirs = [cases_root / c for c in args.cases]
    else:
        case_dirs = sorted(
            p for p in cases_root.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )

    rows = collect(case_dirs, run_pipeline=args.run_pipeline)
    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
