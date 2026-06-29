#!/usr/bin/env python3
"""Sweep ML window-detector thresholds and report window P/R/F1 per setting.

Finds the best precision/recall operating point for the hybrid window pipeline
by running the full pipeline (ML on) across a grid of confidence floors and
wall-distance gates, scoring windows in-memory (no prediction.json is touched).

The classical backbone is identical across settings; only the ML acceptance
thresholds change (read fresh from env on every call by window_detect_ml).

Usage:
    python validation/window_threshold_sweep.py
    python validation/window_threshold_sweep.py --conf 0.4 0.5 0.6 --wall 1.5 3 0
    python validation/window_threshold_sweep.py --out validation/reports/window_sweep.json

Output: a ranked table (by all-real F1) printed to stdout, plus a JSON dump.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Arqen"))

from arqen_validation.runner import load_manifest, run_case_pipeline  # noqa: E402
from arqen_validation.score import score_prediction  # noqa: E402

CASES_ROOT = ROOT / "cases"
DEFAULT_OUT = ROOT / "reports" / "window_sweep.json"


def _classify(case_id: str, gt_n: int, pred_n: int) -> str:
    if case_id.startswith("synth"):
        return "synth"
    if gt_n > 0:
        return "labeled"
    if pred_n > 0:
        return "fp_only"
    return "empty"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def _case_dirs(cases_root: Path) -> list[Path]:
    return sorted(
        p for p in cases_root.iterdir()
        if p.is_dir() and not p.name.startswith("_")
        and (p / "ground_truth.json").exists()
        and (p / "manifest.json").exists()
    )


def run_setting(case_dirs: list[Path], conf: float, wall_ft: float) -> dict:
    """Run pipeline (ML on) for one (conf, wall_ft) and aggregate window counts."""
    os.environ["ARQEN_WINDOW_ML"] = "1"
    os.environ["ARQEN_WINDOW_ML_CONF"] = str(conf)
    os.environ["ARQEN_WINDOW_ML_WALL_FT"] = str(wall_ft)

    buckets: dict[str, dict[str, int]] = {
        k: {"tp": 0, "fp": 0, "fn": 0} for k in ("labeled", "fp_only", "synth", "empty")
    }
    errors: list[str] = []
    for case_dir in case_dirs:
        try:
            manifest = load_manifest(case_dir)
            pred = run_case_pipeline(case_dir, manifest, write_prediction=False)
            if pred.get("error"):
                errors.append(f"{case_dir.name}: {pred['error']}")
                continue
            gt = json.loads((case_dir / "ground_truth.json").read_text(encoding="utf-8"))
            report = score_prediction(gt, pred, case_id=case_dir.name)
            counts = report["categories"].get("windows", {}).get(
                "counts", {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
            )
            tp = counts["true_positives"]
            fp = counts["false_positives"]
            fn = counts["false_negatives"]
            kind = _classify(case_dir.name, tp + fn, tp + fp)
            buckets[kind]["tp"] += tp
            buckets[kind]["fp"] += fp
            buckets[kind]["fn"] += fn
        except Exception as e:  # noqa: BLE001
            errors.append(f"{case_dir.name}: {e}")

    def agg(*kinds: str) -> dict:
        tp = sum(buckets[k]["tp"] for k in kinds)
        fp = sum(buckets[k]["fp"] for k in kinds)
        fn = sum(buckets[k]["fn"] for k in kinds)
        p, r, f1 = _prf(tp, fp, fn)
        return {"tp": tp, "fp": fp, "fn": fn, "p": round(p, 3), "r": round(r, 3), "f1": round(f1, 3)}

    return {
        "conf": conf,
        "wall_ft": wall_ft,
        "labeled": agg("labeled"),
        "fp_only": agg("fp_only"),
        "all_real": agg("labeled", "fp_only"),
        "synth": agg("synth"),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf", type=float, nargs="*", default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--wall", type=float, nargs="*", default=[1.5, 3.0, 0.0],
                        help="Wall-gate distances in ft; 0 disables the gate")
    parser.add_argument("--cases-root", default=str(CASES_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    case_dirs = _case_dirs(Path(args.cases_root))
    combos = [(c, w) for c in args.conf for w in args.wall]
    print(f"Sweeping {len(combos)} settings over {len(case_dirs)} cases "
          f"(~6-7 min each)...\n", flush=True)

    results = []
    t_start = time.time()
    for i, (conf, wall_ft) in enumerate(combos, 1):
        t0 = time.time()
        res = run_setting(case_dirs, conf, wall_ft)
        res["runtime_s"] = round(time.time() - t0, 1)
        results.append(res)
        lab, fpo, allr = res["labeled"], res["fp_only"], res["all_real"]
        print(
            f"[{i}/{len(combos)}] conf={conf} wall={wall_ft}ft  "
            f"labeled P/R/F1={lab['p']}/{lab['r']}/{lab['f1']}  "
            f"all-real F1={allr['f1']} (FP={allr['fp']})  "
            f"synthF1={res['synth']['f1']}  [{res['runtime_s']}s]",
            flush=True,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    results.sort(key=lambda r: (r["all_real"]["f1"], r["labeled"]["f1"]), reverse=True)
    print("\n=== Ranked by all-real window F1 ===")
    print(f"{'conf':>5} {'wall':>5} {'labP':>6} {'labR':>6} {'labF1':>6} "
          f"{'realP':>6} {'realF1':>6} {'realFP':>7} {'synthF1':>7}")
    print("-" * 70)
    for r in results:
        lab, allr = r["labeled"], r["all_real"]
        print(f"{r['conf']:>5} {r['wall_ft']:>5} "
              f"{lab['p']:>6} {lab['r']:>6} {lab['f1']:>6} "
              f"{allr['p']:>6} {allr['f1']:>6} {allr['fp']:>7} {r['synth']['f1']:>7}")

    print(f"\nTotal sweep time: {round((time.time() - t_start) / 60, 1)} min")
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
