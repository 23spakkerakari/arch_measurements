#!/usr/bin/env python3
"""Run pipeline + scorer on all imported labelme_* cases and print a summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Arqen"))

from arqen_validation.runner import run_case_pipeline  # noqa: E402
from arqen_validation.score import score_prediction  # noqa: E402


def main() -> int:
    cases_root = ROOT / "cases"
    cases = sorted(p for p in cases_root.iterdir() if p.name.startswith("labelme_"))
    if not cases:
        print("No labelme_* cases found. Run validation/import_labelme_cases.py first.")
        return 1

    print(f"Scoring {len(cases)} labelme case(s)...\n")
    rows = []
    for case_dir in cases:
        try:
            pred = run_case_pipeline(case_dir)
        except Exception as exc:
            rows.append((case_dir.name, f"ERROR: {exc}", "", "", ""))
            continue
        if pred.get("error"):
            rows.append((case_dir.name, pred["error"], "", "", ""))
            continue
        gt = json.loads((case_dir / "ground_truth.json").read_text(encoding="utf-8"))
        report = score_prediction(gt, pred, case_id=case_dir.name)
        (case_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        rooms = report["categories"]["rooms"]
        walls = report["categories"]["walls"]
        cov = (report.get("closure") or {}).get("interior_coverage") or {}
        rows.append((
            case_dir.name,
            f"{len(pred.get('rooms', []))}/{len(gt.get('rooms', []))}",
            f"{rooms['recall']:.2f}",
            f"{walls['recall']:.2f}",
            f"{(cov.get('coverage') or 0):.2f}",
        ))

    print(f"{'case':<22} {'pred/gt rooms':<14} {'room R':<8} {'wall R':<8} {'coverage'}")
    print("-" * 62)
    for row in rows:
        print(f"{row[0]:<22} {row[1]:<14} {row[2]:<8} {row[3]:<8} {row[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
