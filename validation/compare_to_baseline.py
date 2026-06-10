#!/usr/bin/env python3
"""Re-run all baselined cases and compare against the stored baseline.

This is the regression gate for every code change: run it after modifying
pipeline code and before merging. Exits non-zero on any regression beyond
the tolerances in arqen_validation/compare.py.

Usage:
  python validation/compare_to_baseline.py
  python validation/compare_to_baseline.py --case synth_two_room
  python validation/compare_to_baseline.py --update   # re-capture baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Arqen"))

from arqen_validation.compare import compare_case  # noqa: E402
from capture_baseline import capture_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", default=str(ROOT / "cases"))
    parser.add_argument("--baseline", default=str(ROOT / "baselines" / "baseline.json"))
    parser.add_argument("--case", action="append", help="Limit to specific case id(s)")
    parser.add_argument("--update", action="store_true",
                        help="Re-capture the baseline instead of comparing")
    args = parser.parse_args()

    if args.update:
        cmd = [sys.executable, str(ROOT / "capture_baseline.py")]
        for c in args.case or []:
            cmd += ["--case", c]
        return subprocess.call(cmd)

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"No baseline at {baseline_path}. "
              f"Run: python validation/capture_baseline.py")
        return 2

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    cases_root = Path(args.cases_root)

    case_ids = list(baseline["cases"])
    if args.case:
        wanted = set(args.case)
        case_ids = [c for c in case_ids if c in wanted]

    all_failures: list[str] = []
    for case_id in case_ids:
        case_dir = cases_root / case_id
        if not case_dir.exists():
            all_failures.append(f"{case_id}: case folder missing")
            continue
        print(f"[compare] {case_id} ...")
        current = capture_case(case_dir)
        if current is None:
            all_failures.append(f"{case_id}: could not run (input missing?)")
            continue
        failures = compare_case(case_id, baseline["cases"][case_id], current)
        if failures:
            for f in failures:
                print(f"  FAIL {f}")
            all_failures += failures
        else:
            print("  PASS")

    print()
    if all_failures:
        print(f"REGRESSIONS: {len(all_failures)}")
        for f in all_failures:
            print(f"  - {f}")
        return 1
    print(f"All {len(case_ids)} case(s) within tolerance of baseline "
          f"({baseline['created_utc']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
