"""Tolerance-based comparison of pipeline runs against a stored baseline.

Used by tests/integration (golden-snapshot test) and compare_to_baseline.py
so a single tolerance policy gates every code change.
"""

from __future__ import annotations

# Comparison policy. Counts may drift by max(2, 10%); area by 5%;
# closure/coverage may not regress by more than the listed delta.
TOLERANCES = {
    "count_abs_min": 2,
    "count_rel": 0.10,
    "area_rel": 0.05,
    "px_per_ft_abs": 0.01,
    "closure_drop": 0.05,
    "coverage_delta": 0.10,
    "f1_drop": 0.02,
    "recall_drop": 0.02,
}

_COUNT_FIELDS = ("wall_count", "exterior_wall_count", "interior_wall_count", "room_count")


def _count_ok(baseline: int | None, current: int | None) -> bool:
    if baseline is None or current is None:
        return baseline == current
    allowed = max(TOLERANCES["count_abs_min"], TOLERANCES["count_rel"] * baseline)
    return abs(current - baseline) <= allowed


def compare_structural(case_id: str, baseline: dict, current: dict) -> list[str]:
    """Return a list of human-readable regression descriptions (empty = pass)."""
    failures: list[str] = []

    if bool(baseline.get("error")) != bool(current.get("error")):
        failures.append(
            f"{case_id}: error status changed "
            f"({baseline.get('error')!r} -> {current.get('error')!r})"
        )
        return failures
    if baseline.get("error"):
        return failures  # both errored the same way historically

    for fld in _COUNT_FIELDS:
        if not _count_ok(baseline.get(fld), current.get(fld)):
            failures.append(
                f"{case_id}: {fld} {baseline.get(fld)} -> {current.get(fld)} "
                f"(beyond +/-max({TOLERANCES['count_abs_min']}, "
                f"{TOLERANCES['count_rel']:.0%}))"
            )

    b_area, c_area = baseline.get("total_area_raw"), current.get("total_area_raw")
    if b_area and c_area:
        if abs(c_area - b_area) > TOLERANCES["area_rel"] * abs(b_area):
            failures.append(
                f"{case_id}: total_area {b_area} -> {c_area} "
                f"(beyond +/-{TOLERANCES['area_rel']:.0%})"
            )
    elif bool(b_area) != bool(c_area):
        failures.append(f"{case_id}: total_area {b_area} -> {c_area}")

    b_ppf, c_ppf = baseline.get("px_per_ft"), current.get("px_per_ft")
    if b_ppf is not None and c_ppf is not None:
        if abs(c_ppf - b_ppf) > TOLERANCES["px_per_ft_abs"]:
            failures.append(f"{case_id}: px_per_ft {b_ppf} -> {c_ppf}")

    b_net = baseline.get("wall_network_closure_rate")
    c_net = current.get("wall_network_closure_rate")
    if b_net is not None and c_net is not None:
        if c_net < b_net - TOLERANCES["closure_drop"]:
            failures.append(
                f"{case_id}: wall_network_closure_rate {b_net} -> {c_net} "
                f"(dropped more than {TOLERANCES['closure_drop']})"
            )

    b_cov = baseline.get("interior_coverage")
    c_cov = current.get("interior_coverage")
    if b_cov is not None and c_cov is not None:
        if abs(c_cov - b_cov) > TOLERANCES["coverage_delta"]:
            failures.append(
                f"{case_id}: interior_coverage {b_cov} -> {c_cov} "
                f"(beyond +/-{TOLERANCES['coverage_delta']})"
            )

    return failures


def compare_scores(case_id: str, baseline_scores: dict, current_scores: dict) -> list[str]:
    """Per-category F1/recall must not regress beyond tolerance."""
    failures: list[str] = []
    for category, b in (baseline_scores or {}).items():
        c = (current_scores or {}).get(category)
        if c is None:
            failures.append(f"{case_id}: category {category} missing from current scores")
            continue
        if c["f1"] < b["f1"] - TOLERANCES["f1_drop"]:
            failures.append(
                f"{case_id}: {category} F1 {b['f1']:.3f} -> {c['f1']:.3f}"
            )
        if c["recall"] < b["recall"] - TOLERANCES["recall_drop"]:
            failures.append(
                f"{case_id}: {category} recall {b['recall']:.3f} -> {c['recall']:.3f}"
            )
    return failures


def compare_case(case_id: str, baseline_case: dict, current_case: dict) -> list[str]:
    failures = compare_structural(
        case_id, baseline_case.get("structural", {}), current_case.get("structural", {}),
    )
    if baseline_case.get("scores"):
        failures += compare_scores(
            case_id, baseline_case["scores"], current_case.get("scores"),
        )
    return failures
