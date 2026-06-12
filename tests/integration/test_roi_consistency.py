"""ROI placement should not materially change wall counts on the same plan."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPO_ROOT / "validation" / "cases"

pytestmark = pytest.mark.integration


def _run_trdi_with_roi(roi: dict) -> dict:
    from arqen_validation.runner import load_case_image, load_manifest, run_case_pipeline

    case_dir = CASES_ROOT / "trdi_overall"
    manifest = load_manifest(case_dir)
    manifest = dict(manifest)
    manifest["roi"] = roi
    try:
        load_case_image(case_dir, manifest)
    except FileNotFoundError as exc:
        pytest.skip(f"case input missing: {exc}")
    return run_case_pipeline(case_dir, manifest, write_prediction=False)


TRDI_ROI_VARIANTS = {
    "office_only": {
        "x0_pct": 0.30,
        "y0_pct": 0.02,
        "x1_pct": 0.85,
        "y1_pct": 0.50,
    },
    "nominal": {
        "x0_pct": 0.30,
        "y0_pct": 0.02,
        "x1_pct": 0.85,
        "y1_pct": 0.97,
    },
    "loose": {
        "x0_pct": 0.25,
        "y0_pct": 0.01,
        "x1_pct": 0.90,
        "y1_pct": 0.98,
    },
}


@pytest.fixture(scope="module")
def trdi_roi_results():
    results = {}
    for name, roi in TRDI_ROI_VARIANTS.items():
        results[name] = _run_trdi_with_roi(roi)
    return results


class TestTrdiRoiConsistency:
    def test_all_variants_structurally_sound(self, trdi_roi_results):
        for name, result in trdi_roi_results.items():
            assert "error" not in result, f"{name}: {result.get('error')}"
            walls = result.get("walls") or []
            assert len(walls) >= 15, f"{name}: only {len(walls)} walls"
            exterior = [w for w in walls if w.get("is_exterior")]
            facings = {w["facing"] for w in exterior}
            assert facings == {"North", "South", "East", "West"}, (
                f"{name}: incomplete perimeter {sorted(facings)}"
            )
            area = float(result["total_area"].split()[0])
            assert area > 4000, f"{name}: warehouse missing, area={area}"

    def test_wall_counts_within_tolerance(self, trdi_roi_results):
        counts = [len(r["walls"]) for r in trdi_roi_results.values()]
        lo, hi = min(counts), max(counts)
        assert hi <= lo * 1.15 + 1, (
            f"wall counts diverged too much: {dict(zip(trdi_roi_results, counts))}"
        )

    def test_office_hint_expands_analysis_roi(self, trdi_roi_results):
        office = trdi_roi_results["office_only"]
        ar = office.get("analysis_roi_pct")
        assert ar is not None
        hint = TRDI_ROI_VARIANTS["office_only"]
        hint_height = hint["y1_pct"] - hint["y0_pct"]
        expanded_height = ar["y1_pct"] - ar["y0_pct"]
        assert expanded_height > hint_height + 0.15
