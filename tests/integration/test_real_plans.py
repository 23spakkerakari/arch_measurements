"""End-to-end pipeline on representative real floor plans.

Inputs are repo assets (web-app captures + test.pdf). There is no human
ground truth yet, so these tests assert structural soundness; exact values
are tracked by the committed baseline (validation/baselines/baseline.json).
"""

import json
from pathlib import Path

import pytest

from tests.helpers import assert_no_coaxial_spanning_duplicates

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPO_ROOT / "validation" / "cases"


def _run_case(case_id):
    from arqen_validation.runner import (
        load_case_image,
        load_manifest,
        run_case_pipeline,
    )

    case_dir = CASES_ROOT / case_id
    manifest = load_manifest(case_dir)
    try:
        load_case_image(case_dir, manifest)
    except FileNotFoundError as exc:
        pytest.skip(f"case input missing: {exc}")
    return run_case_pipeline(case_dir, manifest, write_prediction=False)


@pytest.fixture(scope="module")
def capture_165134():
    return _run_case("capture_165134")


@pytest.fixture(scope="module")
def capture_153430():
    return _run_case("capture_153430")


@pytest.fixture(scope="module")
def mcginnies():
    return _run_case("mcginnies_pdf")


@pytest.fixture(scope="module")
def trdi_overall():
    return _run_case("trdi_overall")


def _assert_sound(result, min_walls=5, min_rooms=1, allow_calibration_warnings=False):
    assert "error" not in result, f"pipeline error: {result.get('error')}"
    assert result["px_per_ft"] > 0
    assert len(result["footprint_polygon_px"]) >= 4
    assert float(result["total_area"].split()[0]) > 0

    walls = result["walls"]
    rooms = result["rooms"]
    assert len(walls) >= min_walls
    assert len(rooms) >= min_rooms
    for w in walls:
        assert w["length_raw"] > 0
        x1, y1, x2, y2 = w["px_coords"]
        assert (x1, y1) != (x2, y2)
        assert w["facing"] in {"North", "South", "East", "West"}
    for r in rooms:
        assert r["area_px"] > 0
        assert len(r["bbox_px"]) == 4

    assert_no_coaxial_spanning_duplicates(walls, result["px_per_ft"])

    cal = result.get("calibration")
    assert cal is not None, "missing calibration block"
    if allow_calibration_warnings:
        severities = {i["severity"] for i in cal.get("issues", [])}
        assert severities <= {"warning"}, f"calibration errors: {cal.get('issues')}"
    else:
        assert cal["status"] == "ok", f"unexpected calibration issues: {cal.get('issues')}"
        assert cal["issues"] == []


class TestCapture165134:
    def test_structurally_sound(self, capture_165134):
        _assert_sound(capture_165134, min_walls=8, min_rooms=2)

    def test_exterior_walls_have_rooms(self, capture_165134):
        exterior = [w for w in capture_165134["walls"] if w.get("is_exterior")]
        assert exterior
        with_room = [w for w in exterior if w.get("room_id")]
        assert with_room, "no exterior sub-segment carries a room_id"

    def test_calibration(self, capture_165134):
        # 3/8" = 1 ft @ 144 DPI -> 54 px/ft
        assert capture_165134["px_per_ft"] == pytest.approx(54.0)


class TestCapture153430:
    def test_structurally_sound(self, capture_153430):
        _assert_sound(capture_153430, min_walls=8, min_rooms=1)


class TestMcGinniesPdf:
    def test_structurally_sound(self, mcginnies):
        _assert_sound(mcginnies, min_walls=10, min_rooms=1)

    def test_calibration(self, mcginnies):
        # 1 in = 16 ft @ 150 DPI -> 9.375 px/ft
        assert mcginnies["px_per_ft"] == pytest.approx(9.38, abs=0.01)


class TestTrdiOverall:
    """TRDI Office & Warehouse A1.0 at production resolution (thin strokes).

    Regression for the thin-stroke footprint collapse: the strict footprint
    threshold erased the 1-2 px pair-mask strokes, the footprint shrank to the
    fire wall, and the entire warehouse/exterior perimeter was dropped by
    drop_segments_outside_exterior (5 walls / 227 ft² instead of the full
    building).
    """

    def test_structurally_sound(self, trdi_overall):
        # dpi 71 (144 capped to MAX_ANALYSIS_PX) triggers a calibration
        # warning by design; the warning is part of what this case captures.
        _assert_sound(
            trdi_overall, min_walls=15, min_rooms=4,
            allow_calibration_warnings=True,
        )

    def test_exterior_perimeter_on_all_facings(self, trdi_overall):
        exterior = [w for w in trdi_overall["walls"] if w.get("is_exterior")]
        facings = {w["facing"] for w in exterior}
        assert facings == {"North", "South", "East", "West"}, (
            f"perimeter incomplete, exterior facings: {sorted(facings)}"
        )

    def test_warehouse_included_in_footprint(self, trdi_overall):
        # Office block alone is ~2000 ft²; with the warehouse the footprint
        # is ~5700 ft². The footprint must also be tall enough to span the
        # warehouse south of the fire wall (~100 ft overall vs ~40 ft office).
        area = float(trdi_overall["total_area"].split()[0])
        assert area > 4000, f"warehouse missing from footprint: {area} ft²"

        x0, y0, x1, y1 = trdi_overall["footprint_bbox_px"]
        height_ft = (y1 - y0) / trdi_overall["px_per_ft"]
        assert height_ft > 80, f"footprint height only {height_ft:.0f} ft"


class TestBaselineSnapshot:
    """Golden-snapshot regression: current runs vs committed baseline."""

    @pytest.fixture(scope="class")
    def baseline(self):
        path = REPO_ROOT / "validation" / "baselines" / "baseline.json"
        if not path.exists():
            pytest.skip("baseline not captured yet (run validation/capture_baseline.py)")
        return json.loads(path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize("case_id", ["capture_165134", "capture_153430"])
    def test_capture_matches_baseline(self, baseline, case_id, request):
        from arqen_validation.compare import compare_structural
        from arqen_validation.runner import structural_summary

        if case_id not in baseline["cases"]:
            pytest.skip(f"{case_id} not in baseline")
        result = request.getfixturevalue(case_id)
        current = structural_summary(result)
        failures = compare_structural(
            case_id, baseline["cases"][case_id]["structural"], current,
        )
        assert not failures, "; ".join(failures)
