"""End-to-end pipeline on synthetic plans with exact ground truth.

Assertions here are stable structural floors; exact numbers are tracked by
the baseline snapshot (validation/capture_baseline.py + compare_to_baseline.py).
"""

import os

import pytest

from tests.helpers import assert_no_coaxial_spanning_duplicates

pytestmark = pytest.mark.integration


def _analyze(plan):
    from preprocess import analyze_page

    result = analyze_page(plan.image, plan.scale_str, plan.dpi)
    mask_path = result.pop("mask_cache_path", None)
    if mask_path and os.path.exists(mask_path):
        os.unlink(mask_path)
    return result


@pytest.fixture(scope="module")
def two_room_result(two_room_plan):
    return _analyze(two_room_plan)


@pytest.fixture(scope="module")
def two_room_report(two_room_plan, two_room_result):
    from arqen_validation.score import score_prediction

    return score_prediction(two_room_plan.ground_truth, two_room_result)


@pytest.fixture(scope="module")
def l_shape_result(l_shape_plan):
    return _analyze(l_shape_plan)


@pytest.fixture(scope="module")
def l_shape_report(l_shape_plan, l_shape_result):
    from arqen_validation.score import score_prediction

    return score_prediction(l_shape_plan.ground_truth, l_shape_result)


class TestTwoRoomStructure:
    def test_no_error(self, two_room_result):
        assert "error" not in two_room_result

    def test_calibration(self, two_room_result):
        assert two_room_result["px_per_ft"] == pytest.approx(18.0)
        assert two_room_result["units"] == "imperial"
        cal = two_room_result["calibration"]
        assert cal["status"] == "ok"
        assert cal["issues"] == []
        assert cal["dpi"] == 144

    def test_footprint_area_close_to_truth(self, two_room_result):
        # 60 x 40 ft = 2400 ft²; morphology inflates the footprint somewhat
        area = float(two_room_result["total_area"].split()[0])
        assert 2400 * 0.85 <= area <= 2400 * 1.35

    def test_rooms_detected(self, two_room_result):
        assert len(two_room_result["rooms"]) >= 2
        for room in two_room_result["rooms"]:
            assert room["area_raw"] >= 25.0

    def test_walls_have_required_fields(self, two_room_result):
        walls = two_room_result["walls"]
        assert len(walls) >= 5
        for w in walls:
            assert set(w) >= {"id", "facing", "length_raw", "px_coords"}
            assert w["length_raw"] > 0

    def test_exterior_walls_split_by_room(self, two_room_result):
        exterior = [w for w in two_room_result["walls"] if w.get("is_exterior")]
        assert exterior
        room_ids = {w.get("room_id") for w in exterior if w.get("room_id")}
        assert len(room_ids) >= 2

    def test_no_coaxial_spanning_duplicates(self, two_room_result):
        assert_no_coaxial_spanning_duplicates(
            two_room_result["walls"], two_room_result["px_per_ft"],
        )


class TestTwoRoomAccuracy:
    def test_room_recall(self, two_room_report):
        assert two_room_report["categories"]["rooms"]["recall"] >= 0.99

    def test_wall_recall_floor(self, two_room_report):
        assert two_room_report["categories"]["walls"]["recall"] >= 0.3

    def test_openings_not_detected_yet(self, two_room_report):
        # The CV path does not model doors/windows/dimensions today.
        # Update these when detection lands — they pin the known gap.
        for cat in ("doors", "windows", "dimensions"):
            assert two_room_report["categories"][cat]["counts"]["true_positives"] == 0

    def test_closure_floors(self, two_room_report):
        closure = two_room_report["closure"]
        assert closure["wall_network"]["closure_rate"] >= 0.7
        assert closure["room_boundary"]["mean_boundary_coverage"] >= 0.5
        assert closure["interior_coverage"]["coverage"] >= 0.6


class TestLShapeStructure:
    def test_no_error(self, l_shape_result):
        assert "error" not in l_shape_result

    def test_calibration_ok(self, l_shape_result):
        cal = l_shape_result["calibration"]
        assert cal["status"] == "ok"
        assert cal["issues"] == []

    def test_polygon_captures_notch(self, l_shape_result):
        # L-shape needs at least 6 vertices; rectangle would be 4
        assert l_shape_result["polygon_vertices"] >= 6

    def test_rooms_detected(self, l_shape_result):
        assert len(l_shape_result["rooms"]) >= 2

    def test_footprint_area_close_to_truth(self, l_shape_result):
        # 70x50 - 30x20 notch = 2900 ft²
        area = float(l_shape_result["total_area"].split()[0])
        assert 2900 * 0.7 <= area <= 2900 * 1.3

    def test_no_coaxial_spanning_duplicates(self, l_shape_result):
        assert_no_coaxial_spanning_duplicates(
            l_shape_result["walls"], l_shape_result["px_per_ft"],
        )


class TestLShapeAccuracy:
    def test_wall_recall_floor(self, l_shape_report):
        assert l_shape_report["categories"]["walls"]["recall"] >= 0.7

    def test_room_recall_floor(self, l_shape_report):
        assert l_shape_report["categories"]["rooms"]["recall"] >= 0.4

    def test_closure_floors(self, l_shape_report):
        closure = l_shape_report["closure"]
        assert closure["wall_network"]["closure_rate"] >= 0.6
        assert closure["interior_coverage"]["coverage"] >= 0.6


@pytest.fixture(scope="module")
def corridor_result(corridor_plan):
    return _analyze(corridor_plan)


@pytest.fixture(scope="module")
def corridor_report(corridor_plan, corridor_result):
    from arqen_validation.score import score_prediction

    return score_prediction(corridor_plan.ground_truth, corridor_result)


class TestCorridorStructure:
    def test_no_error(self, corridor_result):
        assert "error" not in corridor_result

    def test_all_four_rooms_detected(self, corridor_result):
        # Two compact rooms, the 4 ft corridor, and the 24 ft² closet.
        assert len(corridor_result["rooms"]) >= 4

    def test_closet_recovered_below_compact_floor(self, corridor_result):
        # The closet (24 ft² GT) is below the 25 ft² compact-room floor and
        # only survives via the corridor (aspect-aware) floor.
        small = [r for r in corridor_result["rooms"] if r["area_raw"] < 25.0]
        assert small
        for r in small:
            assert r["area_raw"] >= 8.0

    def test_corridor_detected(self, corridor_result):
        # One elongated cell ~58 x 4 ft (GT 232 ft²).
        def aspect(r):
            x0, y0, x1, y1 = r["bbox_px"]
            w, h = x1 - x0, y1 - y0
            return max(w, h) / max(min(w, h), 1)
        corridors = [
            r for r in corridor_result["rooms"]
            if aspect(r) >= 10 and 150 <= r["area_raw"] <= 300
        ]
        assert len(corridors) == 1

    def test_no_coaxial_spanning_duplicates(self, corridor_result):
        assert_no_coaxial_spanning_duplicates(
            corridor_result["walls"], corridor_result["px_per_ft"],
        )


class TestCorridorAccuracy:
    def test_room_precision_and_recall(self, corridor_report):
        rooms = corridor_report["categories"]["rooms"]
        assert rooms["precision"] >= 0.99
        assert rooms["recall"] >= 0.99

    def test_coverage_floor(self, corridor_report):
        closure = corridor_report["closure"]
        assert closure["interior_coverage"]["coverage"] >= 0.6

    def test_closure_floor(self, corridor_report):
        closure = corridor_report["closure"]
        assert closure["wall_network"]["closure_rate"] >= 0.7


class TestCalibrationGuards:
    def test_wrong_dpi_emits_warning(self, two_room_plan):
        from preprocess import analyze_page

        # Image rasterized at 144 DPI but caller declares 600 → ~4× inflated px/ft
        result = analyze_page(two_room_plan.image, two_room_plan.scale_str, dpi=600)
        mask_path = result.pop("mask_cache_path", None)
        if mask_path and os.path.exists(mask_path):
            os.unlink(mask_path)
        cal = result["calibration"]
        assert cal["status"] == "warning"
        codes = {i["code"] for i in cal["issues"]}
        assert "dpi_mismatch_suspected" in codes
        assert cal["suggested_dpi"] == 144

    def test_absurd_scale_emits_warning(self, two_room_plan):
        from preprocess import analyze_page

        result = analyze_page(two_room_plan.image, "1in=1ft", dpi=300)
        mask_path = result.pop("mask_cache_path", None)
        if mask_path and os.path.exists(mask_path):
            os.unlink(mask_path)
        cal = result["calibration"]
        assert cal["status"] == "warning"
        codes = {i["code"] for i in cal["issues"]}
        assert "px_per_unit_out_of_range" in codes
