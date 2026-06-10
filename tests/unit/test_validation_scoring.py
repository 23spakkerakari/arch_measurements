"""Validation framework itself — matchers, metrics, normalize, closure, schema."""

import json
from pathlib import Path

import pytest

from arqen_validation.closure import (
    compute_closure,
    derive_tolerance_px,
    interior_coverage,
    point_to_segment_distance,
    room_boundary_closure,
    wall_network_closure,
)
from arqen_validation.geometry import (
    bbox_iou,
    segment_overlap_iou,
    value_within_tolerance,
)
from arqen_validation.matchers import greedy_match, wall_score
from arqen_validation.metrics import build_report
from arqen_validation.normalize import normalize_document
from arqen_validation.score import score_prediction

pytestmark = pytest.mark.unit

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "validation"

RECT_WALLS = [
    {"id": "n", "px_coords": [0, 0, 100, 0]},
    {"id": "e", "px_coords": [100, 0, 100, 100]},
    {"id": "s", "px_coords": [100, 100, 0, 100]},
    {"id": "w", "px_coords": [0, 100, 0, 0]},
]


class TestGeometry:
    def test_bbox_iou_identity(self):
        assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_bbox_iou_disjoint(self):
        assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_segment_overlap_full(self):
        assert segment_overlap_iou([0, 0, 100, 0], [0, 0, 100, 0]) == pytest.approx(1.0)

    def test_segment_overlap_perpendicular_zero(self):
        assert segment_overlap_iou([0, 0, 100, 0], [50, 0, 50, 100]) == 0.0

    def test_value_tolerance(self):
        assert value_within_tolerance(20.5, 20.0, rel_tol=0.05)
        assert not value_within_tolerance(25.0, 20.0, rel_tol=0.05)


class TestMatchers:
    def test_empty_is_perfect(self):
        r = greedy_match("walls", [], [], wall_score, 0.5)
        assert (r.precision, r.recall, r.f1) == (1.0, 1.0, 1.0)

    def test_simple_match(self):
        gt = [{"id": "w1", "px_coords": [0, 0, 100, 0]}]
        pred = [{"id": "p1", "px_coords": [2, 1, 98, 1]}]
        r = greedy_match("walls", gt, pred, wall_score, 0.55)
        assert r.true_positives == 1
        assert r.false_positives == 0 and r.false_negatives == 0

    def test_unmatched_counts(self):
        gt = [{"id": "w1", "px_coords": [0, 0, 100, 0]},
              {"id": "w2", "px_coords": [0, 50, 100, 50]}]
        pred = [{"id": "p1", "px_coords": [0, 0, 100, 0]},
                {"id": "p2", "px_coords": [0, 500, 100, 500]}]
        r = greedy_match("walls", gt, pred, wall_score, 0.55)
        assert r.true_positives == 1
        assert r.false_positives == 1
        assert r.false_negatives == 1
        assert r.missing_objects[0]["id"] == "w2"

    def test_build_report_aggregates(self):
        results = [
            greedy_match("walls", [], [], wall_score, 0.5),
        ]
        report = build_report("case", results)
        assert report["case_id"] == "case"
        assert "macro" in report["summary"] and "micro" in report["summary"]


class TestNormalize:
    def test_wall_endpoint_ordering(self):
        doc = normalize_document({"walls": [{"id": "w1", "px_coords": [100, 0, 0, 0]}]})
        assert doc["walls"][0]["px_coords"] == [0.0, 0.0, 100.0, 0.0]

    def test_room_bbox_from_polygon(self):
        doc = normalize_document({
            "rooms": [{"id": "R1", "polygon_px": [[0, 0], [10, 0], [10, 20], [0, 20]]}],
        })
        assert doc["rooms"][0]["bbox_px"] == [0.0, 0.0, 10.0, 20.0]

    def test_dimension_value_passthrough(self):
        doc = normalize_document({"dimensions": [{"id": "d1", "value_raw": 20.5}]})
        assert doc["dimensions"][0]["value_raw"] == 20.5

    def test_dimension_plain_float_text(self):
        doc = normalize_document({"dimensions": [{"id": "d1", "text": "20.5"}]})
        assert doc["dimensions"][0]["value_raw"] == pytest.approx(20.5)

    def test_arqen_output_shape_accepted(self):
        doc = normalize_document({
            "detected_scale": "1in=16ft",
            "walls": [{"id": "w1", "px_coords": [5, 5, 105, 5], "length_raw": 10.0,
                       "facing": "North", "is_exterior": True}],
            "rooms": [{"id": "R1", "bbox_px": [0, 0, 50, 50], "area_raw": 100.0}],
        })
        assert doc["scale"] == "1in=16ft"
        assert len(doc["walls"]) == 1 and len(doc["rooms"]) == 1


class TestClosure:
    def test_point_to_segment_distance(self):
        assert point_to_segment_distance(5, 5, [0, 0, 10, 0]) == pytest.approx(5.0)
        assert point_to_segment_distance(-3, 0, [0, 0, 10, 0]) == pytest.approx(3.0)
        assert point_to_segment_distance(1, 1, [2, 2, 2, 2]) == pytest.approx(2 ** 0.5)

    def test_closed_rectangle_fully_closed(self):
        net = wall_network_closure(RECT_WALLS, tol_px=5)
        assert net["closure_rate"] == 1.0
        assert net["dangling_endpoints"] == 0

    def test_floating_wall_dangles(self):
        walls = RECT_WALLS + [{"id": "f", "px_coords": [300, 300, 400, 300]}]
        net = wall_network_closure(walls, tol_px=5)
        assert net["dangling_endpoints"] == 2
        assert net["closure_rate"] == pytest.approx(8 / 10)
        assert {d["wall_id"] for d in net["dangling"]} == {"f"}

    def test_t_junction_endpoint_closed(self):
        walls = RECT_WALLS + [{"id": "t", "px_coords": [50, 0, 50, 60]}]
        net = wall_network_closure(walls, tol_px=5)
        # (50, 0) lies on the north wall body; (50, 60) dangles
        assert net["dangling_endpoints"] == 1

    def test_no_walls(self):
        net = wall_network_closure([], tol_px=5)
        assert net["closure_rate"] is None

    def test_room_boundary_closed(self):
        rb = room_boundary_closure(
            [{"id": "R1", "bbox_px": [0, 0, 100, 100]}], RECT_WALLS, tol_px=6,
        )
        assert rb["closure_rate"] == 1.0
        assert rb["per_room"][0]["closed"] is True

    def test_room_boundary_open_side(self):
        three_walls = RECT_WALLS[:3]  # west wall missing
        rb = room_boundary_closure(
            [{"id": "R1", "bbox_px": [0, 0, 100, 100]}], three_walls, tol_px=6,
        )
        assert rb["closure_rate"] == 0.0
        coverage = rb["per_room"][0]["boundary_coverage"]
        assert 0.6 < coverage < 0.9

    def test_room_boundary_mixed_rate(self):
        rooms = [
            {"id": "R1", "bbox_px": [0, 0, 100, 100]},
            {"id": "R2", "bbox_px": [500, 500, 600, 600]},
        ]
        rb = room_boundary_closure(rooms, RECT_WALLS, tol_px=6)
        assert rb["room_count"] == 2
        assert rb["closed_rooms"] == 1
        assert rb["closure_rate"] == 0.5

    def test_interior_coverage(self):
        pred = {
            "footprint_polygon_px": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "rooms": [{"id": "R1", "area_px": 5000}],
        }
        cov = interior_coverage(pred)
        assert cov["coverage"] == pytest.approx(0.5)

    def test_interior_coverage_none_without_footprint(self):
        assert interior_coverage({"rooms": []}) is None

    def test_derive_tolerance(self):
        assert derive_tolerance_px({"px_per_ft": 9.38}) == pytest.approx(18.76)
        assert derive_tolerance_px({}) == 12.0
        assert derive_tolerance_px(None) == 12.0

    def test_compute_closure_shape(self):
        gt = normalize_document({"rooms": [{"id": "R1", "bbox_px": [0, 0, 100, 100]}]})
        pred = normalize_document({"walls": RECT_WALLS})
        out = compute_closure(gt, pred, prediction_raw={"px_per_ft": 18.0})
        assert set(out) == {"tolerance_px", "wall_network", "room_boundary",
                            "interior_coverage"}
        assert out["tolerance_px"] == 36.0


class TestScorePrediction:
    def test_report_includes_closure(self):
        gt = {
            "rooms": [{"id": "R1", "bbox_px": [0, 0, 100, 100]}],
            "walls": [{"id": "w1", "px_coords": [0, 0, 100, 0]}],
            "image_size_px": [200, 200],
        }
        pred = {"walls": [{"id": "p1", "px_coords": [0, 2, 100, 2]}], "rooms": []}
        report = score_prediction(gt, pred)
        assert "closure" in report
        assert report["categories"]["walls"]["recall"] == 1.0
        assert report["categories"]["rooms"]["recall"] == 0.0


class TestGroundTruthSchema:
    @pytest.fixture(scope="class")
    def schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        path = VALIDATION_DIR / "schema" / "ground_truth.schema.json"
        return jsonschema, json.loads(path.read_text(encoding="utf-8"))

    def test_demo_case_validates(self, schema):
        jsonschema, schema_doc = schema
        gt = json.loads(
            (VALIDATION_DIR / "cases" / "demo_minimal" / "ground_truth.json")
            .read_text(encoding="utf-8")
        )
        jsonschema.validate(gt, schema_doc)

    def test_synthetic_plans_validate(self, schema, two_room_plan, l_shape_plan):
        jsonschema, schema_doc = schema
        jsonschema.validate(two_room_plan.ground_truth, schema_doc)
        jsonschema.validate(l_shape_plan.ground_truth, schema_doc)
