"""ROI hint expansion and segment-overlap filtering."""

from pathlib import Path

import cv2
import pytest

from preprocess import (
    _expand_roi_from_hint,
    _filter_wall_segments,
    _segment_overlap_frac,
)
from scale_parse import parse_scale

REPO_ROOT = Path(__file__).resolve().parents[2]
TRDI_IMAGE = REPO_ROOT / "validation" / "cases" / "trdi_overall" / "image.png"


class TestSegmentOverlapFrac:
    def test_horizontal_majority_inside_kept_by_filter(self):
        roi = {"x0_pct": 0.2, "y0_pct": 0.0, "x1_pct": 0.8, "y1_pct": 1.0}
        img_w, img_h = 1000, 1000
        # Midpoint at 0.65; 400–800 px inside ROI → 80% of 500 px segment.
        seg = (400, 500, 900, 500)
        frac = _segment_overlap_frac(*seg, roi, img_w, img_h)
        assert frac == pytest.approx(0.8, abs=0.01)

        kept = _filter_wall_segments(
            [seg], img_w, img_h, roi=roi, max_span_frac=1.0,
        )
        assert kept == [seg]

    def test_horizontal_mostly_outside_dropped(self):
        roi = {"x0_pct": 0.0, "y0_pct": 0.0, "x1_pct": 0.5, "y1_pct": 1.0}
        img_w, img_h = 1000, 1000
        seg = (400, 500, 900, 500)
        frac = _segment_overlap_frac(*seg, roi, img_w, img_h)
        assert frac == pytest.approx(0.2, abs=0.01)

        kept = _filter_wall_segments([seg], img_w, img_h, roi=roi)
        assert kept == []


@pytest.mark.skipif(not TRDI_IMAGE.exists(), reason="TRDI case image missing")
class TestExpandRoiFromHint:
    @pytest.fixture(scope="class")
    def trdi_rgb(self):
        bgr = cv2.imread(str(TRDI_IMAGE))
        assert bgr is not None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @pytest.fixture(scope="class")
    def px_per_unit(self):
        cal = parse_scale('1/8"=1ft', 71, output_unit="ft")
        return cal["px_per_unit"]

    def test_office_only_hint_expands_to_full_building_height(self, trdi_rgb, px_per_unit):
        office_hint = {
            "x0_pct": 0.30,
            "y0_pct": 0.02,
            "x1_pct": 0.85,
            "y1_pct": 0.50,
            "method": "user-roi",
        }
        expanded = _expand_roi_from_hint(trdi_rgb, office_hint, px_per_unit)
        hint_height = office_hint["y1_pct"] - office_hint["y0_pct"]
        height_frac = expanded["y1_pct"] - expanded["y0_pct"]
        assert height_frac > hint_height + 0.15, (
            f"office hint should expand below the office block: "
            f"hint={hint_height:.0%} expanded={height_frac:.0%}"
        )
        assert expanded["y1_pct"] > office_hint["y1_pct"] + 0.15
