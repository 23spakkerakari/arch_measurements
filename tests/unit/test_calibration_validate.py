"""Unit tests for calibration_validate.py (scale/DPI sanity guards)."""

import pytest

from calibration_validate import (
    CalibrationIssue,
    check_dpi_alternatives,
    summarize_calibration,
    suggest_min_dpi_for_reliability,
    validate_dpi,
    validate_footprint_span,
    validate_px_per_unit,
    validate_resolution_reliability,
    validate_total_area,
)

pytestmark = pytest.mark.unit


class TestValidateDpi:
    def test_typical_ok(self):
        assert validate_dpi(144) == []
        assert validate_dpi(300) == []

    def test_low_warning(self):
        issues = validate_dpi(50)
        assert len(issues) == 1
        assert issues[0].code == "dpi_out_of_range"
        assert issues[0].severity == "warning"

    def test_high_warning(self):
        issues = validate_dpi(700)
        assert issues[0].code == "dpi_out_of_range"

    def test_zero_error(self):
        issues = validate_dpi(0)
        assert issues[0].code == "dpi_invalid"
        assert issues[0].severity == "error"


class TestValidatePxPerUnit:
    @pytest.mark.parametrize("ppu", [9.38, 18.0, 54.0, 75.0])
    def test_baseline_values_ok(self, ppu):
        assert validate_px_per_unit(ppu) == []

    def test_too_low(self):
        issues = validate_px_per_unit(0.5)
        assert issues[0].code == "px_per_unit_out_of_range"

    def test_too_high(self):
        issues = validate_px_per_unit(500.0)
        assert issues[0].code == "px_per_unit_out_of_range"

    def test_metric_skipped(self):
        assert validate_px_per_unit(100.0, unit_label="m") == []


class TestValidateResolutionReliability:
    def test_18_ppf_warns(self):
        issues = validate_resolution_reliability(18.0, "1/8in=1ft", 144)
        assert len(issues) == 1
        assert issues[0].code == "low_resolution"
        assert issues[0].severity == "warning"
        assert issues[0].details["suggested_min_dpi"] == 240

    def test_54_ppf_ok(self):
        assert validate_resolution_reliability(54.0, "3/8in=1ft", 144) == []

    def test_suggest_min_dpi(self):
        assert suggest_min_dpi_for_reliability("1/8in=1ft") == 240


class TestValidateFootprintSpan:
    def test_typical_building_ok(self):
        # 60 ft × 40 ft at 18 px/ft → 1080 × 720 px bbox
        bbox = [0, 0, 1080, 720]
        assert validate_footprint_span(bbox, 18.0) == []

    def test_span_too_small(self):
        bbox = [0, 0, 36, 24]  # 2 × 1.33 ft at 18 px/ft
        issues = validate_footprint_span(bbox, 18.0)
        assert issues[0].code == "footprint_span_low"

    def test_span_too_large(self):
        bbox = [0, 0, 18000, 12000]  # 1000 × 666 ft at 18 px/ft
        issues = validate_footprint_span(bbox, 18.0)
        assert issues[0].code == "footprint_span_high"


class TestValidateTotalArea:
    def test_baseline_ok(self):
        assert validate_total_area(2797.0) == []
        assert validate_total_area(8702.1) == []

    def test_too_small(self):
        issues = validate_total_area(50.0)
        assert issues[0].code == "total_area_low"

    def test_too_large(self):
        issues = validate_total_area(150_000.0)
        assert issues[0].code == "total_area_high"


class TestCheckDpiAlternatives:
    def test_current_span_ok_no_suggestion(self):
        # 60 × 40 ft is plausible
        issues = check_dpi_alternatives("1/8in=1ft", 144, (60.0, 40.0))
        assert issues == []

    def test_wrong_dpi_suggests_alternative(self):
        # At dpi=600 vs correct 144, same pixel footprint reads ~14×10 ft (too narrow
        # for alt band) but ~60×40 ft at 144.
        issues = check_dpi_alternatives("1/8in=1ft", 600, (14.4, 9.6))
        assert len(issues) == 1
        assert issues[0].code == "dpi_mismatch_suspected"
        assert issues[0].details["suggested_dpi"] == 144

    def test_implausible_at_all_dpis(self):
        issues = check_dpi_alternatives("1/8in=1ft", 144, (2.0, 1.0))
        assert issues == [] or issues[0].code == "dpi_mismatch_suspected"


class TestSummarizeCalibration:
    def test_ok_status(self):
        out = summarize_calibration(
            [],
            dpi=144,
            px_per_unit=18.0,
            footprint_span_ft=(60.0, 40.0),
            total_area_raw=2400.0,
        )
        assert out["status"] == "ok"
        assert out["issues"] == []
        assert out["suggested_dpi"] is None

    def test_warning_status(self):
        issue = CalibrationIssue(
            code="footprint_span_low",
            severity="warning",
            message="test",
            details={},
        )
        out = summarize_calibration(
            [issue],
            dpi=144,
            px_per_unit=18.0,
            footprint_span_ft=(5.0, 4.0),
            total_area_raw=20.0,
        )
        assert out["status"] == "warning"
        assert len(out["issues"]) == 1

    def test_error_status(self):
        issue = CalibrationIssue(
            code="dpi_invalid",
            severity="error",
            message="test",
            details={},
        )
        out = summarize_calibration(
            [issue],
            dpi=0,
            px_per_unit=18.0,
            footprint_span_ft=(60.0, 40.0),
            total_area_raw=2400.0,
        )
        assert out["status"] == "error"

    def test_suggested_dpi_extracted(self):
        issue = CalibrationIssue(
            code="dpi_mismatch_suspected",
            severity="warning",
            message="test",
            details={"suggested_dpi": 144},
        )
        out = summarize_calibration(
            [issue],
            dpi=300,
            px_per_unit=37.5,
            footprint_span_ft=(30.0, 20.0),
            total_area_raw=600.0,
        )
        assert out["suggested_dpi"] == 144

    def test_low_resolution_suggested_dpi(self):
        issue = CalibrationIssue(
            code="low_resolution",
            severity="warning",
            message="test",
            details={"suggested_min_dpi": 240},
        )
        out = summarize_calibration(
            [issue],
            dpi=144,
            px_per_unit=18.0,
            footprint_span_ft=(60.0, 40.0),
            total_area_raw=2400.0,
        )
        assert out["suggested_dpi"] == 240
