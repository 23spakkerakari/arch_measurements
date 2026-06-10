"""Stage [2] Scale calibration — scale_parse.py (pure string parsing)."""

import pytest

from scale_parse import (
    _parse_arch_length_to_inches,
    _parse_fraction_or_float,
    _parse_real_length_to_feet,
    parse_scale,
)

pytestmark = pytest.mark.unit


class TestHelpers:
    def test_fraction(self):
        assert _parse_fraction_or_float("1/16") == pytest.approx(0.0625)

    def test_decimal(self):
        assert _parse_fraction_or_float("0.25") == pytest.approx(0.25)

    @pytest.mark.parametrize("s,expected", [
        ('1/16"', 0.0625),
        ("1in", 1.0),
        ("0.25in", 0.25),
        ("10mm", 10 / 25.4),
        ("2.5cm", 2.5 / 2.54),
    ])
    def test_arch_lengths(self, s, expected):
        assert _parse_arch_length_to_inches(s) == pytest.approx(expected)

    def test_arch_length_unsupported(self):
        with pytest.raises(ValueError):
            _parse_arch_length_to_inches("1banana")

    @pytest.mark.parametrize("s,expected", [
        ("1'-0\"", 1.0),
        ("12'-6\"", 12.5),
        ("32'", 32.0),
        ("1ft", 1.0),
        ("12in", 1.0),
        ('6"', 0.5),
        ("2500mm", 2.5 * 3.28084),
    ])
    def test_real_lengths(self, s, expected):
        assert _parse_real_length_to_feet(s) == pytest.approx(expected)

    def test_real_length_meters_unsupported(self):
        # Meters on the real-world side are intentionally not parsed yet
        # (scale_parse.py line ~89 commented out).
        with pytest.raises(ValueError):
            _parse_real_length_to_feet("3m")


class TestParseScaleEquality:
    def test_quarter_inch(self):
        cal = parse_scale("1/4in=1ft", 300)
        assert cal["px_per_unit"] == pytest.approx(75.0)
        assert cal["unit_label"] == "ft"

    def test_eighth_inch_at_144(self):
        cal = parse_scale('1/8"=1ft', 144)
        assert cal["px_per_unit"] == pytest.approx(18.0)

    def test_one_inch_sixteen_feet(self):
        cal = parse_scale("1in=16ft", 150)
        assert cal["px_per_unit"] == pytest.approx(9.375)

    def test_arch_feet_inches_form(self):
        cal = parse_scale('1/16" = 1\'-0"', 300)
        assert cal["px_per_unit"] == pytest.approx(18.75)

    def test_three_eighths(self):
        cal = parse_scale('3/8"=1ft', 144)
        assert cal["px_per_unit"] == pytest.approx(54.0)

    def test_metric_output(self):
        cal = parse_scale("1/4in=1ft", 300, output_unit="m")
        assert cal["unit_label"] == "m"
        assert cal["px_per_unit"] == pytest.approx(75.0 * 3.28084)

    def test_dpi_scaling_is_linear(self):
        a = parse_scale("1/8in=1ft", 150)["px_per_unit"]
        b = parse_scale("1/8in=1ft", 300)["px_per_unit"]
        assert b == pytest.approx(2 * a)


class TestParseScaleRatio:
    def test_1_to_100_imperial(self):
        cal = parse_scale("1:100", 300)
        # 1 drawing inch = 100 real inches = 100/12 ft -> 300 / (100/12) = 36
        assert cal["px_per_unit"] == pytest.approx(36.0)
        assert cal["unit_label"] == "ft"

    def test_1_to_50_metric(self):
        cal = parse_scale("1:50", 300, output_unit="m")
        assert cal["px_per_unit"] == pytest.approx(300 / (50 * 0.0254))
        assert cal["unit_label"] == "m"


class TestParseScaleErrors:
    def test_garbage(self):
        with pytest.raises(ValueError):
            parse_scale("banana", 300)

    def test_unsupported_output_unit(self):
        with pytest.raises(ValueError):
            parse_scale("1/4in=1ft", 300, output_unit="yd")

    def test_meters_real_side(self):
        with pytest.raises(ValueError):
            parse_scale("1cm=1m", 300)
