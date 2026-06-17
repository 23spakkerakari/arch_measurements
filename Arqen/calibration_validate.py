"""
Calibration sanity checks for scale/DPI inputs.

Derived from baseline cases (2026-06-10): px_per_ft 9.38–75, areas 2300–8700 ft².
Warns on implausible calibration before users trust measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scale_parse import parse_scale

# Typical architectural rasterization DPI range.
DPI_MIN_WARN = 72
DPI_MAX_WARN = 600

# px_per_ft observed on baseline cases: 9.38, 18, 54, 75 — allow margin.
PX_PER_FT_MIN_WARN = 3.0
PX_PER_FT_MAX_WARN = 250.0

# Below this px/ft, footprint/wall tagging and window detection are unreliable on
# real 1/8" overall sheets (measured on TRDI overall @ 144dpi, ppf=18).
PX_PER_FT_RELIABILITY_MIN = 30.0

# Footprint span in feet (proposal: 10–500; upper soft limit 800 for commercial).
SPAN_FT_MIN_WARN = 10.0
SPAN_FT_MAX_WARN = 800.0

# Plausible building floor area (baseline: 2300–8700 ft²).
AREA_FT2_MIN_WARN = 100.0
AREA_FT2_MAX_WARN = 100_000.0

# When probing DPI alternatives, span in this band suggests a better match.
DPI_ALT_SPAN_MIN_FT = 15.0
DPI_ALT_SPAN_MAX_FT = 400.0

COMMON_DPIS = (72, 96, 144, 150, 200, 240, 300)

# When multiple DPI values yield a plausible span, prefer these (web/CLI defaults).
PREFERRED_DPIS = (144, 150, 240, 300, 200, 96, 72)


@dataclass(frozen=True)
class CalibrationIssue:
    code: str
    severity: str  # "warning" | "error"
    message: str
    details: dict


def validate_dpi(dpi: int) -> list[CalibrationIssue]:
    issues: list[CalibrationIssue] = []
    if dpi < 1:
        issues.append(CalibrationIssue(
            code="dpi_invalid",
            severity="error",
            message=f"DPI must be positive (got {dpi})",
            details={"dpi": dpi},
        ))
    elif dpi < DPI_MIN_WARN or dpi > DPI_MAX_WARN:
        issues.append(CalibrationIssue(
            code="dpi_out_of_range",
            severity="warning",
            message=(
                f"DPI {dpi} is outside typical range "
                f"({DPI_MIN_WARN}–{DPI_MAX_WARN}); verify rasterization DPI"
            ),
            details={"dpi": dpi, "min": DPI_MIN_WARN, "max": DPI_MAX_WARN},
        ))
    return issues


def suggest_min_dpi_for_reliability(scale_str: str, min_px_per_ft: float = PX_PER_FT_RELIABILITY_MIN) -> int | None:
    """Smallest common DPI where ``parse_scale`` yields at least ``min_px_per_ft``."""
    for dpi in PREFERRED_DPIS + tuple(d for d in COMMON_DPIS if d not in PREFERRED_DPIS):
        try:
            cal = parse_scale(scale_str, dpi, output_unit="ft")
        except ValueError:
            continue
        if cal["px_per_unit"] >= min_px_per_ft:
            return dpi
    return None


def validate_resolution_reliability(
    px_per_unit: float,
    scale_str: str,
    dpi: int,
    unit_label: str = "ft",
) -> list[CalibrationIssue]:
    """Warn when px/ft is too low for reliable geometry extraction."""
    if unit_label != "ft":
        return []
    if px_per_unit >= PX_PER_FT_RELIABILITY_MIN:
        return []
    suggested_dpi = suggest_min_dpi_for_reliability(scale_str)
    msg = (
        f"px_per_ft {px_per_unit:.1f} is below the reliability floor "
        f"({PX_PER_FT_RELIABILITY_MIN:.0f}); wall and window detection may be wrong. "
        "Re-render at higher DPI or use the enlarged-scale plan sheet."
    )
    details: dict = {
        "px_per_ft": round(px_per_unit, 4),
        "min_reliable_px_per_ft": PX_PER_FT_RELIABILITY_MIN,
        "dpi": dpi,
    }
    if suggested_dpi is not None:
        details["suggested_min_dpi"] = suggested_dpi
    return [CalibrationIssue(
        code="low_resolution",
        severity="warning",
        message=msg,
        details=details,
    )]


def validate_px_per_unit(px_per_unit: float, unit_label: str = "ft") -> list[CalibrationIssue]:
    if unit_label != "ft":
        return []
    issues: list[CalibrationIssue] = []
    if px_per_unit < PX_PER_FT_MIN_WARN or px_per_unit > PX_PER_FT_MAX_WARN:
        issues.append(CalibrationIssue(
            code="px_per_unit_out_of_range",
            severity="warning",
            message=(
                f"px_per_ft {px_per_unit:.2f} is outside typical range "
                f"({PX_PER_FT_MIN_WARN}–{PX_PER_FT_MAX_WARN}); check scale and DPI"
            ),
            details={
                "px_per_ft": round(px_per_unit, 4),
                "min": PX_PER_FT_MIN_WARN,
                "max": PX_PER_FT_MAX_WARN,
            },
        ))
    return issues


def _bbox_span_ft(bbox_px: list, px_per_unit: float) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox_px
    width_ft = abs(x1 - x0) / px_per_unit
    height_ft = abs(y1 - y0) / px_per_unit
    return width_ft, height_ft


def _span_in_plausible_band(width_ft: float, height_ft: float) -> bool:
    span_min = min(width_ft, height_ft)
    span_max = max(width_ft, height_ft)
    return (
        span_min >= DPI_ALT_SPAN_MIN_FT
        and span_max <= DPI_ALT_SPAN_MAX_FT
        and span_max >= SPAN_FT_MIN_WARN
    )


def validate_footprint_span(
    bbox_px: list,
    px_per_unit: float,
    unit_label: str = "ft",
) -> list[CalibrationIssue]:
    if unit_label != "ft":
        return []
    width_ft, height_ft = _bbox_span_ft(bbox_px, px_per_unit)
    span_min = min(width_ft, height_ft)
    span_max = max(width_ft, height_ft)
    issues: list[CalibrationIssue] = []

    if span_max < SPAN_FT_MIN_WARN:
        issues.append(CalibrationIssue(
            code="footprint_span_low",
            severity="warning",
            message=(
                f"Footprint span {span_max:.1f} ft is below {SPAN_FT_MIN_WARN} ft; "
                "scale or DPI may be wrong"
            ),
            details={
                "width_ft": round(width_ft, 2),
                "height_ft": round(height_ft, 2),
                "span_max_ft": round(span_max, 2),
                "min_ft": SPAN_FT_MIN_WARN,
            },
        ))
    elif span_max > SPAN_FT_MAX_WARN:
        issues.append(CalibrationIssue(
            code="footprint_span_high",
            severity="warning",
            message=(
                f"Footprint span {span_max:.1f} ft exceeds {SPAN_FT_MAX_WARN} ft; "
                "scale or DPI may be wrong"
            ),
            details={
                "width_ft": round(width_ft, 2),
                "height_ft": round(height_ft, 2),
                "span_max_ft": round(span_max, 2),
                "max_ft": SPAN_FT_MAX_WARN,
            },
        ))
    return issues


def validate_total_area(
    area_raw: float,
    unit_label: str = "ft",
) -> list[CalibrationIssue]:
    if unit_label != "ft":
        return []
    issues: list[CalibrationIssue] = []
    if area_raw < AREA_FT2_MIN_WARN:
        issues.append(CalibrationIssue(
            code="total_area_low",
            severity="warning",
            message=(
                f"Total area {area_raw:.1f} ft² is below {AREA_FT2_MIN_WARN} ft²; "
                "scale or DPI may be wrong"
            ),
            details={"area_ft2": round(area_raw, 2), "min_ft2": AREA_FT2_MIN_WARN},
        ))
    elif area_raw > AREA_FT2_MAX_WARN:
        issues.append(CalibrationIssue(
            code="total_area_high",
            severity="warning",
            message=(
                f"Total area {area_raw:.1f} ft² exceeds {AREA_FT2_MAX_WARN} ft²; "
                "scale or DPI may be wrong"
            ),
            details={"area_ft2": round(area_raw, 2), "max_ft2": AREA_FT2_MAX_WARN},
        ))
    return issues


def check_dpi_alternatives(
    scale_str: str,
    dpi: int,
    footprint_span_ft: tuple[float, float],
) -> list[CalibrationIssue]:
    """Suggest a different DPI if footprint span fits better at another common value."""
    width_ft, height_ft = footprint_span_ft
    current_ok = _span_in_plausible_band(width_ft, height_ft)
    if current_ok:
        return []

    try:
        cal = parse_scale(scale_str, dpi, output_unit="ft")
        px_per_unit = cal["px_per_unit"]
    except ValueError:
        return []

    # Reconstruct pixel span from current feet span and px_per_unit.
    width_px = width_ft * px_per_unit
    height_px = height_ft * px_per_unit

    candidates: list[int] = []
    for alt_dpi in COMMON_DPIS:
        if alt_dpi == dpi:
            continue
        try:
            alt_cal = parse_scale(scale_str, alt_dpi, output_unit="ft")
        except ValueError:
            continue
        alt_ppu = alt_cal["px_per_unit"]
        alt_w = width_px / alt_ppu
        alt_h = height_px / alt_ppu
        if _span_in_plausible_band(alt_w, alt_h):
            candidates.append(alt_dpi)

    if not candidates:
        return []

    for preferred in PREFERRED_DPIS:
        if preferred in candidates:
            best_dpi = preferred
            break
    else:
        best_dpi = min(candidates, key=lambda d: abs(d - dpi))

    return [CalibrationIssue(
        code="dpi_mismatch_suspected",
        severity="warning",
        message=(
            f"Footprint span looks implausible at DPI {dpi}; "
            f"DPI {best_dpi} would yield a more typical building size"
        ),
        details={
            "dpi": dpi,
            "suggested_dpi": best_dpi,
            "footprint_span_ft": [round(width_ft, 2), round(height_ft, 2)],
        },
    )]


def summarize_calibration(
    issues: list[CalibrationIssue],
    *,
    dpi: int,
    px_per_unit: float,
    footprint_span_ft: tuple[float, float],
    total_area_raw: float,
    unit_label: str = "ft",
) -> dict:
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if errors:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    suggested_dpi = None
    for issue in issues:
        if issue.code == "dpi_mismatch_suspected":
            suggested_dpi = issue.details.get("suggested_dpi")
            break
        if issue.code == "low_resolution" and suggested_dpi is None:
            suggested_dpi = issue.details.get("suggested_min_dpi")

    px_key = "px_per_ft" if unit_label == "ft" else "px_per_unit"

    return {
        "status": status,
        "dpi": dpi,
        px_key: round(px_per_unit, 2),
        "footprint_span_ft": [round(footprint_span_ft[0], 2), round(footprint_span_ft[1], 2)],
        "total_area_raw": round(total_area_raw, 2),
        "issues": [asdict(i) for i in issues],
        "suggested_dpi": suggested_dpi,
    }


def issue_to_log_line(issue: CalibrationIssue) -> str:
    return f"[calibration] {issue.severity}: {issue.message}"
