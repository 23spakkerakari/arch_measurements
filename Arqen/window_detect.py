"""Geometric window detection on exterior walls via sill-line signatures.

Two complementary strategies:

1. **In-segment openings** — scan each exterior wall span for bands where
   ``wall_pair_mask`` ink drops out and a parallel sill stroke is present in
   the raw ink. This is the common case: the wall extractor emits one long
   segment per room-side sub-wall, with window openings *inside* the span.

2. **Collinear segment gaps** — when the extractor happens to split an
   exterior wall at a window (same geometry as door detection, sill required).

All coordinates are in the same (cropped) frame as the input walls;
``analyze_page`` shifts results by the ROI offset alongside the walls.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from door_detect import (
    OPEN_GAP_MAX_INK_FRAC,
    SILL_COVER_FRAC,
    _crop,
    _gap_has_bilateral_break,
    _gap_is_open,
    _gap_rect,
    _looks_like_dimension_line,
    _merged_interval_gaps,
    _orient,
    gap_sill_evidence,
    ink_mask_from_image,
)

# Standard window widths; synth plans use 4 ft openings.
WINDOW_MIN_FT = 2.0
WINDOW_MAX_FT = 8.0
WINDOW_MIN_FT_CROP = 2.0
WINDOW_MAX_FT_CROP = 8.0
OPEN_GAP_MAX_CROP = 0.25           # crop_mode open-band threshold (unchanged; 0.18 regressed fp_20 recall)
HORIZ_MAX_FT_CROP = 8.75           # reject oversized horizontal detections in crop_mode
SPAN_REFINE_MIN_FRAC = 0.80
JAMB_INK_PEAK_FRAC = 0.30
JAMB_INK_PEAK_FRAC_CROP = 0.42       # fewer jamb splits on low-px plans (TRDI overall)
SPLIT_MIN_SEGMENT_FT_CROP = 3.25     # discard tiny split fragments in crop_mode
TOUCH_MERGE_GAP_FT = 0.35            # merge touching over-split fragments
OVER_SPLIT_WIDTH_FT = 3.75           # narrow fragment threshold for touch merge
MERGE_COMBINED_MAX_FT = 10.0         # touch-merge span cap
MULLION_GAP_MAX_FT_CROP = 4.5        # max gap bridged across a structural mullion (crop_mode)
MULLION_COMBINED_MAX_FT_CROP = 17.0  # tall curtain-wall span cap after mullion merge
MULLION_FRAGMENT_MAX_FT_CROP = 7.5   # both fragments must be no wider than this
MIN_SCAN_WALL_FT_CROP = 5.0          # skip short interior coaxial walls in crop_mode
HATCH_PEAK_FRAC = 0.20               # feature-ink peak threshold along opening span
HATCH_LOW_PX_PER_FT = 22.0           # apply east-facade corroboration below this calibration

# Multi-cue acceptance scoring (Window Accuracy V2, Phase 1).
# A candidate is accepted when its confidence >= CONF_TAU. The score combines
# independent cues so a strong sill can compensate for a weak opening signal
# and vice-versa, while requiring corroboration rejects lone strokes (dimension
# strings, annotations) that the binary sill gate used to admit.
SILL_RAMP_LO = 0.25          # sill cover below this contributes nothing
SILL_RAMP_HI = SILL_COVER_FRAC  # 0.60: full-strength sill
W_SILL = 0.50                # weight of the sill-cover cue
W_OPEN = 0.30                # wall-pair band reads as an open gap
W_BILATERAL = 0.22           # both wall strokes break across the span
W_TRIPLE_BONUS = 0.08        # CAD triple-line window symbol
W_DIM_PENALTY = 0.45         # looks like a dimension string (no bilateral break)

# Envelope / wall-flank gating (Window Accuracy V2, Phase 2).
# A real window is an opening *in a wall*: the double-stroke wall ink must
# continue on both sides of the gap. Phantom exterior sub-segments from an
# overshooting footprint polygon run across whitespace/borders with no flanking
# wall, so this gate removes their false positives without touching real
# windows (whose host wall continues past the jambs).
FLANK_COVER_FRAC = 0.50      # min along-axis wall-ink coverage in a flank probe
FLANK_PROBE_FT = 0.75        # probe length on each side of the gap (feet)

# Recall unlock (Window Accuracy V2, Phase 3) + precision guard (Phase 2 fix).
# Scan only walls that lie on the building envelope (outermost axis on each side),
# regardless of the is_exterior tag. Perimeter walls mis-tagged interior are
# recovered; interior walls mis-tagged exterior are suppressed.
# The flank gate + sill requirement keep precision on envelope scans.
ENVELOPE_TOL_FT = 2.0        # axis distance to the building envelope edge (feet)
MIN_ENVELOPE_AXIS_SPAN_PX = 12  # ignore point/degenerate segments when computing envelope

# Symbol-on-wall detection (Window Symbol Recall V3).
# Some plans draw windows as periodic glyph markers on a continuous centerline
# glazing line *without* breaking the wall, so the opening-based strategies
# never fire. The signature is feature ink -- present in ink_mask, absent from
# wall_pair_mask (so wall strokes and crossing walls are excluded) -- sustained
# in a thin band at the wall axis while the wall pair stays continuous. Plain
# walls have an empty centre channel (measured ~0 on FP-only cases), so a
# run-level coverage gate keeps precision.
GLAZE_BAND_FT = 0.25         # centre-channel half-width (feet)
SYMBOL_MARKER_FRAC = 0.28    # per-column centre-channel density marking a glyph
SYMBOL_MARKER_MIN_FT = 0.15  # min along-wall extent of a marker blob (feet)
SYMBOL_MARKER_MAX_FT = 1.50  # max along-wall extent of a marker blob (feet)
SYMBOL_MARKER_GAP_FT = 0.40  # bridge tiny breaks within one marker blob (feet)
SYMBOL_CONFIDENCE = 0.6      # fixed acceptance confidence for symbol candidates
# Windows of this convention are drawn as periodic glyph markers on the wall
# centreline. A lone on-axis blob is ambiguous (text, fixture, junction); a
# regularly-spaced *series* is an unmistakable window run. Requiring several
# markers with consistent spacing is the core precision guard -- plain walls and
# FP-only plans have no such periodic on-axis series.
SYMBOL_MIN_MARKERS = 3       # min markers on a wall to accept the series
SYMBOL_SPACING_CV_MAX = 0.55  # max coeff. of variation of marker spacings

__all__ = [
    "detect_windows",
    "detect_window_candidates",
    "ink_mask_from_image",
]


def _along_wall_profile(
    mask: np.ndarray,
    horiz: bool,
    axis: float,
    span_lo: float,
    span_hi: float,
    band_half: int,
) -> tuple[np.ndarray, int]:
    """Mean ink fraction along the wall axis for a band centered on ``axis``."""
    lo, hi = int(round(span_lo)), int(round(span_hi))
    if horiz:
        y0, y1 = int(round(axis - band_half)), int(round(axis + band_half))
        region = _crop(mask, (lo, y0, hi, y1))
        if region.size == 0:
            return np.array([]), lo
        return (region > 0).mean(axis=0), lo
    x0, x1 = int(round(axis - band_half)), int(round(axis + band_half))
    region = _crop(mask, (x0, lo, x1, hi))
    if region.size == 0:
        return np.array([]), lo
    return (region > 0).mean(axis=1), lo


def _split_at_interior_peaks(
    profile: np.ndarray,
    offset: int,
    run_lo: float,
    run_hi: float,
    min_gap_px: float,
    max_gap_px: float,
    peak_frac: float,
) -> list[tuple[float, float]]:
    """Split a span at interior ink peaks (jambs between adjacent openings)."""
    span_px = run_hi - run_lo
    if profile.size == 0 or span_px < min_gap_px:
        return []

    peak_idx = np.where(profile >= peak_frac)[0]
    if peak_idx.size == 0:
        if min_gap_px <= span_px:
            return [(run_lo, run_hi)]
        return []

    # Cluster peak indices; split at cluster midpoints.
    splits: list[int] = []
    cluster_start = int(peak_idx[0])
    prev = int(peak_idx[0])
    for idx in peak_idx[1:]:
        idx = int(idx)
        if idx - prev > max(2, int(0.15 * profile.size)):
            splits.append((cluster_start + prev) // 2)
            cluster_start = idx
        prev = idx

    boundaries = [0] + splits + [profile.size]
    runs: list[tuple[float, float]] = []
    for i in range(len(boundaries) - 1):
        sub_lo = offset + boundaries[i]
        sub_hi = offset + boundaries[i + 1]
        gap_px = sub_hi - sub_lo
        if min_gap_px <= gap_px <= max_gap_px:
            runs.append((float(sub_lo), float(sub_hi)))
    return runs


def _split_run_on_wall_ink_peaks(
    wall_pair_mask: np.ndarray,
    horiz: bool,
    axis: float,
    run_lo: float,
    run_hi: float,
    band_half: int,
    min_gap_px: float,
    max_gap_px: float,
    peak_frac: float = JAMB_INK_PEAK_FRAC,
) -> list[tuple[float, float]]:
    """Split a long open run at jamb ink in the wall-pair band."""
    profile, offset = _along_wall_profile(
        wall_pair_mask, horiz, axis, run_lo, run_hi, band_half,
    )
    span_px = run_hi - run_lo
    if span_px <= max_gap_px:
        if min_gap_px <= span_px:
            return [(run_lo, run_hi)]
        return []
    return _split_at_interior_peaks(
        profile, offset, run_lo, run_hi, min_gap_px, max_gap_px, peak_frac,
    )


def _split_run_on_sill_peaks(
    ink_mask: np.ndarray,
    horiz: bool,
    axis: float,
    run_lo: float,
    run_hi: float,
    band_half: int,
    min_gap_px: float,
    max_gap_px: float,
) -> list[tuple[float, float]]:
    """Split using sill-line ink peaks when wall-pair mask is one long open span."""
    rect = _gap_rect(horiz, axis, run_lo, run_hi, band_half)
    x0, y0, x1, y1 = rect
    region = _crop(ink_mask, (x0, y0, x1, y1))
    if region.size == 0:
        return [(run_lo, run_hi)] if run_hi - run_lo >= min_gap_px else []

    if horiz:
        along = (region > 0).mean(axis=0)
        offset = x0
    else:
        along = (region > 0).mean(axis=1)
        offset = y0

    sill_peak_frac = max(0.35, SILL_COVER_FRAC - 0.15)
    return _split_at_interior_peaks(
        along, offset, run_lo, run_hi, min_gap_px, max_gap_px, sill_peak_frac,
    )


def _expand_runs(
    wall_pair_mask: np.ndarray,
    ink_mask: np.ndarray,
    horiz: bool,
    axis: float,
    runs: list[tuple[float, float]],
    band_half: int,
    min_gap_px: float,
    max_gap_px: float,
    *,
    jamb_peak_frac: float = JAMB_INK_PEAK_FRAC,
    split_min_px: Optional[float] = None,
) -> list[tuple[float, float]]:
    """Split wide or multi-opening runs into per-window spans."""
    out: list[tuple[float, float]] = []
    for run_lo, run_hi in runs:
        wall_splits = _split_run_on_wall_ink_peaks(
            wall_pair_mask, horiz, axis, run_lo, run_hi,
            band_half, min_gap_px, max_gap_px, peak_frac=jamb_peak_frac,
        )
        if len(wall_splits) > 1:
            out.extend(wall_splits)
            continue
        if run_hi - run_lo > max_gap_px or (
            len(wall_splits) == 1 and wall_splits[0][1] - wall_splits[0][0] > max_gap_px
        ):
            sill_splits = _split_run_on_sill_peaks(
                ink_mask, horiz, axis, run_lo, run_hi,
                band_half, min_gap_px, max_gap_px,
            )
            if len(sill_splits) > 1:
                out.extend(sill_splits)
            elif wall_splits:
                out.extend(wall_splits)
            elif run_hi - run_lo >= min_gap_px:
                out.append((run_lo, run_hi))
            continue
        if wall_splits:
            out.extend(wall_splits)
        elif min_gap_px <= run_hi - run_lo <= max_gap_px:
            out.append((run_lo, run_hi))
    if split_min_px is not None and len(out) > 1:
        large = [(lo, hi) for lo, hi in out if hi - lo >= split_min_px]
        out = large if large else [max(out, key=lambda t: t[1] - t[0])]
    return out


def _open_runs_along_wall(
    wall_pair_mask: np.ndarray,
    horiz: bool,
    axis: float,
    span_lo: float,
    span_hi: float,
    band_half: int,
    min_gap_px: float,
    max_gap_px: float,
    max_ink_frac: float,
    end_inset_px: int,
) -> list[tuple[float, float]]:
    """Contiguous along-wall spans where the wall-pair band is essentially open."""
    lo = int(round(span_lo)) + end_inset_px
    hi = int(round(span_hi)) - end_inset_px
    if hi <= lo:
        return []

    if horiz:
        y0, y1 = int(round(axis - band_half)), int(round(axis + band_half))
        region = _crop(wall_pair_mask, (lo, y0, hi, y1))
        if region.size == 0:
            return []
        ink_frac = (region > 0).mean(axis=0)
    else:
        x0, x1 = int(round(axis - band_half)), int(round(axis + band_half))
        region = _crop(wall_pair_mask, (x0, lo, x1, hi))
        if region.size == 0:
            return []
        ink_frac = (region > 0).mean(axis=1)

    runs: list[tuple[float, float]] = []
    in_run = False
    start = 0
    for i, frac in enumerate(ink_frac):
        is_open = float(frac) <= max_ink_frac
        if is_open and not in_run:
            start = i
            in_run = True
        elif not is_open and in_run:
            run_lo = lo + start
            run_hi = lo + i
            gap_px = run_hi - run_lo
            if gap_px >= min_gap_px:
                runs.append((float(run_lo), float(run_hi)))
            in_run = False
    if in_run:
        run_lo = lo + start
        run_hi = lo + len(ink_frac)
        gap_px = run_hi - run_lo
        if gap_px >= min_gap_px:
            runs.append((float(run_lo), float(run_hi)))
    return runs


def _symbol_runs_along_wall(
    wall_pair_mask: np.ndarray,
    ink_mask: np.ndarray,
    horiz: bool,
    axis: float,
    span_lo: float,
    span_hi: float,
    band_half: int,
    min_gap_px: float,
    max_gap_px: float,
    end_inset_px: int,
    px_per_unit: float,
) -> list[tuple[float, float]]:
    """Per-window spans for periodic glyph markers on a continuous centre-channel.

    Some plans draw windows as a regularly-spaced series of compact markers on
    the wall centreline (the wall pair never breaks). Detect on-axis feature
    blobs (present in ``ink_mask``, absent from ``wall_pair_mask`` so the wall
    strokes and crossing walls drop out); a series of >= ``SYMBOL_MIN_MARKERS``
    with consistent spacing is emitted as one window per marker.
    """
    lo = int(round(span_lo)) + end_inset_px
    hi = int(round(span_hi)) - end_inset_px
    if hi - lo < min_gap_px:
        return []

    chan = min(max(band_half - 1, 1), max(2, int(round(GLAZE_BAND_FT * px_per_unit))))
    a = int(round(axis))
    feat = (ink_mask > 0) & (wall_pair_mask == 0)
    if horiz:
        region = feat[a - chan:a + chan + 1, lo:hi]
        prof = region.mean(axis=0) if region.size else None
    else:
        region = feat[lo:hi, a - chan:a + chan + 1]
        prof = region.mean(axis=1) if region.size else None
    if prof is None or prof.size == 0:
        return []

    centers = _marker_centers(prof, lo, px_per_unit)
    if len(centers) < SYMBOL_MIN_MARKERS:
        return []

    spacings = np.diff(centers)
    mean_sp = float(spacings.mean())
    if mean_sp <= 0:
        return []
    cv = float(spacings.std() / mean_sp)
    if cv > SYMBOL_SPACING_CV_MAX:
        return []

    # Emit one window per marker, tiling the series at the half-spacing so the
    # spans centre on the glyphs and stay within the window-width band.
    half = max(min_gap_px / 2.0, min(mean_sp / 2.0, max_gap_px / 2.0))
    spans: list[tuple[float, float]] = []
    for c in centers:
        spans.append((c - half, c + half))
    return spans


def _marker_centers(prof: np.ndarray, offset: int, px_per_unit: float) -> list[float]:
    """Centres (in image coords) of compact on-axis marker blobs in ``prof``."""
    present = prof >= SYMBOL_MARKER_FRAC
    bridge = max(0, int(round(SYMBOL_MARKER_GAP_FT * px_per_unit)))
    min_w = max(1, int(round(SYMBOL_MARKER_MIN_FT * px_per_unit)))
    max_w = max(min_w, int(round(SYMBOL_MARKER_MAX_FT * px_per_unit)))

    centers: list[float] = []
    n = present.size
    i = 0
    while i < n:
        if not present[i]:
            i += 1
            continue
        j = i
        gap = 0
        last = i
        while j < n:
            if present[j]:
                last = j
                gap = 0
            else:
                gap += 1
                if gap > bridge:
                    break
            j += 1
        width = last - i + 1
        if min_w <= width <= max_w:
            centers.append(offset + (i + last) / 2.0)
        i = j
    return centers


def _refine_window_span(
    ink_mask: np.ndarray,
    horiz: bool,
    rect: tuple[int, int, int, int],
    run_lo: float,
    run_hi: float,
) -> tuple[float, float, bool]:
    """Tighten along-wall span to jamb/sill ink; fall back if ambiguous."""
    orig_span = run_hi - run_lo
    if orig_span <= 0:
        return run_lo, run_hi, False

    x0, y0, x1, y1 = rect
    region = _crop(ink_mask, (x0, y0, x1, y1))
    if region.size == 0:
        return run_lo, run_hi, False

    if horiz:
        along = (region > 0).mean(axis=0)
        offset = x0
    else:
        along = (region > 0).mean(axis=1)
        offset = y0

    if along.size < 3:
        return run_lo, run_hi, False

    ink_idx = np.where(along >= 0.12)[0]
    if ink_idx.size == 0:
        return run_lo, run_hi, False

    new_lo = float(offset + ink_idx[0])
    new_hi = float(offset + ink_idx[-1] + 1)
    new_span = new_hi - new_lo
    if new_span < orig_span * SPAN_REFINE_MIN_FRAC:
        return run_lo, run_hi, False
    if new_span > orig_span * 1.05:
        return run_lo, run_hi, False
    return new_lo, new_hi, True


def _make_window(
    horiz: bool,
    axis: float,
    span_lo: float,
    span_hi: float,
    band_half: int,
    host_wall: dict,
    px_per_unit: float,
    unit_label: str,
    *,
    ink_mask: Optional[np.ndarray] = None,
    evidence_detail: str = "sill",
) -> dict:
    rect = _gap_rect(horiz, axis, span_lo, span_hi, band_half)
    if ink_mask is not None:
        span_lo, span_hi, _ = _refine_window_span(ink_mask, horiz, rect, span_lo, span_hi)
        rect = _gap_rect(horiz, axis, span_lo, span_hi, band_half)

    gap_px = span_hi - span_lo
    width_units = gap_px / px_per_unit
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    return {
        "id": "",
        "host_wall_id": host_wall.get("id"),
        "bbox_px": list(rect),
        "center_px": [cx, cy],
        "width": f"{width_units:.2f} {unit_label}",
        "width_raw": round(width_units, 2),
        "is_exterior": True,
        "evidence": evidence_detail if evidence_detail in ("sill", "symbol") else "sill",
        "_horiz": horiz,
        "_axis": axis,
        "_span_lo": span_lo,
        "_span_hi": span_hi,
        "_strategy": host_wall.get("_detect_strategy", ""),
    }


def _wall_axis_span(wall: dict) -> Optional[tuple[bool, float, float, float]]:
    """(horiz, axis, span_lo, span_hi) for a wall, or None if degenerate."""
    coords = wall.get("px_coords")
    if not coords or len(coords) < 4:
        return None
    horiz = _orient(coords)
    if horiz is None:
        return None
    x1, y1, x2, y2 = coords
    if horiz:
        return True, (y1 + y2) / 2.0, min(x1, x2), max(x1, x2)
    return False, (x1 + x2) / 2.0, min(y1, y2), max(y1, y2)


def _building_envelope(walls: list[dict]) -> Optional[tuple[float, float, float, float]]:
    """(x_lo, x_hi, y_lo, y_hi) from axis extremes over all walls.

    The outermost wall on each side defines the building edge regardless of its
    exterior/interior tag, so perimeter walls mis-tagged interior still land on
    this envelope while true interior walls fall inside it.
    """
    h_axes, v_axes = [], []
    for w in walls:
        info = _wall_axis_span(w)
        if info is None:
            continue
        horiz, axis, span_lo, span_hi = info
        if span_hi - span_lo < MIN_ENVELOPE_AXIS_SPAN_PX:
            continue
        (h_axes if horiz else v_axes).append(axis)
    if not h_axes or not v_axes:
        return None
    return min(v_axes), max(v_axes), min(h_axes), max(h_axes)


def _on_building_envelope(
    horiz: bool, axis: float,
    envelope: Optional[tuple[float, float, float, float]],
    tol: float,
) -> bool:
    if envelope is None:
        return False
    x_lo, x_hi, y_lo, y_hi = envelope
    if horiz:
        return abs(axis - y_lo) <= tol or abs(axis - y_hi) <= tol
    return abs(axis - x_lo) <= tol or abs(axis - x_hi) <= tol


def _wall_on_building_envelope(
    wall: dict,
    envelope: Optional[tuple[float, float, float, float]],
    tol: float,
) -> bool:
    if envelope is None:
        # Degenerate wall lists (common in unit tests): fall back to the tag.
        return bool(wall.get("is_exterior"))
    info = _wall_axis_span(wall)
    if info is None:
        return False
    horiz, axis, _, _ = info
    return _on_building_envelope(horiz, axis, envelope, tol)


def _wall_length_ft(wall: dict, px_per_unit: float) -> float:
    raw = wall.get("length_raw")
    if raw is not None:
        return float(raw)
    info = _wall_axis_span(wall)
    if info is None:
        return 0.0
    _, _, span_lo, span_hi = info
    return (span_hi - span_lo) / px_per_unit


def _wall_scannable_for_windows(
    wall: dict,
    envelope: Optional[tuple[float, float, float, float]],
    envelope_tol: float,
    *,
    crop_mode: bool,
    px_per_unit: float,
) -> bool:
    """Envelope-perimeter gate plus crop_mode guards for coaxial interior stubs."""
    if not _wall_on_building_envelope(wall, envelope, envelope_tol):
        return False
    if crop_mode and not wall.get("is_exterior"):
        if _wall_length_ft(wall, px_per_unit) < MIN_SCAN_WALL_FT_CROP:
            return False
    return True


def _parent_wall_id(host_wall_id: Optional[str]) -> str:
    if not host_wall_id:
        return ""
    return host_wall_id.split(".", 1)[0]


def _opening_flanked_by_wall(
    wall_pair_mask: np.ndarray,
    horiz: bool,
    axis: float,
    run_lo: float,
    run_hi: float,
    band_half: int,
    probe_px: int,
) -> bool:
    """True when double-stroke wall ink continues on both sides of the gap.

    Distinguishes a real opening in a wall from a low-ink run over whitespace
    on a phantom exterior sub-segment (overshooting footprint polygon).
    """
    if probe_px <= 0:
        return True

    def _flank_cover(lo: float, hi: float) -> float:
        """Fraction of along-axis positions in the probe that carry wall ink."""
        if hi <= lo:
            return 0.0
        if horiz:
            y0, y1 = int(round(axis - band_half)), int(round(axis + band_half))
            region = _crop(wall_pair_mask, (int(round(lo)), y0, int(round(hi)), y1))
            along_axis = 0
        else:
            x0, x1 = int(round(axis - band_half)), int(round(axis + band_half))
            region = _crop(wall_pair_mask, (x0, int(round(lo)), x1, int(round(hi)))) 
            along_axis = 1
        if region.size == 0:
            return 0.0
        present = (region > 0).any(axis=along_axis)
        return float(present.mean()) if present.size else 0.0

    left = _flank_cover(run_lo - probe_px, run_lo)
    right = _flank_cover(run_hi, run_hi + probe_px)
    return left >= FLANK_COVER_FRAC and right >= FLANK_COVER_FRAC


def _window_confidence(
    sill_cover: float,
    evidence_detail: str,
    gap_open: bool,
    bilateral: bool,
    is_dimension: bool,
) -> tuple[float, dict]:
    """Weighted multi-cue confidence in [~0, 1] plus a per-cue breakdown.

    Decoupled from the binary tiers in ``gap_sill_evidence`` so acceptance can
    trade sill strength against opening corroboration; calibrated against the
    validation set (see ``validation/window_metrics.py``).
    """
    sill_c = (sill_cover - SILL_RAMP_LO) / (SILL_RAMP_HI - SILL_RAMP_LO)
    sill_c = max(0.0, min(1.0, sill_c))

    score = W_SILL * sill_c
    if gap_open:
        score += W_OPEN
    if bilateral:
        score += W_BILATERAL
    if evidence_detail == "sill+triple":
        score += W_TRIPLE_BONUS
    if is_dimension and not bilateral:
        score -= W_DIM_PENALTY

    cues = {
        "sill_cover": round(float(sill_cover), 3),
        "sill_component": round(sill_c, 3),
        "gap_open": bool(gap_open),
        "bilateral": bool(bilateral),
        "evidence_detail": evidence_detail,
        "dimension_penalty": bool(is_dimension and not bilateral),
    }
    return score, cues


def _building_x_span(walls: list[dict]) -> tuple[float, float]:
    xs: list[float] = []
    for wall in walls:
        coords = wall.get("px_coords")
        if coords and len(coords) >= 4:
            xs.extend((float(coords[0]), float(coords[2])))
    if not xs:
        return 0.0, 0.0
    return min(xs), max(xs)


def _wall_fill_hatch_metrics(
    ink_mask: np.ndarray,
    wall_pair_mask: np.ndarray,
    horiz: bool,
    axis: float,
    run_lo: float,
    run_hi: float,
    band_half: int,
) -> tuple[int, float, float]:
    """Peak count, mean spacing (px), and spacing CV of feature ink along a span."""
    rect = _gap_rect(horiz, axis, run_lo, run_hi, band_half)
    x0, y0, x1, y1 = rect
    region_ink = _crop(ink_mask, (x0, y0, x1, y1))
    region_wall = _crop(wall_pair_mask, (x0, y0, x1, y1))
    if region_ink.size == 0:
        return 0, 0.0, 99.0
    feat = (region_ink > 0) & (region_wall == 0)
    along = feat.mean(axis=0 if horiz else 1)
    if along.size < 6:
        return 0, 0.0, 99.0

    peak_idx: list[int] = []
    for i, val in enumerate(along):
        if float(val) >= HATCH_PEAK_FRAC:
            if not peak_idx or i - peak_idx[-1] > 2:
                peak_idx.append(i)
            else:
                peak_idx[-1] = i

    if len(peak_idx) < 2:
        return len(peak_idx), 0.0, 99.0
    spacings = np.diff(peak_idx).astype(float)
    mean_sp = float(spacings.mean())
    cv = float(spacings.std() / (mean_sp + 1e-6))
    return len(peak_idx), mean_sp, cv


def _is_wall_fill_hatch_pattern(
    ink_mask: np.ndarray,
    wall_pair_mask: np.ndarray,
    horiz: bool,
    axis: float,
    run_lo: float,
    run_hi: float,
    band_half: int,
    px_per_unit: float,
) -> bool:
    """Repeating wall-fill hatch (CMU/storefront hash) masquerading as window sills."""
    if not horiz:
        return False
    n_peaks, mean_sp_px, cv = _wall_fill_hatch_metrics(
        ink_mask, wall_pair_mask, horiz, axis, run_lo, run_hi, band_half,
    )
    if n_peaks < 3:
        return False
    mean_sp_ft = mean_sp_px / px_per_unit
    if n_peaks >= 5 and mean_sp_ft < 1.25 and cv < 0.70:
        return True
    return (
        n_peaks >= 3
        and 1.0 <= mean_sp_ft < 1.25
        and cv < 0.70
    )


def _weak_east_facade_opening(
    w: dict,
    walls: list[dict],
    wall_pair_mask: np.ndarray,
    px_per_unit: float,
    band_half: int,
    open_gap_max: float,
) -> bool:
    """Low-px east-facade horizontals need opening corroboration, not sill alone."""
    if px_per_unit >= HATCH_LOW_PX_PER_FT:
        return False
    horiz, axis, lo, hi = _annotate_window_span(w)
    if not horiz:
        return False
    x_lo, x_hi = _building_x_span(walls)
    span = x_hi - x_lo
    if span <= 0:
        return False
    if w["center_px"][0] <= x_lo + 0.5 * span:
        return False
    rect = _gap_rect(horiz, axis, lo, hi, band_half)
    bilateral = _gap_has_bilateral_break(
        wall_pair_mask, horiz, rect, max_ink_frac=open_gap_max,
    )
    gap_open = _gap_is_open(wall_pair_mask, horiz, rect, max_ink_frac=open_gap_max)
    return not (bilateral or gap_open)


def _crop_mode_precision_filter(
    windows: list[dict],
    walls: list[dict],
    wall_pair_mask: np.ndarray,
    ink_mask: np.ndarray,
    px_per_unit: float,
    *,
    open_gap_max: float,
) -> list[dict]:
    """Crop-mode precision guards for wall hatch and weak east-facade sills."""
    if not windows or ink_mask is None:
        return windows
    band_half = max(4, int(math.ceil(0.75 * px_per_unit)))
    kept: list[dict] = []
    for w in windows:
        horiz, axis, lo, hi = _annotate_window_span(w)
        if _is_wall_fill_hatch_pattern(
            ink_mask, wall_pair_mask, horiz, axis, lo, hi, band_half, px_per_unit,
        ):
            continue
        if _weak_east_facade_opening(
            w, walls, wall_pair_mask, px_per_unit, band_half, open_gap_max,
        ):
            continue
        kept.append(w)
    return kept


def _near_any(center: list[float], others: list[list[float]], dist: float) -> bool:
    cx, cy = center
    return any(math.hypot(cx - o[0], cy - o[1]) < dist for o in others)


def _spans_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return lo_a < hi_b and lo_b < hi_a


def _merge_span_dedup(
    windows: list[dict],
    px_per_unit: float,
    axis_tol_px: int,
) -> list[dict]:
    """Merge detections that overlap along the same wall axis (not merely adjacent)."""
    groups: dict[tuple[bool, int], list[dict]] = {}
    for w in windows:
        horiz = w.get("_horiz", True)
        axis = w.get("_axis", w["center_px"][1 if horiz else 0])
        w["_span_lo"] = w.get("_span_lo", float(w["bbox_px"][0 if horiz else 1]))
        w["_span_hi"] = w.get("_span_hi", float(w["bbox_px"][2 if horiz else 3]))
        key = (horiz, int(round(axis / max(axis_tol_px, 1))))
        groups.setdefault(key, []).append(w)

    merged: list[dict] = []
    for horiz_key in sorted(groups, key=lambda k: (k[0], k[1])):
        horiz = horiz_key[0]
        group = groups[horiz_key]
        group.sort(key=lambda w: w["_span_lo"])
        current: Optional[dict] = None
        for w in group:
            if current is None:
                current = w
                continue
            overlap = _spans_overlap(
                current["_span_lo"], current["_span_hi"],
                w["_span_lo"], w["_span_hi"],
            )
            # Merge duplicate strategy hits on the same opening only.
            same_opening_dup = (
                overlap
                and current.get("_strategy") != w.get("_strategy")
            )
            if overlap and (same_opening_dup or current.get("_strategy") == w.get("_strategy")):
                if w["_span_hi"] - w["_span_lo"] > current["_span_hi"] - current["_span_lo"]:
                    current.update(w)
                current["_span_lo"] = min(current["_span_lo"], w["_span_lo"])
                current["_span_hi"] = max(current["_span_hi"], w["_span_hi"])
                lo, hi = current["_span_lo"], current["_span_hi"]
                if horiz:
                    current["bbox_px"][0] = int(round(lo))
                    current["bbox_px"][2] = int(round(hi))
                else:
                    current["bbox_px"][1] = int(round(lo))
                    current["bbox_px"][3] = int(round(hi))
                width_units = (hi - lo) / px_per_unit
                current["width_raw"] = round(width_units, 2)
                unit = current.get("width", " ft").split()[-1] if current.get("width") else "ft"
                current["width"] = f"{width_units:.2f} {unit}"
                cx = (current["bbox_px"][0] + current["bbox_px"][2]) / 2.0
                cy = (current["bbox_px"][1] + current["bbox_px"][3]) / 2.0
                current["center_px"] = [cx, cy]
            else:
                merged.append(current)
                current = w
        if current is not None:
            merged.append(current)

    for w in merged:
        for k in ("_span_lo", "_span_hi", "_horiz", "_axis", "_strategy"):
            w.pop(k, None)
    return merged


def _annotate_window_span(w: dict) -> tuple[bool, float, float, float]:
    b = w["bbox_px"]
    horiz = (b[2] - b[0]) >= (b[3] - b[1])
    if horiz:
        return horiz, (b[1] + b[3]) / 2.0, float(b[0]), float(b[2])
    return horiz, (b[0] + b[2]) / 2.0, float(b[1]), float(b[3])


def _apply_span_to_window(w: dict, horiz: bool, lo: float, hi: float, px_per_unit: float) -> None:
    b = w["bbox_px"]
    if horiz:
        b[0] = int(round(lo))
        b[2] = int(round(hi))
    else:
        b[1] = int(round(lo))
        b[3] = int(round(hi))
    width_units = (hi - lo) / px_per_unit
    w["width_raw"] = round(width_units, 2)
    unit = w.get("width", " ft").split()[-1] if w.get("width") else "ft"
    w["width"] = f"{width_units:.2f} {unit}"
    w["center_px"] = [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]


def _merge_over_split_fragments(
    windows: list[dict],
    px_per_unit: float,
    axis_tol_px: int,
) -> list[dict]:
    """Merge touching narrow fragments produced by over-splitting one opening."""
    sill = [w for w in windows if w.get("evidence") != "symbol"]
    symbols = [w for w in windows if w.get("evidence") == "symbol"]
    if len(sill) < 2:
        return windows

    touch_gap_px = max(2.0, TOUCH_MERGE_GAP_FT * px_per_unit)
    over_split_px = OVER_SPLIT_WIDTH_FT * px_per_unit
    max_combined_px = MERGE_COMBINED_MAX_FT * px_per_unit

    def _group_key(w: dict) -> tuple:
        horiz, axis, lo, hi = _annotate_window_span(w)
        w["_horiz"] = horiz
        w["_axis"] = axis
        w["_span_lo"] = lo
        w["_span_hi"] = hi
        axis_key = int(round(axis / max(axis_tol_px, 1)))
        return horiz, axis_key

    groups: dict[tuple, list[dict]] = {}
    for w in sill:
        groups.setdefault(_group_key(w), []).append(w)

    merged: list[dict] = []
    for group in groups.values():
        group.sort(key=lambda w: w["_span_lo"])
        current: Optional[dict] = None
        for w in group:
            if current is None:
                current = w
                continue
            gap = w["_span_lo"] - current["_span_hi"]
            combined = w["_span_hi"] - current["_span_lo"]
            cur_w = current["_span_hi"] - current["_span_lo"]
            new_w = w["_span_hi"] - w["_span_lo"]
            touch_merge = (
                gap <= touch_gap_px
                and combined <= max_combined_px
                and min(cur_w, new_w) <= over_split_px
            )
            if touch_merge:
                lo = min(current["_span_lo"], w["_span_lo"])
                hi = max(current["_span_hi"], w["_span_hi"])
                if new_w > cur_w:
                    current["host_wall_id"] = w.get("host_wall_id", current.get("host_wall_id"))
                    current["confidence"] = max(
                        float(current.get("confidence") or 0),
                        float(w.get("confidence") or 0),
                    )
                current["_span_lo"] = lo
                current["_span_hi"] = hi
                _apply_span_to_window(current, current["_horiz"], lo, hi, px_per_unit)
            else:
                merged.append(current)
                current = w
        if current is not None:
            merged.append(current)

    out = merged + symbols
    for w in out:
        for k in ("_horiz", "_axis", "_span_lo", "_span_hi"):
            w.pop(k, None)
    return out


def _merge_across_mullion_gaps(
    windows: list[dict],
    wall_pair_mask: np.ndarray,
    px_per_unit: float,
    axis_tol_px: int,
    *,
    band_half: int,
    open_gap_max: float,
) -> list[dict]:
    """Bridge adjacent sill fragments split by a structural mullion (crop_mode).

    Low-px plans sometimes split one tall opening at an interior jamb peak while
    real adjacent windows on the same parent wall stay separated by wider gaps
    and form a longer combined span than curtain-wall panels allow.
    """
    sill = [w for w in windows if w.get("evidence") != "symbol"]
    symbols = [w for w in windows if w.get("evidence") == "symbol"]
    if len(sill) < 2:
        return windows

    gap_max_px = MULLION_GAP_MAX_FT_CROP * px_per_unit
    combined_max_px = MULLION_COMBINED_MAX_FT_CROP * px_per_unit

    def _group_key(w: dict) -> tuple:
        horiz, axis, lo, hi = _annotate_window_span(w)
        w["_horiz"] = horiz
        w["_axis"] = axis
        w["_span_lo"] = lo
        w["_span_hi"] = hi
        axis_key = int(round(axis / max(axis_tol_px, 1)))
        parent = _parent_wall_id(w.get("host_wall_id"))
        return horiz, axis_key, parent

    groups: dict[tuple, list[dict]] = {}
    for w in sill:
        groups.setdefault(_group_key(w), []).append(w)

    merged: list[dict] = []
    for group in groups.values():
        group.sort(key=lambda w: w["_span_lo"])
        current: Optional[dict] = None
        for w in group:
            if current is None:
                current = w
                continue
            if current["_horiz"]:
                merged.append(current)
                current = w
                continue
            gap_lo = current["_span_hi"]
            gap_hi = w["_span_lo"]
            gap = gap_hi - gap_lo
            combined = w["_span_hi"] - current["_span_lo"]
            cur_w = current["_span_hi"] - current["_span_lo"]
            new_w = w["_span_hi"] - w["_span_lo"]
            frag_max_px = MULLION_FRAGMENT_MAX_FT_CROP * px_per_unit
            if not (
                0 < gap <= gap_max_px
                and combined <= combined_max_px
                and cur_w <= frag_max_px
                and new_w <= frag_max_px
            ):
                merged.append(current)
                current = w
                continue
            rect = _gap_rect(
                current["_horiz"], current["_axis"], gap_lo, gap_hi, band_half,
            )
            if _gap_is_open(wall_pair_mask, current["_horiz"], rect, max_ink_frac=open_gap_max):
                merged.append(current)
                current = w
                continue
            lo = min(current["_span_lo"], w["_span_lo"])
            hi = max(current["_span_hi"], w["_span_hi"])
            if (w["_span_hi"] - w["_span_lo"]) > (current["_span_hi"] - current["_span_lo"]):
                current["host_wall_id"] = w.get("host_wall_id", current.get("host_wall_id"))
                current["confidence"] = max(
                    float(current.get("confidence") or 0),
                    float(w.get("confidence") or 0),
                )
            current["_span_lo"] = lo
            current["_span_hi"] = hi
            _apply_span_to_window(current, current["_horiz"], lo, hi, px_per_unit)
        if current is not None:
            merged.append(current)

    out = merged + symbols
    for w in out:
        for k in ("_horiz", "_axis", "_span_lo", "_span_hi"):
            w.pop(k, None)
    return out


def _symbol_perp_extends(
    ink_mask: np.ndarray,
    wall_pair_mask: np.ndarray,
    horiz: bool,
    axis: float,
    run_lo: float,
    run_hi: float,
    band_half: int,
) -> bool:
    """True when centre-channel ink continues far perpendicular (a crossing wall).

    A window glyph is confined to the wall band; a perpendicular interior wall
    meeting the host keeps going past it, so feature ink persists well beyond
    the band on at least one side.
    """
    feat = (ink_mask > 0) & (wall_pair_mask == 0)
    lo, hi = int(round(run_lo)), int(round(run_hi))
    near = band_half
    far0 = 2 * band_half
    far1 = 4 * band_half
    a = int(round(axis))

    def cover(c0: int, c1: int) -> float:
        """Fraction of perpendicular positions in the slab reached by feature ink.

        Reduced with ``any`` along the wall so a thin crossing line still scores
        high (it touches every perpendicular position it passes through).
        """
        if horiz:
            region = _crop(feat.astype(np.uint8), (lo, c0, hi, c1))
            if region.size == 0:
                return 0.0
            return float((region > 0).any(axis=1).mean())  # perp = rows
        region = _crop(feat.astype(np.uint8), (c0, lo, c1, hi))
        if region.size == 0:
            return 0.0
        return float((region > 0).any(axis=0).mean())       # perp = cols

    near_cov = max(cover(a - near, a), cover(a, a + near))
    far_cov = max(cover(a - far1, a - far0), cover(a + far0, a + far1))
    return far_cov >= 0.30 and far_cov >= 0.6 * max(near_cov, 1e-6)


def _evaluate_symbol_candidate(
    record: dict,
    horiz: bool,
    axis: float,
    run_lo: float,
    run_hi: float,
    band_half: int,
    host_wall: dict,
    wall_pair_mask: np.ndarray,
    ink_mask: np.ndarray,
    px_per_unit: float,
    unit_label: str,
    open_gap_max: float,
    door_centers: list[list[float]],
    door_dedup_dist: float,
    strategy: str,
) -> dict:
    """Acceptance for symbol-on-wall candidates (continuous wall, no open gap)."""
    rect = record["bbox_px"]

    # The glyph rides a continuous wall; if the band reads open, this is an
    # opening and belongs to the opening-based strategies (avoid double-count).
    if _gap_is_open(wall_pair_mask, horiz, tuple(rect), max_ink_frac=open_gap_max):
        record["status"] = "rejected"
        record["reject_reason"] = "not_symbol_open_gap"
        return record

    # Reject dimension strings sitting beside the wall.
    if _looks_like_dimension_line(ink_mask, horiz, tuple(rect), wall_pair_mask=wall_pair_mask):
        record["status"] = "rejected"
        record["reject_reason"] = "dimension_line"
        return record

    # The host wall must continue on both sides (this is an in-wall glyph).
    probe_px = max(band_half, int(round(FLANK_PROBE_FT * px_per_unit)))
    if not _opening_flanked_by_wall(
        wall_pair_mask, horiz, axis, run_lo, run_hi, band_half, probe_px,
    ):
        record["status"] = "rejected"
        record["reject_reason"] = "no_wall_flank"
        return record

    # Reject perpendicular crossing walls (feature ink extends past the band).
    if _symbol_perp_extends(
        ink_mask, wall_pair_mask, horiz, axis, run_lo, run_hi, band_half,
    ):
        record["status"] = "rejected"
        record["reject_reason"] = "crossing_wall"
        return record

    record["confidence"] = SYMBOL_CONFIDENCE
    record["cues"] = {"evidence_detail": "symbol"}
    wall_tag = {**host_wall, "_detect_strategy": strategy}
    cand = _make_window(
        horiz, axis, run_lo, run_hi, band_half,
        wall_tag, px_per_unit, unit_label,
        ink_mask=None,
        evidence_detail="symbol",
    )
    cand["confidence"] = SYMBOL_CONFIDENCE
    if _near_any(cand["center_px"], door_centers, door_dedup_dist):
        record["status"] = "rejected"
        record["reject_reason"] = "near_door"
        record["window"] = cand
        return record

    record["window"] = cand
    record["evidence_detail"] = "symbol"
    return record


def _evaluate_candidate(
    horiz: bool,
    axis: float,
    run_lo: float,
    run_hi: float,
    band_half: int,
    host_wall: dict,
    wall_pair_mask: np.ndarray,
    ink_mask: np.ndarray,
    px_per_unit: float,
    unit_label: str,
    open_gap_max: float,
    door_centers: list[list[float]],
    door_dedup_dist: float,
    min_gap_px: float,
    max_gap_px: float,
    *,
    strategy: str,
    require_open_check: bool,
    strict_open: bool = False,
    symbol: bool = False,
) -> dict:
    """Build accepted/rejected candidate record for detect + debug."""
    gap_px = run_hi - run_lo
    width_raw = gap_px / px_per_unit
    rect = _gap_rect(horiz, axis, run_lo, run_hi, band_half)
    record = {
        "strategy": strategy,
        "host_wall_id": host_wall.get("id"),
        "span_lo": run_lo,
        "span_hi": run_hi,
        "width_raw": round(width_raw, 2),
        "bbox_px": list(rect),
        "status": "accepted",
        "reject_reason": None,
    }

    if gap_px < min_gap_px or gap_px > max_gap_px:
        record["status"] = "rejected"
        record["reject_reason"] = "width_out_of_range"
        return record

    if symbol:
        return _evaluate_symbol_candidate(
            record, horiz, axis, run_lo, run_hi, band_half, host_wall,
            wall_pair_mask, ink_mask, px_per_unit, unit_label,
            open_gap_max, door_centers, door_dedup_dist, strategy,
        )

    has_sill, evidence_detail, sill_cover = gap_sill_evidence(
        ink_mask, horiz, rect, wall_pair_mask=wall_pair_mask,
    )
    bilateral = _gap_has_bilateral_break(
        wall_pair_mask, horiz, rect, max_ink_frac=open_gap_max,
    )
    gap_open = _gap_is_open(
        wall_pair_mask, horiz, rect, max_ink_frac=open_gap_max,
    )
    is_dimension = _looks_like_dimension_line(
        ink_mask, horiz, rect, wall_pair_mask=wall_pair_mask,
    )

    confidence, cues = _window_confidence(
        sill_cover, evidence_detail, gap_open, bilateral, is_dimension,
    )
    record["confidence"] = round(confidence, 3)
    record["cues"] = cues

    # A usable sill is required (tiered detector encodes full / triple /
    # partial-on-very-open-gap). Note: many real windows do not register as an
    # open gap in wall_pair_mask, so opening evidence is scored, not required.
    if not has_sill:
        record["status"] = "rejected"
        record["reject_reason"] = "no_sill"
        return record

    # Collinear gaps are only proposed when two real sub-segments flank the
    # span, so a filled (un-open) gap there is a hard contradiction. Interior
    # envelope walls (Phase 3) are riskier, so they also require opening
    # corroboration: an open band or a bilateral stroke break.
    if require_open_check and not gap_open:
        record["status"] = "rejected"
        record["reject_reason"] = "ink_not_open"
        return record
    if strict_open and not (gap_open or bilateral):
        record["status"] = "rejected"
        record["reject_reason"] = "ink_not_open"
        return record

    if cues["dimension_penalty"]:
        record["status"] = "rejected"
        record["reject_reason"] = "dimension_line"
        return record

    # In-segment openings must sit in a real wall: double-stroke ink has to
    # continue on both sides of the gap. Collinear gaps already have two real
    # flanking sub-segments by construction.
    if not require_open_check:
        probe_px = max(band_half, int(round(FLANK_PROBE_FT * px_per_unit)))
        if not _opening_flanked_by_wall(
            wall_pair_mask, horiz, axis, run_lo, run_hi, band_half, probe_px,
        ):
            record["status"] = "rejected"
            record["reject_reason"] = "no_wall_flank"
            return record

    wall_tag = {**host_wall, "_detect_strategy": strategy}
    cand = _make_window(
        horiz, axis, run_lo, run_hi, band_half,
        wall_tag, px_per_unit, unit_label,
        ink_mask=ink_mask,
        evidence_detail=evidence_detail,
    )
    cand["confidence"] = round(confidence, 3)
    if _near_any(cand["center_px"], door_centers, door_dedup_dist):
        record["status"] = "rejected"
        record["reject_reason"] = "near_door"
        record["window"] = cand
        return record

    record["window"] = cand
    record["evidence_detail"] = evidence_detail
    return record


def detect_window_candidates(
    walls: list[dict],
    wall_pair_mask: np.ndarray,
    ink_mask: Optional[np.ndarray],
    px_per_unit: float,
    unit_label: str = "ft",
    axis_tol_px: Optional[int] = None,
    crop_mode: bool = False,
    doors: Optional[list[dict]] = None,
) -> list[dict]:
    """Return all window candidates with acceptance/rejection metadata."""
    if ink_mask is None:
        return []

    if axis_tol_px is None:
        axis_tol_px = max(12, int(0.6 * px_per_unit))
    win_axis_tol = max(axis_tol_px, int(1.0 * px_per_unit))

    to_ft = 1.0 if unit_label == "ft" else 3.2808
    min_ft = WINDOW_MIN_FT_CROP if crop_mode else WINDOW_MIN_FT
    max_ft = WINDOW_MAX_FT_CROP if crop_mode else WINDOW_MAX_FT
    min_gap_px = min_ft / to_ft * px_per_unit
    max_gap_px = max_ft / to_ft * px_per_unit
    open_gap_max = OPEN_GAP_MAX_CROP if crop_mode else OPEN_GAP_MAX_INK_FRAC
    band_half = max(4, int(math.ceil(0.75 * px_per_unit)))
    end_inset_px = max(band_half, int(0.5 * px_per_unit))

    door_centers = [d["center_px"] for d in (doors or []) if d.get("center_px")]
    door_dedup_dist = max(6.0, 0.5 * px_per_unit)

    jamb_peak_frac = JAMB_INK_PEAK_FRAC_CROP if crop_mode else JAMB_INK_PEAK_FRAC
    split_min_px = (
        SPLIT_MIN_SEGMENT_FT_CROP / to_ft * px_per_unit if crop_mode else None
    )

    candidates: list[dict] = []

    # Scan envelope-perimeter walls only (see ENVELOPE_TOL_FT).
    envelope = _building_envelope(walls)
    envelope_tol = max(band_half, ENVELOPE_TOL_FT * px_per_unit)

    for wall in walls:
        info = _wall_axis_span(wall)
        if info is None:
            continue
        horiz, axis, span_lo, span_hi = info
        if not _wall_scannable_for_windows(
            wall, envelope, envelope_tol, crop_mode=crop_mode, px_per_unit=px_per_unit,
        ):
            continue
        exterior = bool(wall.get("is_exterior"))

        raw_runs = _open_runs_along_wall(
            wall_pair_mask, horiz, axis, span_lo, span_hi,
            band_half, min_gap_px, max_gap_px, open_gap_max, end_inset_px,
        )
        runs = _expand_runs(
            wall_pair_mask, ink_mask, horiz, axis, raw_runs,
            band_half, min_gap_px, max_gap_px,
            jamb_peak_frac=jamb_peak_frac,
            split_min_px=split_min_px,
        )
        for run_lo, run_hi in runs:
            candidates.append(_evaluate_candidate(
                horiz, axis, run_lo, run_hi, band_half, wall,
                wall_pair_mask, ink_mask, px_per_unit, unit_label,
                open_gap_max, door_centers, door_dedup_dist,
                min_gap_px, max_gap_px,
                strategy="in_segment",
                require_open_check=False,
                strict_open=not exterior,
            ))

        # Symbol-on-wall: periodic glyphs on a continuous centreline glazing line.
        symbol_spans = _symbol_runs_along_wall(
            wall_pair_mask, ink_mask, horiz, axis, span_lo, span_hi,
            band_half, min_gap_px, max_gap_px, end_inset_px, px_per_unit,
        )
        for run_lo, run_hi in symbol_spans:
            candidates.append(_evaluate_candidate(
                horiz, axis, run_lo, run_hi, band_half, wall,
                wall_pair_mask, ink_mask, px_per_unit, unit_label,
                open_gap_max, door_centers, door_dedup_dist,
                min_gap_px, max_gap_px,
                strategy="symbol_on_wall",
                require_open_check=False,
                symbol=True,
            ))

    groups: dict[bool, list[tuple[float, float, float, dict]]] = {True: [], False: []}
    for w in walls:
        if not _wall_scannable_for_windows(
            w, envelope, envelope_tol, crop_mode=crop_mode, px_per_unit=px_per_unit,
        ):
            continue
        coords = w.get("px_coords")
        if not coords or len(coords) < 4:
            continue
        horiz = _orient(coords)
        if horiz is None:
            continue
        x1, y1, x2, y2 = coords
        if horiz:
            groups[True].append(((y1 + y2) / 2.0, min(x1, x2), max(x1, x2), w))
        else:
            groups[False].append(((x1 + x2) / 2.0, min(y1, y2), max(y1, y2), w))

    for horiz, entries in groups.items():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: e[0])
        clusters: list[list[tuple[float, float, float, dict]]] = []
        for e in entries:
            if clusters and abs(e[0] - clusters[-1][0][0]) <= win_axis_tol:
                clusters[-1].append(e)
            else:
                clusters.append([e])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            cluster.sort(key=lambda e: e[1])
            for axis, span_lo, span_hi, wall_a, wall_b in _merged_interval_gaps(
                cluster, min_gap_px, max_gap_px,
            ):
                if not (
                    _wall_scannable_for_windows(
                        wall_a, envelope, envelope_tol,
                        crop_mode=crop_mode, px_per_unit=px_per_unit,
                    )
                    and _wall_scannable_for_windows(
                        wall_b, envelope, envelope_tol,
                        crop_mode=crop_mode, px_per_unit=px_per_unit,
                    )
                ):
                    continue
                ca, cb = wall_a.get("px_coords", []), wall_b.get("px_coords", [])
                len_a = math.hypot(ca[2] - ca[0], ca[3] - ca[1]) if len(ca) >= 4 else 0
                len_b = math.hypot(cb[2] - cb[0], cb[3] - cb[1]) if len(cb) >= 4 else 0
                host = wall_a if len_a >= len_b else wall_b
                candidates.append(_evaluate_candidate(
                    horiz, axis, span_lo, span_hi, band_half, host,
                    wall_pair_mask, ink_mask, px_per_unit, unit_label,
                    open_gap_max, door_centers, door_dedup_dist,
                    min_gap_px, max_gap_px,
                    strategy="collinear_gap",
                    require_open_check=True,
                ))

    return candidates


def detect_windows(
    walls: list[dict],
    wall_pair_mask: np.ndarray,
    ink_mask: Optional[np.ndarray],
    px_per_unit: float,
    unit_label: str = "ft",
    axis_tol_px: Optional[int] = None,
    crop_mode: bool = False,
    doors: Optional[list[dict]] = None,
) -> list[dict]:
    """Detect windows on exterior walls via the sill-line signature."""
    if axis_tol_px is None:
        axis_tol_px = max(12, int(0.6 * px_per_unit))

    candidates = detect_window_candidates(
        walls, wall_pair_mask, ink_mask, px_per_unit, unit_label,
        axis_tol_px=axis_tol_px, crop_mode=crop_mode, doors=doors,
    )

    accepted = [c["window"] for c in candidates if c["status"] == "accepted"]
    kept = _merge_span_dedup(accepted, px_per_unit, axis_tol_px)
    kept = _merge_over_split_fragments(kept, px_per_unit, axis_tol_px)
    if crop_mode:
        band_half = max(4, int(math.ceil(0.75 * px_per_unit)))
        kept = _merge_across_mullion_gaps(
            kept, wall_pair_mask, px_per_unit, axis_tol_px,
            band_half=band_half, open_gap_max=OPEN_GAP_MAX_CROP,
        )
        kept = [
            w for w in kept
            if not (
                (w["bbox_px"][2] - w["bbox_px"][0]) >= (w["bbox_px"][3] - w["bbox_px"][1])
                and (w["bbox_px"][2] - w["bbox_px"][0]) / px_per_unit > HORIZ_MAX_FT_CROP
            )
        ]
        kept = _crop_mode_precision_filter(
            kept, walls, wall_pair_mask, ink_mask, px_per_unit,
            open_gap_max=OPEN_GAP_MAX_CROP,
        )

    dedup_dist = max(6.0, 0.5 * px_per_unit)
    final: list[dict] = []
    for w in sorted(kept, key=lambda w: (w["center_px"][0], w["center_px"][1])):
        if _near_any(w["center_px"], [k["center_px"] for k in final], dedup_dist):
            continue
        final.append(w)

    for i, w in enumerate(final):
        w["id"] = f"win{i + 1}"
        for k in ("_horiz", "_axis", "_span_lo", "_span_hi", "_strategy"):
            w.pop(k, None)
    return final
