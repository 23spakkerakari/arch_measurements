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
    _crop,
    _gap_has_bilateral_break,
    _gap_has_sill,
    _gap_is_open,
    _gap_mean_ink_frac,
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
WINDOW_MIN_FT_REFINED = 1.5
WINDOW_MAX_FT_REFINED = 10.0
SPAN_REFINE_MIN_FRAC = 0.80

__all__ = [
    "detect_windows",
    "detect_window_candidates",
    "ink_mask_from_image",
]


def _smooth_ink_profile(ink_frac: np.ndarray, window_size: int) -> np.ndarray:
    """Moving average to suppress dimension-line speckle along wall scans."""
    if ink_frac.size == 0:
        return ink_frac
    if window_size <= 1:
        return ink_frac.astype(np.float64)
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(ink_frac.astype(np.float64), kernel, mode="same")


def _end_inset_px(band_half: int, px_per_unit: float, span_px: float) -> int:
    """Trim wall ends before scanning; shorter inset on long walls."""
    base = max(band_half, int(0.5 * px_per_unit))
    if span_px >= 15.0 * px_per_unit:
        return max(band_half, int(0.25 * px_per_unit))
    return base


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
    px_per_unit: float,
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

    smooth_size = max(3, int(0.15 * px_per_unit))
    ink_frac = _smooth_ink_profile(ink_frac, smooth_size)

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
            if min_gap_px <= gap_px <= max_gap_px:
                runs.append((float(run_lo), float(run_hi)))
            in_run = False
    if in_run:
        run_lo = lo + start
        run_hi = lo + len(ink_frac)
        gap_px = run_hi - run_lo
        if min_gap_px <= gap_px <= max_gap_px:
            runs.append((float(run_lo), float(run_hi)))
    return runs


def _refine_window_span(
    ink_mask: np.ndarray,
    horiz: bool,
    rect: tuple[int, int, int, int],
    run_lo: float,
    run_hi: float,
) -> tuple[float, float, bool]:
    """Tighten along-wall span to jamb/sill ink; fall back if ambiguous."""
    orig_lo, orig_hi = run_lo, run_hi
    orig_span = orig_hi - orig_lo
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

    threshold = 0.12
    ink_idx = np.where(along >= threshold)[0]
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


def _refine_window_thickness(
    wall_pair_mask: np.ndarray,
    horiz: bool,
    axis: float,
    span_lo: float,
    span_hi: float,
    band_half: int,
) -> tuple[float, float, bool]:
    """Set perpendicular extent from actual wall-pair stroke positions."""
    mid = (span_lo + span_hi) / 2.0
    if horiz:
        x0, x1 = int(round(span_lo)), int(round(span_hi))
        y0, y1 = int(round(axis - band_half)), int(round(axis + band_half))
        region = _crop(wall_pair_mask, (x0, y0, x1, y1))
        if region.size == 0:
            return axis - band_half, axis + band_half, False
        row_ink = (region > 0).mean(axis=1)
        ink_rows = np.where(row_ink >= 0.05)[0]
    else:
        x0, x1 = int(round(axis - band_half)), int(round(axis + band_half))
        y0, y1 = int(round(span_lo)), int(round(span_hi))
        region = _crop(wall_pair_mask, (x0, y0, x1, y1))
        if region.size == 0:
            return axis - band_half, axis + band_half, False
        row_ink = (region > 0).mean(axis=0)
        ink_rows = np.where(row_ink >= 0.05)[0]

    if ink_rows.size == 0:
        return axis - band_half, axis + band_half, False

    pad = max(2, int(0.15 * band_half))
    lo = float(y0 + ink_rows[0] - pad) if horiz else float(x0 + ink_rows[0] - pad)
    hi = float(y0 + ink_rows[-1] + 1 + pad) if horiz else float(x0 + ink_rows[-1] + 1 + pad)
    return lo, hi, True


def _rect_from_thickness(
    horiz: bool,
    span_lo: float,
    span_hi: float,
    perp_lo: float,
    perp_hi: float,
) -> tuple[int, int, int, int]:
    if horiz:
        return (
            int(round(span_lo)), int(round(perp_lo)),
            int(round(span_hi)), int(round(perp_hi)),
        )
    return (
        int(round(perp_lo)), int(round(span_lo)),
        int(round(perp_hi)), int(round(span_hi)),
    )


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
    wall_pair_mask: Optional[np.ndarray] = None,
    ink_mask: Optional[np.ndarray] = None,
    evidence_detail: str = "sill",
    refined: bool = False,
) -> dict:
    perp_lo = axis - band_half
    perp_hi = axis + band_half
    span_refined = False
    thick_refined = False

    if ink_mask is not None:
        rough = _gap_rect(horiz, axis, span_lo, span_hi, band_half)
        span_lo, span_hi, span_refined = _refine_window_span(
            ink_mask, horiz, rough, span_lo, span_hi,
        )

    if wall_pair_mask is not None:
        perp_lo, perp_hi, thick_refined = _refine_window_thickness(
            wall_pair_mask, horiz, axis, span_lo, span_hi, band_half,
        )

    rect = _rect_from_thickness(horiz, span_lo, span_hi, perp_lo, perp_hi)
    gap_px = span_hi - span_lo
    width_units = gap_px / px_per_unit
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    out = {
        "id": "",
        "host_wall_id": host_wall.get("id"),
        "bbox_px": list(rect),
        "center_px": [cx, cy],
        "width": f"{width_units:.2f} {unit_label}",
        "width_raw": round(width_units, 2),
        "is_exterior": True,
        "evidence": "sill",
        "evidence_detail": evidence_detail,
        "refined": span_refined or thick_refined or refined,
        "_horiz": horiz,
        "_axis": axis,
        "_span_lo": span_lo,
        "_span_hi": span_hi,
    }
    return out


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
    """Merge detections that overlap along the same wall axis."""
    merge_gap_px = 1.0 * px_per_unit
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
            gap = w["_span_lo"] - current["_span_hi"]
            if _spans_overlap(
                current["_span_lo"], current["_span_hi"],
                w["_span_lo"], w["_span_hi"],
            ) or gap < merge_gap_px:
                if w["_span_hi"] - w["_span_lo"] > current["_span_hi"] - current["_span_lo"]:
                    detail = w.get("evidence_detail", "sill")
                    current.update(w)
                    current["evidence_detail"] = detail
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
        for k in ("_span_lo", "_span_hi", "_horiz", "_axis"):
            w.pop(k, None)
    return merged


def _passes_width_gate(width_raw: float, crop_mode: bool) -> bool:
    min_ft = WINDOW_MIN_FT_REFINED
    max_ft = WINDOW_MAX_FT_REFINED
    if not crop_mode:
        min_ft = max(WINDOW_MIN_FT, min_ft)
        max_ft = min(WINDOW_MAX_FT, max_ft)
    return min_ft <= width_raw <= max_ft


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
    crop_mode: bool,
    *,
    strategy: str,
    require_open_check: bool,
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

    if not _passes_width_gate(width_raw, crop_mode):
        record["status"] = "rejected"
        record["reject_reason"] = "width_out_of_range"
        return record

    if require_open_check and not _gap_is_open(
        wall_pair_mask, horiz, rect, max_ink_frac=open_gap_max,
    ):
        record["status"] = "rejected"
        record["reject_reason"] = "ink_not_open"
        return record

    if not _gap_has_bilateral_break(
        wall_pair_mask, horiz, rect, max_ink_frac=open_gap_max,
    ):
        record["status"] = "rejected"
        record["reject_reason"] = "no_bilateral_break"
        return record

    has_sill, evidence_detail, sill_cover = gap_sill_evidence(
        ink_mask, horiz, rect, wall_pair_mask=wall_pair_mask,
    )
    if not has_sill:
        record["status"] = "rejected"
        record["reject_reason"] = "no_sill"
        return record

    if (
        _looks_like_dimension_line(
            ink_mask, horiz, rect, wall_pair_mask=wall_pair_mask,
        )
        and sill_cover < 0.60
    ):
        record["status"] = "rejected"
        record["reject_reason"] = "dimension_line"
        return record

    cand = _make_window(
        horiz, axis, run_lo, run_hi, band_half,
        host_wall, px_per_unit, unit_label,
        wall_pair_mask=wall_pair_mask,
        ink_mask=ink_mask,
        evidence_detail=evidence_detail,
    )
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
    open_gap_max = 0.20 if crop_mode else 0.12
    band_half = max(4, int(math.ceil(0.75 * px_per_unit)))

    door_centers = [d["center_px"] for d in (doors or []) if d.get("center_px")]
    door_dedup_dist = max(6.0, 0.5 * px_per_unit)

    candidates: list[dict] = []

    for wall in walls:
        if not wall.get("is_exterior"):
            continue
        coords = wall.get("px_coords")
        if not coords or len(coords) < 4:
            continue
        horiz = _orient(coords)
        if horiz is None:
            continue
        x1, y1, x2, y2 = coords
        if horiz:
            axis = (y1 + y2) / 2.0
            span_lo, span_hi = min(x1, x2), max(x1, x2)
        else:
            axis = (x1 + x2) / 2.0
            span_lo, span_hi = min(y1, y2), max(y1, y2)

        inset = _end_inset_px(band_half, px_per_unit, span_hi - span_lo)
        for run_lo, run_hi in _open_runs_along_wall(
            wall_pair_mask, horiz, axis, span_lo, span_hi,
            band_half, min_gap_px, max_gap_px, open_gap_max, inset, px_per_unit,
        ):
            candidates.append(_evaluate_candidate(
                horiz, axis, run_lo, run_hi, band_half, wall,
                wall_pair_mask, ink_mask, px_per_unit, unit_label,
                open_gap_max, door_centers, door_dedup_dist, crop_mode,
                strategy="in_segment",
                require_open_check=False,
            ))

    groups: dict[bool, list[tuple[float, float, float, dict]]] = {True: [], False: []}
    for w in walls:
        if not w.get("is_exterior"):
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
                if not (wall_a.get("is_exterior") and wall_b.get("is_exterior")):
                    continue
                ca, cb = wall_a.get("px_coords", []), wall_b.get("px_coords", [])
                len_a = math.hypot(ca[2] - ca[0], ca[3] - ca[1]) if len(ca) >= 4 else 0
                len_b = math.hypot(cb[2] - cb[0], cb[3] - cb[1]) if len(cb) >= 4 else 0
                host = wall_a if len_a >= len_b else wall_b
                candidates.append(_evaluate_candidate(
                    horiz, axis, span_lo, span_hi, band_half, host,
                    wall_pair_mask, ink_mask, px_per_unit, unit_label,
                    open_gap_max, door_centers, door_dedup_dist, crop_mode,
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

    dedup_dist = max(6.0, 0.5 * px_per_unit)
    final: list[dict] = []
    for w in sorted(kept, key=lambda w: (w["center_px"][0], w["center_px"][1])):
        if _near_any(w["center_px"], [k["center_px"] for k in final], dedup_dist):
            continue
        final.append(w)

    for i, w in enumerate(final):
        w["id"] = f"win{i + 1}"
        for k in ("evidence_detail", "refined", "_horiz", "_axis", "_span_lo", "_span_hi"):
            w.pop(k, None)
    return final
