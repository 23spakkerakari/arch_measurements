"""Geometric door detection from collinear wall gaps.

The wall extractor deliberately never bridges two collinear endpoints facing
each other across a gap (see ``snap_wall_endpoints``): that gap *is* the
doorway. This module turns those gaps into ``doors[]``:

1. Group axis-aligned walls by orientation and axis position.
2. For each adjacent collinear pair, the span gap is a door candidate when
   its width is in the standard door range (1.5–5 ft incl. trim).
3. Open-gap verification: the gap region of ``wall_pair_mask`` must be
   essentially ink-free — a dedup/cleanup split still has wall ink there,
   a true doorway does not.
4. Sill discriminator: a thin stroke in the raw ink running parallel to the
   wall axis across most of the gap is a window sill, not a door.

All coordinates are in the same (cropped) frame as the input walls;
``analyze_page`` shifts results by the ROI offset alongside the walls.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

# Standard door widths are 2–4 ft; allow trim/jamb slack on both sides.
DOOR_MIN_FT = 1.5
DOOR_MAX_FT = 5.0
# Cropped real plans: walls often stop short of the frame after snap/sub-segment.
DOOR_MIN_FT_CROP = 1.25
DOOR_MAX_FT_CROP = 5.5
# Fraction of gap columns allowed to contain wall-pair ink before the gap is
# considered a dedup/cleanup split rather than a real opening.
OPEN_GAP_MAX_INK_FRAC = 0.15
# A raw-ink row/column covering at least this fraction of the gap span,
# parallel to the wall axis, is a window sill.
SILL_COVER_FRAC = 0.60
SILL_COVER_FRAC_PARTIAL = 0.40
SILL_STRONG_GAP_FRAC = 0.08


def ink_mask_from_image(image: np.ndarray) -> np.ndarray:
    """Binary ink mask (255 = ink) from an RGB/grayscale plan image."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _orient(coords: list) -> Optional[bool]:
    """True = horizontal, False = vertical, None = diagonal."""
    x1, y1, x2, y2 = coords
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    if dy <= max(2, 0.05 * dx):
        return True
    if dx <= max(2, 0.05 * dy):
        return False
    return None


def _gap_rect(
    horiz: bool, axis: float, span_lo: float, span_hi: float, band_half: int,
) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the gap region between two collinear walls."""
    if horiz:
        return (
            int(round(span_lo)), int(round(axis - band_half)),
            int(round(span_hi)), int(round(axis + band_half)),
        )
    return (
        int(round(axis - band_half)), int(round(span_lo)),
        int(round(axis + band_half)), int(round(span_hi)),
    )


def _crop(mask: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    h, w = mask.shape[:2]
    x0 = max(0, min(w, rect[0]))
    y0 = max(0, min(h, rect[1]))
    x1 = max(0, min(w, rect[2]))
    y1 = max(0, min(h, rect[3]))
    return mask[y0:y1, x0:x1]


def _gap_is_open(
    wall_pair_mask: np.ndarray,
    horiz: bool,
    rect: tuple[int, int, int, int],
    inset_px: int = 2,
    max_ink_frac: float = OPEN_GAP_MAX_INK_FRAC,
) -> bool:
    """True when the gap band contains essentially no wall-pair ink.

    The span is inset slightly on both ends so wall endpoint pixels (which sit
    exactly at the gap boundary) don't count against an otherwise open gap.
    """
    x0, y0, x1, y1 = rect
    if horiz:
        x0, x1 = x0 + inset_px, x1 - inset_px
    else:
        y0, y1 = y0 + inset_px, y1 - inset_px
    region = _crop(wall_pair_mask, (x0, y0, x1, y1))
    if region.size == 0:
        return False
    # Fraction of along-axis positions containing any ink.
    profile = (region > 0).any(axis=0 if horiz else 1)
    n = profile.size
    if n == 0:
        return False
    return float(profile.sum()) / n <= max_ink_frac


def _gap_mean_ink_frac(
    wall_pair_mask: np.ndarray,
    horiz: bool,
    rect: tuple[int, int, int, int],
    inset_px: int = 2,
) -> float:
    """Mean fraction of along-axis positions with wall-pair ink in the gap band."""
    x0, y0, x1, y1 = rect
    if horiz:
        x0, x1 = x0 + inset_px, x1 - inset_px
    else:
        y0, y1 = y0 + inset_px, y1 - inset_px
    region = _crop(wall_pair_mask, (x0, y0, x1, y1))
    if region.size == 0:
        return 1.0
    profile = (region > 0).any(axis=0 if horiz else 1)
    if profile.size == 0:
        return 1.0
    return float(profile.sum()) / profile.size


def _sill_cover_profile(
    ink_mask: np.ndarray,
    horiz: bool,
    rect: tuple[int, int, int, int],
    inset_px: int = 2,
) -> tuple[float, str]:
    """Return (max_cover, evidence_detail) for sill ink parallel to the wall."""
    x0, y0, x1, y1 = rect
    if horiz:
        x0, x1 = x0 + inset_px, x1 - inset_px
    else:
        y0, y1 = y0 + inset_px, y1 - inset_px
    region = _crop(ink_mask, (x0, y0, x1, y1))
    if region.size == 0:
        return 0.0, "none"

    cover = (region > 0).mean(axis=1 if horiz else 0)
    if cover.size == 0:
        return 0.0, "none"

    max_cover = float(cover.max())
    if max_cover >= SILL_COVER_FRAC:
        return max_cover, "sill"

    # Partial sill when the wall-pair gap is very open.
    gap_frac = _gap_mean_ink_frac(
        np.zeros_like(ink_mask), horiz, rect, inset_px=inset_px,
    )
    # gap_frac above uses empty mask — recompute from caller context in has_sill.

    # Triple-line: two moderate rows plus one stronger row (CAD window symbol).
    strong_rows = int(np.sum(cover >= SILL_COVER_FRAC_PARTIAL))
    if strong_rows >= 2 and max_cover >= SILL_COVER_FRAC_PARTIAL:
        return max_cover, "sill+triple"

    if max_cover >= SILL_COVER_FRAC_PARTIAL:
        return max_cover, "sill+partial"

    return max_cover, "none"


def _gap_has_sill(
    ink_mask: Optional[np.ndarray],
    horiz: bool,
    rect: tuple[int, int, int, int],
    inset_px: int = 2,
    *,
    wall_pair_mask: Optional[np.ndarray] = None,
    gap_open_frac: Optional[float] = None,
) -> bool:
    """True when a stroke parallel to the wall axis spans most of the gap."""
    if ink_mask is None:
        return False
    x0, y0, x1, y1 = rect
    if horiz:
        x0, x1 = x0 + inset_px, x1 - inset_px
    else:
        y0, y1 = y0 + inset_px, y1 - inset_px
    region = _crop(ink_mask, (x0, y0, x1, y1))
    if region.size == 0:
        return False

    cover = (region > 0).mean(axis=1 if horiz else 0)
    if cover.size == 0:
        return False

    max_cover = float(cover.max())
    if max_cover >= SILL_COVER_FRAC:
        return True

    if gap_open_frac is None and wall_pair_mask is not None:
        gap_open_frac = _gap_mean_ink_frac(wall_pair_mask, horiz, rect, inset_px)

    # Accept partial sill when the wall-pair band is clearly open.
    if (
        max_cover >= SILL_COVER_FRAC_PARTIAL
        and gap_open_frac is not None
        and gap_open_frac <= SILL_STRONG_GAP_FRAC
    ):
        return True

    # Triple-line window symbol: multiple parallel strokes in the gap band.
    strong_rows = int(np.sum(cover >= SILL_COVER_FRAC_PARTIAL))
    if strong_rows >= 2 and max_cover >= SILL_COVER_FRAC_PARTIAL:
        return True

    return False


def gap_sill_evidence(
    ink_mask: Optional[np.ndarray],
    horiz: bool,
    rect: tuple[int, int, int, int],
    *,
    wall_pair_mask: Optional[np.ndarray] = None,
) -> tuple[bool, str, float]:
    """Return (has_sill, evidence_detail, max_cover) for debug / window emit."""
    if ink_mask is None:
        return False, "none", 0.0
    gap_open = (
        _gap_mean_ink_frac(wall_pair_mask, horiz, rect)
        if wall_pair_mask is not None
        else None
    )
    has = _gap_has_sill(
        ink_mask, horiz, rect,
        wall_pair_mask=wall_pair_mask,
        gap_open_frac=gap_open,
    )
    max_cover, detail = _sill_cover_profile(ink_mask, horiz, rect)
    if has and detail == "none":
        detail = "sill+partial" if max_cover >= SILL_COVER_FRAC_PARTIAL else "sill"
    return has, detail, max_cover


def _gap_has_bilateral_break(
    wall_pair_mask: np.ndarray,
    horiz: bool,
    rect: tuple[int, int, int, int],
    inset_px: int = 2,
    max_ink_frac: float = OPEN_GAP_MAX_INK_FRAC,
) -> bool:
    """True when both wall-pair strokes show an opening across the gap span."""
    x0, y0, x1, y1 = rect
    if horiz:
        x0, x1 = x0 + inset_px, x1 - inset_px
    else:
        y0, y1 = y0 + inset_px, y1 - inset_px
    region = _crop(wall_pair_mask, (x0, y0, x1, y1))
    if region.size == 0:
        return False

    min_open_frac = max(0.35, 1.0 - max_ink_frac)

    if horiz:
        h = region.shape[0]
        if h < 3:
            return _gap_is_open(wall_pair_mask, horiz, rect, inset_px, max_ink_frac)
        q = max(1, h // 4)
        band_top = region[:q, :]
        band_bot = region[-q:, :]
        prof_top = (band_top > 0).any(axis=0)
        prof_bot = (band_bot > 0).any(axis=0)
    else:
        w = region.shape[1]
        if w < 3:
            return _gap_is_open(wall_pair_mask, horiz, rect, inset_px, max_ink_frac)
        q = max(1, w // 4)
        band_top = region[:, :q]
        band_bot = region[:, -q:]
        prof_top = (band_top > 0).any(axis=1)
        prof_bot = (band_bot > 0).any(axis=1)

    n = prof_top.size
    if n == 0:
        return False
    open_top = float((~prof_top).sum()) / n
    open_bot = float((~prof_bot).sum()) / n
    return open_top >= min_open_frac and open_bot >= min_open_frac


def _looks_like_dimension_line(
    ink_mask: np.ndarray,
    horiz: bool,
    rect: tuple[int, int, int, int],
    inset_px: int = 2,
    *,
    wall_pair_mask: Optional[np.ndarray] = None,
) -> bool:
    """True when ink in the gap looks like a dimension string, not a window."""
    x0, y0, x1, y1 = rect
    if horiz:
        x0, x1 = x0 + inset_px, x1 - inset_px
    else:
        y0, y1 = y0 + inset_px, y1 - inset_px
    region = _crop(ink_mask, (x0, y0, x1, y1))
    if region.size == 0:
        return False

    if horiz:
        row_cover = (region > 0).mean(axis=1)
        along = (region > 0).any(axis=0)
        peak_rows = int(np.sum(row_cover > 0.05))
    else:
        row_cover = (region > 0).mean(axis=0)
        along = (region > 0).any(axis=1)
        peak_rows = int(np.sum(row_cover > 0.05))

    if along.size < 4:
        return False

    max_along = float(along.mean())
    edge = max(2, along.size // 8)
    edge_cover = float(np.maximum(along[:edge], along[-edge:]).mean()) if along.size >= edge * 2 else 0.0
    center_cover = float(along[edge:-edge].mean()) if along.size > edge * 2 else float(along.mean())
    max_row_cover = float(row_cover.max()) if row_cover.size else 0.0

    # Single-row stroke spanning most of the gap — dimension unless the
    # wall-pair band is clearly open (thin window sill in a real opening).
    if peak_rows <= 1 and max_row_cover >= 0.85:
        if wall_pair_mask is not None:
            gap_open = _gap_mean_ink_frac(wall_pair_mask, horiz, rect, inset_px)
            if gap_open <= SILL_STRONG_GAP_FRAC:
                return False
        return True

    # Dimension: thin parallel line, high center cover, no side jambs.
    if (
        peak_rows <= 2
        and max_along >= 0.55
        and edge_cover < 0.20
        and center_cover >= 0.35
    ):
        return True
    return False


def _merged_interval_gaps(
    cluster: list[tuple[float, float, float, dict]],
    min_gap_px: float,
    max_gap_px: float,
    touch_tol_px: int = 3,
) -> list[tuple[float, float, float, dict, dict]]:
    """Gaps between merged collinear spans (handles fragmented sub-segments)."""
    if len(cluster) < 2:
        return []
    intervals = sorted(
        ((e[1], e[2], e[3]) for e in cluster),
        key=lambda t: (t[0], t[1]),
    )
    merged: list[tuple[float, float, dict]] = []
    for lo, hi, wall in intervals:
        if merged and lo <= merged[-1][1] + touch_tol_px:
            prev_lo, prev_hi, prev_wall = merged[-1]
            if (hi - lo) >= (prev_hi - prev_lo):
                host = wall
            else:
                host = prev_wall
            merged[-1] = (prev_lo, max(prev_hi, hi), host)
        else:
            merged.append((lo, hi, wall))

    axis = sum(e[0] for e in cluster) / len(cluster)
    gaps: list[tuple[float, float, float, dict, dict]] = []
    for (a_lo, a_hi, wall_a), (b_lo, b_hi, wall_b) in zip(merged, merged[1:]):
        gap_px = b_lo - a_hi
        if min_gap_px <= gap_px <= max_gap_px:
            gaps.append((axis, a_hi, b_lo, wall_a, wall_b))
    return gaps


def detect_doors(
    walls: list[dict],
    wall_pair_mask: np.ndarray,
    ink_mask: Optional[np.ndarray],
    px_per_unit: float,
    unit_label: str = "ft",
    axis_tol_px: Optional[int] = None,
    crop_mode: bool = False,
) -> list[dict]:
    """Detect doorway gaps between collinear axis-aligned walls.

    Args:
        walls: measured wall dicts (``px_coords`` in crop coordinates).
        wall_pair_mask: full-resolution double-stroke wall mask (crop frame).
        ink_mask: raw binary ink (255 = ink) for the sill discriminator, or
            None to skip it.
        px_per_unit: pixels per real-world unit (ft or m).
        unit_label: "ft" or "m"; sets the door width range.
        axis_tol_px: perpendicular tolerance for treating walls as collinear
            (defaults to the dedup tolerance, ~8 real inches).
    """
    if axis_tol_px is None:
        axis_tol_px = max(12, int(0.6 * px_per_unit))
    # Wider axis clustering for doors — fragmented sub-segments can sit on
    # slightly different centerlines after room split / snap.
    door_axis_tol = max(axis_tol_px, int(1.0 * px_per_unit))

    to_ft = 1.0 if unit_label == "ft" else 3.2808
    min_ft = DOOR_MIN_FT_CROP if crop_mode else DOOR_MIN_FT
    max_ft = DOOR_MAX_FT_CROP if crop_mode else DOOR_MAX_FT
    min_gap_px = min_ft / to_ft * px_per_unit
    max_gap_px = max_ft / to_ft * px_per_unit
    open_gap_max = 0.25 if crop_mode else OPEN_GAP_MAX_INK_FRAC
    # Wall-pair strokes sit up to ~0.75 units off the centerline (see
    # wall_pair_gap_range); the band must cover both faces.
    band_half = max(4, int(math.ceil(0.75 * px_per_unit)))

    # entry: (axis, span_lo, span_hi, wall)
    groups: dict[bool, list[tuple[float, float, float, dict]]] = {True: [], False: []}
    for w in walls:
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

    doors: list[dict] = []
    for horiz, entries in groups.items():
        if len(entries) < 2:
            continue
        # Cluster by axis position: sort, then split where the gap to the
        # cluster's anchor axis exceeds the tolerance (anchor-linkage prevents
        # chaining distinct parallel walls together).
        entries.sort(key=lambda e: e[0])
        clusters: list[list[tuple[float, float, float, dict]]] = []
        for e in entries:
            if clusters and abs(e[0] - clusters[-1][0][0]) <= door_axis_tol:
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
                rect = _gap_rect(horiz, axis, span_lo, span_hi, band_half)
                if not _gap_is_open(
                    wall_pair_mask, horiz, rect,
                    max_ink_frac=open_gap_max,
                ):
                    continue
                if _gap_has_sill(
                    ink_mask, horiz, rect,
                    wall_pair_mask=wall_pair_mask,
                ):
                    continue

                gap_px = span_hi - span_lo
                ca, cb = wall_a.get("px_coords", []), wall_b.get("px_coords", [])
                len_a = math.hypot(ca[2] - ca[0], ca[3] - ca[1]) if len(ca) >= 4 else 0
                len_b = math.hypot(cb[2] - cb[0], cb[3] - cb[1]) if len(cb) >= 4 else 0
                host = wall_a if len_a >= len_b else wall_b
                width_units = gap_px / px_per_unit
                cx = (rect[0] + rect[2]) / 2.0
                cy = (rect[1] + rect[3]) / 2.0
                doors.append({
                    "id": "",  # assigned after dedup
                    "host_wall_id": host.get("id"),
                    "bbox_px": list(rect),
                    "center_px": [cx, cy],
                    "width": f"{width_units:.2f} {unit_label}",
                    "width_raw": round(width_units, 2),
                    "is_exterior": bool(
                        wall_a.get("is_exterior") and wall_b.get("is_exterior")
                    ),
                    "evidence": "gap",
                })

    # Dedup: overlapping axis clusters can yield the same gap twice.
    dedup_dist = max(6.0, 0.5 * px_per_unit)
    kept: list[dict] = []
    for d in sorted(doors, key=lambda d: (d["center_px"][0], d["center_px"][1])):
        cx, cy = d["center_px"]
        if any(
            math.hypot(cx - k["center_px"][0], cy - k["center_px"][1]) < dedup_dist
            for k in kept
        ):
            continue
        kept.append(d)

    for i, d in enumerate(kept):
        d["id"] = f"d{i + 1}"
    return kept
