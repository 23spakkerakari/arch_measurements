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
# Fraction of gap columns allowed to contain wall-pair ink before the gap is
# considered a dedup/cleanup split rather than a real opening.
OPEN_GAP_MAX_INK_FRAC = 0.15
# A raw-ink row/column covering at least this fraction of the gap span,
# parallel to the wall axis, is a window sill.
SILL_COVER_FRAC = 0.60


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
    return float(profile.sum()) / n <= OPEN_GAP_MAX_INK_FRAC


def _gap_has_sill(
    ink_mask: Optional[np.ndarray],
    horiz: bool,
    rect: tuple[int, int, int, int],
    inset_px: int = 2,
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
    # For a horizontal wall the sill is a near-full row; for vertical, a column.
    cover = (region > 0).mean(axis=1 if horiz else 0)
    return bool(cover.size) and float(cover.max()) >= SILL_COVER_FRAC


def detect_doors(
    walls: list[dict],
    wall_pair_mask: np.ndarray,
    ink_mask: Optional[np.ndarray],
    px_per_unit: float,
    unit_label: str = "ft",
    axis_tol_px: Optional[int] = None,
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

    to_ft = 1.0 if unit_label == "ft" else 3.2808
    min_gap_px = DOOR_MIN_FT / to_ft * px_per_unit
    max_gap_px = DOOR_MAX_FT / to_ft * px_per_unit
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
            if clusters and abs(e[0] - clusters[-1][0][0]) <= axis_tol_px:
                clusters[-1].append(e)
            else:
                clusters.append([e])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            cluster.sort(key=lambda e: e[1])
            for a, b in zip(cluster, cluster[1:]):
                gap_px = b[1] - a[2]
                if not (min_gap_px <= gap_px <= max_gap_px):
                    continue
                axis = (a[0] + b[0]) / 2.0
                rect = _gap_rect(horiz, axis, a[2], b[1], band_half)
                if not _gap_is_open(wall_pair_mask, horiz, rect):
                    continue
                if _gap_has_sill(ink_mask, horiz, rect):
                    continue

                wall_a, wall_b = a[3], b[3]
                len_a = abs(a[2] - a[1])
                len_b = abs(b[2] - b[1])
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
