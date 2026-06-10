"""Synthetic floor-plan renderer with exact ground truth.

Draws plans the way the CV pipeline expects them (double-stroke walls, white
sheet, annotation ink) so true accuracy can be measured without manual
annotation. All geometry is parameterized off ``px_per_unit`` so the rendered
ink and the returned ground truth stay in lockstep.

Conventions (matching the pipeline's drawing assumptions):
- Walls are two parallel 3 px strokes separated by the wall thickness.
- Exterior wall thickness: 1.0 ft. Interior partitions: 0.5 ft.
- The building sits inside the sheet's "safe zone" (outside the hard-coded
  title-block/margin exclusion fractions in ``preprocess``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

STROKE = 3  # px ink stroke width

# 1/8" = 1'-0" at 144 DPI -> 18 px per foot
DEFAULT_SCALE_STR = "1/8in=1ft"
DEFAULT_DPI = 144
DEFAULT_PX_PER_FT = 18.0


@dataclass
class SyntheticPlan:
    name: str
    image: np.ndarray  # RGB uint8, white sheet with black ink
    scale_str: str
    dpi: int
    px_per_unit: float
    ground_truth: dict = field(default_factory=dict)

    @property
    def image_size_px(self) -> list[int]:
        h, w = self.image.shape[:2]
        return [w, h]


def _new_sheet(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _line(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> None:
    cv2.line(img, p1, p2, (0, 0, 0), STROKE)


def _polyline(img: np.ndarray, pts: list[tuple[int, int]], closed: bool = True) -> None:
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], closed, (0, 0, 0), STROKE)


def _erase(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    img[y0:y1, x0:x1] = 255


def _dimension_string(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    text: str,
) -> None:
    """Single-stroke dimension line with end ticks and a text label."""
    _line(img, p1, p2)
    horiz = abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1])
    tick = 12
    for px, py in (p1, p2):
        if horiz:
            _line(img, (px, py - tick), (px, py + tick))
        else:
            _line(img, (px - tick, py), (px + tick, py))
    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    if horiz:
        org = (mx - 50, my + 55)
    else:
        org = (mx + 20, my)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)


def render_two_room_plan(
    px_per_unit: float = DEFAULT_PX_PER_FT,
    scale_str: str = DEFAULT_SCALE_STR,
    dpi: int = DEFAULT_DPI,
) -> SyntheticPlan:
    """Rectangular 60 x 40 ft building, one vertical partition, door + windows.

    Layout (px at the default 18 px/ft):
      sheet 2200 x 1500, building outer box (180, 220)-(1260, 940),
      exterior wall thickness 18 px (1 ft), partition centered at x=780
      with a 2 ft door gap, two 4 ft windows in the north wall, and two
      single-stroke dimension strings outside the building.

    The building sits inside the sheet safe zone (clear of the hard-coded
    margin/title-block exclusion fractions in ``preprocess``); the dimension
    strings are intentionally close to the walls — annotation ink the
    pipeline must reject.
    """
    if px_per_unit != DEFAULT_PX_PER_FT:
        raise ValueError("render_two_room_plan currently supports px_per_unit=18 only")

    sheet_w, sheet_h = 2200, 1500
    img = _new_sheet(sheet_w, sheet_h)

    g = int(round(1.0 * px_per_unit))         # exterior wall thickness: 18 px
    pg = int(round(0.5 * px_per_unit))        # partition thickness: 9 px
    x0, y0, x1, y1 = 180, 220, 1260, 940      # outer faces (60 ft x 40 ft)
    ix0, iy0, ix1, iy1 = x0 + g, y0 + g, x1 - g, y1 - g  # inner faces

    # Exterior: outer + inner rectangle strokes (double-line wall)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), STROKE)
    cv2.rectangle(img, (ix0, iy0), (ix1, iy1), (0, 0, 0), STROKE)

    # Interior partition: two vertical strokes centered on xp, inner face to inner face
    xp = 780
    px_a = xp - pg // 2  # 776
    px_b = px_a + pg     # 785
    _line(img, (px_a, iy0), (px_a, iy1))
    _line(img, (px_b, iy0), (px_b, iy1))

    # Door: 2 ft gap in the partition
    door_h = int(round(2.0 * px_per_unit))  # 36 px
    door_y0 = 600
    _erase(img, px_a - STROKE, door_y0, px_b + STROKE + 1, door_y0 + door_h)

    # Windows: two 4 ft gaps in the north wall (both strokes), thin sill line
    win_w = int(round(4.0 * px_per_unit))  # 72 px
    windows = []
    for wi, wx0 in enumerate((360, 980), start=1):
        wx1 = wx0 + win_w
        _erase(img, wx0, y0 - STROKE, wx1, iy0 + STROKE + 1)
        sill_y = y0 + g // 2
        cv2.line(img, (wx0, sill_y), (wx1, sill_y), (0, 0, 0), 1)
        windows.append({
            "id": f"win{wi}",
            "host_wall_id": "ext_n",
            "bbox_px": [wx0, y0, wx1, iy0],
            "center_px": [(wx0 + wx1) / 2.0, (y0 + iy0) / 2.0],
        })

    # Dimension strings (single-stroke annotation ink the pipeline must reject)
    _dimension_string(img, (x0, 1040), (x1, 1040), "60'-0\"")
    _dimension_string(img, (140, y0), (140, y1), "40'-0\"")

    half = g / 2.0
    gt = {
        "id": "synth_two_room",
        "scale": scale_str,
        "image_size_px": [sheet_w, sheet_h],
        "rooms": [
            {"id": "R1", "bbox_px": [ix0, iy0, px_a, iy1],
             "area_raw": round((px_a - ix0) * (iy1 - iy0) / px_per_unit ** 2, 1)},
            {"id": "R2", "bbox_px": [px_b, iy0, ix1, iy1],
             "area_raw": round((ix1 - px_b) * (iy1 - iy0) / px_per_unit ** 2, 1)},
        ],
        "walls": [
            {"id": "ext_n", "px_coords": [x0, y0 + half, x1, y0 + half],
             "facing": "North", "is_exterior": True,
             "length_raw": round((x1 - x0) / px_per_unit, 2)},
            {"id": "ext_s", "px_coords": [x0, y1 - half, x1, y1 - half],
             "facing": "South", "is_exterior": True,
             "length_raw": round((x1 - x0) / px_per_unit, 2)},
            {"id": "ext_w", "px_coords": [x0 + half, y0, x0 + half, y1],
             "facing": "West", "is_exterior": True,
             "length_raw": round((y1 - y0) / px_per_unit, 2)},
            {"id": "ext_e", "px_coords": [x1 - half, y0, x1 - half, y1],
             "facing": "East", "is_exterior": True,
             "length_raw": round((y1 - y0) / px_per_unit, 2)},
            {"id": "int_1", "px_coords": [xp, iy0, xp, iy1],
             "facing": "West", "is_exterior": False,
             "length_raw": round((iy1 - iy0) / px_per_unit, 2)},
        ],
        "doors": [
            {"id": "d1", "host_wall_id": "int_1",
             "bbox_px": [px_a - STROKE, door_y0, px_b + STROKE, door_y0 + door_h],
             "center_px": [xp, door_y0 + door_h / 2.0]},
        ],
        "windows": windows,
        "labels": [],
        "dimensions": [
            {"id": "dim1", "text": "60'-0\"", "value_raw": 60.0, "unit": "ft",
             "center_px": [(x0 + x1) / 2.0, 1040.0]},
            {"id": "dim2", "text": "40'-0\"", "value_raw": 40.0, "unit": "ft",
             "center_px": [140.0, (y0 + y1) / 2.0]},
        ],
    }

    return SyntheticPlan(
        name="synth_two_room", image=img, scale_str=scale_str, dpi=dpi,
        px_per_unit=px_per_unit, ground_truth=gt,
    )


def render_l_shape_plan(
    px_per_unit: float = DEFAULT_PX_PER_FT,
    scale_str: str = DEFAULT_SCALE_STR,
    dpi: int = DEFAULT_DPI,
) -> SyntheticPlan:
    """L-shaped 60 x 50 ft building (25 x 20 ft notch), horizontal partition.

    Outer boundary (px): (160,220) -> (790,220) -> (790,580) -> (1240,580)
    -> (1240,1120) -> (160,1120). Notch removed from the top-right corner.
    Sized to stay clear of the sheet-fraction exclusion zones (in particular
    the bottom-right title-block zone at x > 58%, y > 50%), which blank ink
    before footprint detection.
    """
    if px_per_unit != DEFAULT_PX_PER_FT:
        raise ValueError("render_l_shape_plan currently supports px_per_unit=18 only")

    sheet_w, sheet_h = 2400, 1600
    img = _new_sheet(sheet_w, sheet_h)

    g = int(round(1.0 * px_per_unit))   # 18 px
    pg = int(round(0.5 * px_per_unit))  # 9 px

    outer = [(160, 220), (790, 220), (790, 580), (1240, 580), (1240, 1120), (160, 1120)]
    inner = [(178, 238), (772, 238), (772, 598), (1222, 598), (1222, 1102), (178, 1102)]
    _polyline(img, outer, closed=True)
    _polyline(img, inner, closed=True)

    # Horizontal partition at y=780 (centerline), inner west to inner east face
    yp = 780
    py_a = yp - pg // 2  # 776
    py_b = py_a + pg     # 785
    _line(img, (178, py_a), (1222, py_a))
    _line(img, (178, py_b), (1222, py_b))

    # Door: 2 ft gap in the partition
    door_w = int(round(2.0 * px_per_unit))  # 36 px
    door_x0 = 700
    _erase(img, door_x0, py_a - STROKE, door_x0 + door_w, py_b + STROKE + 1)

    # Dimension string along the south side
    _dimension_string(img, (160, 1240), (1240, 1240), "60'-0\"")

    half = g / 2.0
    gt = {
        "id": "synth_l_shape",
        "scale": scale_str,
        "image_size_px": [sheet_w, sheet_h],
        "rooms": [
            # Upper room is L-shaped; bbox covers its extent
            {"id": "R1", "bbox_px": [178, 238, 1222, py_a]},
            {"id": "R2", "bbox_px": [178, py_b, 1222, 1102]},
        ],
        "walls": [
            {"id": "ext_top", "px_coords": [160, 220 + half, 790, 220 + half],
             "facing": "North", "is_exterior": True, "length_raw": 35.0},
            {"id": "ext_notch_v", "px_coords": [790 - half, 220, 790 - half, 580],
             "facing": "East", "is_exterior": True, "length_raw": 20.0},
            {"id": "ext_notch_h", "px_coords": [790, 580 + half, 1240, 580 + half],
             "facing": "North", "is_exterior": True, "length_raw": 25.0},
            {"id": "ext_e", "px_coords": [1240 - half, 580, 1240 - half, 1120],
             "facing": "East", "is_exterior": True, "length_raw": 30.0},
            {"id": "ext_s", "px_coords": [160, 1120 - half, 1240, 1120 - half],
             "facing": "South", "is_exterior": True, "length_raw": 60.0},
            {"id": "ext_w", "px_coords": [160 + half, 220, 160 + half, 1120],
             "facing": "West", "is_exterior": True, "length_raw": 50.0},
            {"id": "int_1", "px_coords": [178, yp, 1222, yp],
             "facing": "North", "is_exterior": False,
             "length_raw": round((1222 - 178) / px_per_unit, 2)},
        ],
        "doors": [
            {"id": "d1", "host_wall_id": "int_1",
             "bbox_px": [door_x0, py_a - STROKE, door_x0 + door_w, py_b + STROKE],
             "center_px": [door_x0 + door_w / 2.0, yp]},
        ],
        "windows": [],
        "labels": [],
        "dimensions": [
            {"id": "dim1", "text": "60'-0\"", "value_raw": 60.0, "unit": "ft",
             "center_px": [700.0, 1240.0]},
        ],
    }

    return SyntheticPlan(
        name="synth_l_shape", image=img, scale_str=scale_str, dpi=dpi,
        px_per_unit=px_per_unit, ground_truth=gt,
    )


ALL_PLANS = {
    "synth_two_room": render_two_room_plan,
    "synth_l_shape": render_l_shape_plan,
}
