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
            # Upper room is L-shaped: bbox for matching, exact polygon for
            # boundary-closure sampling (a bbox perimeter would run through
            # the notch where no wall exists).
            {"id": "R1", "bbox_px": [178, 238, 1222, py_a],
             "polygon_px": [[178, 238], [772, 238], [772, 598], [1222, 598],
                            [1222, py_a], [178, py_a]]},
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


def render_corridor_plan(
    px_per_unit: float = DEFAULT_PX_PER_FT,
    scale_str: str = DEFAULT_SCALE_STR,
    dpi: int = DEFAULT_DPI,
) -> SyntheticPlan:
    """60 x 40 ft building with a 4 ft central corridor and a 24 ft² closet.

    Layout (px at 18 px/ft): outer box (180, 220)-(1260, 940). Two horizontal
    partitions (centerlines y=560 and y=641) bound a 4 ft corridor spanning
    the full inner width; doors connect it to the rooms above and below. A
    3 x 8 ft closet (24 ft² — below the 25 ft² compact-room floor, aspect
    2.7; any narrower and the 2.5 ft doorway close would absorb it) sits in
    the bottom-left corner, walled by a vertical partition and a horizontal
    stub, with a door to the bottom room.

    Exercises the interior-coverage recovery path: the corridor must survive
    the doorway close, and the closet must pass via the corridor area floor.
    """
    if px_per_unit != DEFAULT_PX_PER_FT:
        raise ValueError("render_corridor_plan currently supports px_per_unit=18 only")

    sheet_w, sheet_h = 2200, 1500
    img = _new_sheet(sheet_w, sheet_h)

    g = int(round(1.0 * px_per_unit))         # exterior wall thickness: 18 px
    pg = int(round(0.5 * px_per_unit))        # partition thickness: 9 px
    x0, y0, x1, y1 = 180, 220, 1260, 940      # outer faces (60 ft x 40 ft)
    ix0, iy0, ix1, iy1 = x0 + g, y0 + g, x1 - g, y1 - g  # inner faces

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), STROKE)
    cv2.rectangle(img, (ix0, iy0), (ix1, iy1), (0, 0, 0), STROKE)

    # Corridor partitions A (top) and B (bottom): 4 ft clear between them
    yp_a, yp_b = 560, 641
    pa_a = yp_a - pg // 2  # 556
    pa_b = pa_a + pg       # 565
    pb_a = yp_b - pg // 2  # 637
    pb_b = pb_a + pg       # 646
    for py in (pa_a, pa_b, pb_a, pb_b):
        _line(img, (ix0, py), (ix1, py))

    # Doors: 2 ft gaps connecting corridor to the rooms above and below
    door_w = int(round(2.0 * px_per_unit))  # 36 px
    da_x0, db_x0 = 400, 900
    _erase(img, da_x0, pa_a - STROKE, da_x0 + door_w, pa_b + STROKE + 1)
    _erase(img, db_x0, pb_a - STROKE, db_x0 + door_w, pb_b + STROKE + 1)

    # Closet: 3 ft wide x 8 ft tall in the bottom-left corner.
    # Vertical partition (centerline x=256) + horizontal stub (centerline y=794).
    xc = ix0 + int(round(3.0 * px_per_unit)) + pg // 2  # 256
    xc_a = xc - pg // 2  # 252
    xc_b = xc_a + pg     # 261
    yh = pb_b + int(round(8.0 * px_per_unit)) + pg // 2  # 794
    yh_a = yh - pg // 2  # 790
    yh_b = yh_a + pg     # 799
    _line(img, (xc_a, pb_b), (xc_a, yh_b))
    _line(img, (xc_b, pb_b), (xc_b, yh_b))
    _line(img, (ix0, yh_a), (xc_b, yh_a))
    _line(img, (ix0, yh_b), (xc_b, yh_b))

    # Closet door: 2 ft gap in the vertical closet partition
    door_h = door_w
    dc_y0 = 690
    _erase(img, xc_a - STROKE, dc_y0, xc_b + STROKE + 1, dc_y0 + door_h)

    # Dimension strings (annotation ink the pipeline must reject)
    _dimension_string(img, (x0, 1040), (x1, 1040), "60'-0\"")
    _dimension_string(img, (140, y0), (140, y1), "40'-0\"")

    half = g / 2.0
    gt = {
        "id": "synth_corridor",
        "scale": scale_str,
        "image_size_px": [sheet_w, sheet_h],
        "rooms": [
            {"id": "R1", "bbox_px": [ix0, iy0, ix1, pa_a],
             "area_raw": round((ix1 - ix0) * (pa_a - iy0) / px_per_unit ** 2, 1)},
            {"id": "R_corridor", "bbox_px": [ix0, pa_b, ix1, pb_a],
             "area_raw": round((ix1 - ix0) * (pb_a - pa_b) / px_per_unit ** 2, 1)},
            # Bottom room is L-shaped around the closet: bbox for matching,
            # polygon for boundary-closure sampling.
            {"id": "R2", "bbox_px": [ix0, pb_b, ix1, iy1],
             "polygon_px": [[xc_b, pb_b], [ix1, pb_b], [ix1, iy1],
                            [ix0, iy1], [ix0, yh_b], [xc_b, yh_b]]},
            {"id": "R_closet", "bbox_px": [ix0, pb_b, xc_a, yh_a],
             "area_raw": round((xc_a - ix0) * (yh_a - pb_b) / px_per_unit ** 2, 1)},
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
            {"id": "int_a", "px_coords": [ix0, yp_a, ix1, yp_a],
             "facing": "North", "is_exterior": False,
             "length_raw": round((ix1 - ix0) / px_per_unit, 2)},
            {"id": "int_b", "px_coords": [ix0, yp_b, ix1, yp_b],
             "facing": "North", "is_exterior": False,
             "length_raw": round((ix1 - ix0) / px_per_unit, 2)},
            {"id": "int_closet_v", "px_coords": [xc, pb_b, xc, yh_b],
             "facing": "West", "is_exterior": False,
             "length_raw": round((yh_b - pb_b) / px_per_unit, 2)},
            {"id": "int_closet_h", "px_coords": [ix0, yh, xc_b, yh],
             "facing": "North", "is_exterior": False,
             "length_raw": round((xc_b - ix0) / px_per_unit, 2)},
        ],
        "doors": [
            {"id": "d1", "host_wall_id": "int_a",
             "bbox_px": [da_x0, pa_a - STROKE, da_x0 + door_w, pa_b + STROKE],
             "center_px": [da_x0 + door_w / 2.0, yp_a]},
            {"id": "d2", "host_wall_id": "int_b",
             "bbox_px": [db_x0, pb_a - STROKE, db_x0 + door_w, pb_b + STROKE],
             "center_px": [db_x0 + door_w / 2.0, yp_b]},
            {"id": "d3", "host_wall_id": "int_closet_v",
             "bbox_px": [xc_a - STROKE, dc_y0, xc_b + STROKE, dc_y0 + door_h],
             "center_px": [xc, dc_y0 + door_h / 2.0]},
        ],
        "windows": [],
        "labels": [],
        "dimensions": [
            {"id": "dim1", "text": "60'-0\"", "value_raw": 60.0, "unit": "ft",
             "center_px": [(x0 + x1) / 2.0, 1040.0]},
            {"id": "dim2", "text": "40'-0\"", "value_raw": 40.0, "unit": "ft",
             "center_px": [140.0, (y0 + y1) / 2.0]},
        ],
    }

    return SyntheticPlan(
        name="synth_corridor", image=img, scale_str=scale_str, dpi=dpi,
        px_per_unit=px_per_unit, ground_truth=gt,
    )


def _window_marker(img: np.ndarray, cx: int, cy: int, size: int) -> None:
    """Filled glyph marker (small square with a tiny open centre) on the wall axis."""
    h = max(2, size // 2)
    cv2.rectangle(img, (cx - h, cy - h), (cx + h, cy + h), (0, 0, 0), -1)
    cv2.rectangle(img, (cx - 1, cy - 1), (cx + 1, cy + 1), (255, 255, 255), -1)


def render_symbol_window_plan(
    px_per_unit: float = DEFAULT_PX_PER_FT,
    scale_str: str = DEFAULT_SCALE_STR,
    dpi: int = DEFAULT_DPI,
) -> SyntheticPlan:
    """Rectangular building whose east wall carries windows as a glyph series.

    Unlike ``render_two_room_plan`` (windows as gap+sill openings), here the
    east exterior wall stays continuous and the windows are drawn as a regularly
    spaced series of compact markers on the wall centreline -- the convention
    the ``symbol_on_wall`` strategy targets. The opening-based strategies cannot
    see these (the wall pair never breaks), so this fixture isolates the symbol
    detector.
    """
    if px_per_unit != DEFAULT_PX_PER_FT:
        raise ValueError("render_symbol_window_plan currently supports px_per_unit=18 only")

    sheet_w, sheet_h = 2200, 1500
    img = _new_sheet(sheet_w, sheet_h)

    g = int(round(1.0 * px_per_unit))         # exterior wall thickness: 18 px
    x0, y0, x1, y1 = 180, 220, 1260, 940      # outer faces (60 ft x 40 ft)
    ix0, iy0, ix1, iy1 = x0 + g, y0 + g, x1 - g, y1 - g

    # Continuous double-stroke exterior box (no openings broken into it).
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), STROKE)
    cv2.rectangle(img, (ix0, iy0), (ix1, iy1), (0, 0, 0), STROKE)

    half = g / 2.0
    east_axis = int(round(x1 - half))         # 1251: east wall centreline
    marker_size = int(round(0.6 * px_per_unit))  # ~11 px glyph
    spacing = int(round(5.0 * px_per_unit))   # 90 px (5 ft) on-centre
    first_y = 360
    n_markers = 5
    windows = []
    for wi in range(n_markers):
        my = first_y + wi * spacing
        _window_marker(img, east_axis, my, marker_size)
        windows.append({
            "id": f"win{wi + 1}",
            "host_wall_id": "ext_e",
            "bbox_px": [ix1, my - spacing // 2, x1, my + spacing // 2],
            "center_px": [float(east_axis), float(my)],
        })

    # Dimension string (annotation ink the pipeline must reject).
    _dimension_string(img, (x0, 1040), (x1, 1040), "60'-0\"")

    gt = {
        "id": "synth_symbol_window",
        "scale": scale_str,
        "image_size_px": [sheet_w, sheet_h],
        "rooms": [
            {"id": "R1", "bbox_px": [ix0, iy0, ix1, iy1],
             "area_raw": round((ix1 - ix0) * (iy1 - iy0) / px_per_unit ** 2, 1)},
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
        ],
        "doors": [],
        "windows": windows,
        "labels": [],
        "dimensions": [
            {"id": "dim1", "text": "60'-0\"", "value_raw": 60.0, "unit": "ft",
             "center_px": [(x0 + x1) / 2.0, 1040.0]},
        ],
    }

    return SyntheticPlan(
        name="synth_symbol_window", image=img, scale_str=scale_str, dpi=dpi,
        px_per_unit=px_per_unit, ground_truth=gt,
    )


ALL_PLANS = {
    "synth_two_room": render_two_room_plan,
    "synth_l_shape": render_l_shape_plan,
    "synth_corridor": render_corridor_plan,
    "synth_symbol_window": render_symbol_window_plan,
}
