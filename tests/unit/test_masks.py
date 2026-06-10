"""Stages [3]-[4] Mask extraction and footprint — preprocess.py mask functions."""

import numpy as np
import pytest

from preprocess import (
    _build_exclusion_mask,
    _find_wall_pairs,
    _point_in_exclusion,
    _strip_spanning_grid_lines,
    dedup_axis_tol_px,
    find_footprint,
    find_footprint_contour,
    flood_fill_interior,
    simplify_polygon,
    wall_pair_gap_range,
)

pytestmark = pytest.mark.unit


class TestScaleAdaptiveParams:
    def test_dedup_axis_tol_floor(self):
        assert dedup_axis_tol_px(10.0) == 12  # floor

    def test_dedup_axis_tol_scales(self):
        assert dedup_axis_tol_px(40.0) == 24  # 0.6 * 40

    def test_wall_pair_gap_range_at_18(self):
        assert wall_pair_gap_range(18.0) == (3, 27)

    def test_wall_pair_gap_range_floors(self):
        min_gap, max_gap = wall_pair_gap_range(6.0)
        assert min_gap == 3
        assert max_gap == 12


class TestFindWallPairs:
    def _mask_with_rows(self, rows, h=200, w=200, x0=20, x1=180):
        mask = np.zeros((h, w), np.uint8)
        for y in rows:
            mask[y, x0:x1] = 255
        return mask

    def test_paired_rows_kept(self):
        mask = self._mask_with_rows([50, 60])
        out = _find_wall_pairs(mask, scan_rows=True, min_gap_px=2, max_gap_px=60)
        assert out[50].sum() > 0
        assert out[60].sum() > 0

    def test_isolated_row_removed(self):
        mask = self._mask_with_rows([50, 60, 150])
        out = _find_wall_pairs(mask, scan_rows=True, min_gap_px=2, max_gap_px=60)
        assert out[150].sum() == 0

    def test_gap_below_min_removed(self):
        mask = self._mask_with_rows([50, 60])
        out = _find_wall_pairs(mask, scan_rows=True, min_gap_px=15, max_gap_px=60)
        assert out.sum() == 0

    def test_gap_above_max_removed(self):
        mask = self._mask_with_rows([20, 150])
        out = _find_wall_pairs(mask, scan_rows=True, min_gap_px=2, max_gap_px=60)
        assert out.sum() == 0

    def test_column_pairs(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[20:180, 50] = 255
        mask[20:180, 62] = 255
        mask[20:180, 150] = 255  # isolated
        out = _find_wall_pairs(mask, scan_rows=False, min_gap_px=2, max_gap_px=60)
        assert out[:, 50].sum() > 0
        assert out[:, 62].sum() > 0
        assert out[:, 150].sum() == 0


class TestStripSpanningGridLines:
    def test_spanning_horizontal_stripped(self):
        mask = np.zeros((1000, 1000), np.uint8)
        mask[100:103, :] = 255          # full-width grid line
        mask[500:503, 100:300] = 255    # short wall run
        out = _strip_spanning_grid_lines(mask, span_frac=0.42)
        assert out[100:103].sum() == 0
        assert out[500:503, 100:300].sum() > 0

    def test_vertical_kept_when_disabled(self):
        mask = np.zeros((1000, 1000), np.uint8)
        mask[:, 200:203] = 255
        out = _strip_spanning_grid_lines(mask, span_frac=0.42, strip_vertical=False)
        assert out[:, 200:203].sum() > 0

    def test_vertical_stripped_when_enabled(self):
        mask = np.zeros((1000, 1000), np.uint8)
        mask[:, 200:203] = 255
        out = _strip_spanning_grid_lines(mask, span_frac=0.42, strip_vertical=True)
        assert out[:, 200:203].sum() == 0


class TestFloodFillInterior:
    def test_fills_closed_rectangle(self):
        import cv2

        binary = np.zeros((300, 300), np.uint8)
        cv2.rectangle(binary, (50, 50), (250, 250), 255, 3)
        filled = flood_fill_interior(binary)
        assert filled[150, 150] == 255          # interior now solid
        assert binary[150, 150] == 0            # input untouched
        assert filled.sum() > binary.sum()

    def test_open_shape_returned_unchanged(self):
        binary = np.zeros((300, 300), np.uint8)
        binary[100, 50:250] = 255  # a single line encloses nothing
        filled = flood_fill_interior(binary)
        assert np.array_equal(filled, binary)


class TestFindFootprint:
    def _plan_mask(self):
        """Hollow rectangle in the sheet's safe zone + a small noise blob."""
        import cv2

        binary = np.zeros((1000, 1000), np.uint8)
        cv2.rectangle(binary, (150, 200), (550, 700), 255, 3)
        cv2.rectangle(binary, (700, 200), (760, 260), 255, -1)  # noise blob
        return binary

    def test_picks_building_component(self):
        mask = find_footprint(self._plan_mask())
        assert mask is not None
        # Footprint contains the building interior, not the noise blob
        assert mask[450, 350] == 255
        assert mask[230, 730] == 0

    def test_blank_input_returns_none(self):
        assert find_footprint(np.zeros((500, 500), np.uint8)) is None

    def test_roi_mode_allows_edge_touch(self):
        import cv2

        binary = np.zeros((600, 800), np.uint8)
        # Building filling much of an ROI crop, touching the top edge
        cv2.rectangle(binary, (40, 0), (700, 500), 255, 3)
        mask = find_footprint(binary, use_exclusion=False)
        assert mask is not None
        assert mask[250, 370] == 255

    def test_contour_and_polygon(self):
        mask = find_footprint(self._plan_mask())
        contour = find_footprint_contour(mask)
        assert contour is not None
        poly = simplify_polygon(contour, epsilon_factor=0.01)
        assert poly.shape[1] == 2
        assert len(poly) >= 4
        xs, ys = poly[:, 0], poly[:, 1]
        assert 140 <= xs.min() <= 160
        assert 540 <= xs.max() <= 560
        assert 190 <= ys.min() <= 210
        assert 690 <= ys.max() <= 710

    def test_contour_none_for_none_mask(self):
        assert find_footprint_contour(None) is None


class TestExclusionZones:
    def test_mask_and_point_check_agree(self):
        h, w = 1000, 2000
        mask = _build_exclusion_mask(h, w)
        for (px, py) in [(100, 50), (1900, 900), (1500, 700), (1000, 500), (300, 400)]:
            assert (mask[py, px] > 0) == _point_in_exclusion(px, py, w, h)

    def test_center_not_excluded(self):
        assert not _point_in_exclusion(800, 600, 2000, 1500)

    def test_title_block_excluded(self):
        # bottom-right quadrant (x > 58%, y > 50%)
        assert _point_in_exclusion(1800, 1100, 2000, 1500)
