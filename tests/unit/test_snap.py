"""Stage [8] Snap segments to wall ink — preprocess.snap_segments_to_walls."""

import numpy as np
import pytest

from preprocess import snap_segments_to_walls

pytestmark = pytest.mark.unit


def _horizontal_wall_mask(h=400, w=400, rows=((100, 103), (112, 115)), x0=50, x1=351):
    mask = np.zeros((h, w), np.uint8)
    for r0, r1 in rows:
        mask[r0:r1, x0:x1] = 255
    return mask


def _vertical_wall_mask(h=400, w=400, cols=((100, 103), (112, 115)), y0=50, y1=351):
    mask = np.zeros((h, w), np.uint8)
    for c0, c1 in cols:
        mask[y0:y1, c0:c1] = 255
    return mask


class TestHorizontalSnap:
    def test_snaps_to_median_of_wall_pair(self):
        mask = _horizontal_wall_mask()
        snapped = snap_segments_to_walls(
            [(60, 130, 340, 130)], mask, px_per_unit=18.0,
        )
        (x1, y1, x2, y2) = snapped[0]
        assert y1 == y2
        assert 100 <= y1 <= 114          # within the ink band
        assert (x1, x2) == (60, 340)     # along-axis extent preserved

    def test_near_top_edge_snaps_to_outermost_row(self):
        # cy < 12% of height -> snap to min row, not median
        mask = _horizontal_wall_mask(rows=((30, 33), (42, 45)))
        snapped = snap_segments_to_walls(
            [(60, 20, 340, 20)], mask, px_per_unit=18.0,
        )
        assert snapped[0][1] == 30

    def test_no_ink_leaves_segment_unchanged(self):
        mask = _horizontal_wall_mask()
        seg = (60, 300, 340, 300)
        snapped = snap_segments_to_walls([seg], mask, px_per_unit=18.0)
        assert snapped[0] == seg


class TestVerticalSnap:
    def test_snaps_vertical_segment(self):
        mask = _vertical_wall_mask()
        snapped = snap_segments_to_walls(
            [(130, 60, 130, 340)], mask, px_per_unit=18.0,
        )
        (x1, y1, x2, y2) = snapped[0]
        assert x1 == x2
        assert 100 <= x1 <= 114
        assert (y1, y2) == (60, 340)


class TestRadius:
    def test_radius_scales_with_px_per_unit(self):
        # Ink 150 px away: default radius 80 misses it, 2*px_per_unit=160 finds it
        mask = _horizontal_wall_mask(rows=((100, 103),))
        seg = (60, 250, 340, 250)
        far = snap_segments_to_walls([seg], mask, px_per_unit=18.0)
        assert far[0] == seg
        near = snap_segments_to_walls([seg], mask, px_per_unit=80.0)
        assert near[0][1] in range(100, 103)
