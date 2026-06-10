"""Stage [6] Polygon -> exterior segments — extract_wall_segments_class.py."""

import numpy as np
import pytest

from extract_wall_segments_class import (
    angle_diff_deg,
    extract_wall_segments,
    filter_non_orthogonal_segments,
    filter_short_segments,
    merge_collinear_segments,
    polygon_to_segments,
    segment_angle_deg,
    segment_length,
)

pytestmark = pytest.mark.unit

SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])


class TestBasics:
    def test_segment_length(self):
        assert segment_length((0, 0, 3, 4)) == pytest.approx(5.0)

    def test_segment_angle_horizontal(self):
        assert segment_angle_deg((0, 0, 10, 0)) == pytest.approx(0.0)

    def test_segment_angle_vertical(self):
        assert segment_angle_deg((0, 0, 0, 10)) == pytest.approx(90.0)

    def test_angle_diff_wraps_mod_180(self):
        assert angle_diff_deg(170, 10) == pytest.approx(20.0)
        assert angle_diff_deg(0, 90) == pytest.approx(90.0)


class TestPolygonToSegments:
    def test_square_closes_loop(self):
        segs = polygon_to_segments(SQUARE)
        assert len(segs) == 4
        assert segs[-1] == (0, 100, 0, 0)  # closing edge back to first vertex


class TestFilters:
    def test_filter_short(self):
        segs = [(0, 0, 100, 0), (0, 0, 5, 0)]
        assert filter_short_segments(segs, min_length_px=15) == [(0, 0, 100, 0)]

    def test_filter_non_orthogonal_drops_diagonal(self):
        segs = [(0, 0, 100, 0), (0, 0, 100, 100), (0, 0, 0, 100)]
        kept = filter_non_orthogonal_segments(segs, angle_tolerance_deg=10)
        assert (0, 0, 100, 100) not in kept
        assert len(kept) == 2

    def test_filter_non_orthogonal_keeps_within_tolerance(self):
        # ~8 degrees off horizontal: kept at the 10 degree default
        seg = (0, 0, 100, 14)
        assert filter_non_orthogonal_segments([seg]) == [seg]


class TestMergeCollinear:
    def test_merges_split_edge(self):
        # Square with a redundant midpoint on the top edge
        poly = np.array([[0, 0], [50, 0], [100, 0], [100, 100], [0, 100]])
        segs = polygon_to_segments(poly)
        merged = merge_collinear_segments(segs)
        assert len(merged) == 4

    def test_keeps_distinct_corners(self):
        segs = polygon_to_segments(SQUARE)
        merged = merge_collinear_segments(segs)
        assert len(merged) == 4

    def test_empty(self):
        assert merge_collinear_segments([]) == []


class TestExtractWallSegments:
    def test_rectangle_with_midpoints(self):
        poly = np.array([
            [0, 0], [60, 0], [120, 0],
            [120, 80], [120, 160],
            [60, 160], [0, 160],
            [0, 80],
        ])
        segs = extract_wall_segments(poly, min_length_px=15)
        assert len(segs) == 4
        lengths = sorted(round(segment_length(s)) for s in segs)
        assert lengths == [120, 120, 160, 160]

    def test_diagonal_edges_dropped(self):
        # Octagon-ish shape: 4 orthogonal + 4 diagonal edges
        poly = np.array([
            [40, 0], [120, 0],          # top
            [160, 40], [160, 120],      # right (diag in, then vertical)
            [120, 160], [40, 160],      # bottom
            [0, 120], [0, 40],          # left
        ])
        segs = extract_wall_segments(poly, min_length_px=15)
        for seg in segs:
            angle = segment_angle_deg(seg)
            dist = min(angle, 180 - angle, abs(angle - 90))
            assert dist <= 10.0
