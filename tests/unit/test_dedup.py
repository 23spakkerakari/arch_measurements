"""Stages [10b]/[12] Dedup and cleanup — preprocess.py wall-list passes."""

import pytest

from preprocess import (
    cleanup_wall_list,
    coaxial_spanning_wall_indices,
    consolidate_coaxial_wall_duplicates,
    dedup_axis_tol_px,
    drop_dimension_like_walls,
    drop_spanning_coaxial_walls,
    merge_and_deduplicate_segments,
    pixel_length,
    _run_cleanup_pass,
)

pytestmark = pytest.mark.unit

PPU = 18.0


def _wall(wid, coords, facing="North", is_exterior=False, **extra):
    length = pixel_length(*coords) / PPU
    record = {
        "id": wid,
        "name": f"{facing} Wall",
        "facing": facing,
        "length": f"{length:.2f} ft",
        "length_raw": round(length, 2),
        "angle_deg": 90.0,
        "px_coords": list(coords),
        "is_exterior": is_exterior,
    }
    record.update(extra)
    return record


class TestMergeAndDeduplicateSegments:
    def test_merges_double_stroke(self):
        segs = [(100, 200, 500, 200), (100, 208, 500, 208)]
        merged = merge_and_deduplicate_segments(segs, axis_tol_px=12, gap_tol_px=8)
        assert len(merged) == 1
        (x1, y1, x2, y2) = merged[0]
        assert (min(x1, x2), max(x1, x2)) == (100, 500)
        assert 200 <= y1 <= 208

    def test_distinct_parallel_walls_survive(self):
        segs = [(100, 200, 500, 200), (100, 260, 500, 260)]
        merged = merge_and_deduplicate_segments(segs, axis_tol_px=12, gap_tol_px=8)
        assert len(merged) == 2

    def test_collinear_gap_within_tolerance_merged(self):
        segs = [(100, 200, 300, 200), (305, 200, 500, 200)]
        merged = merge_and_deduplicate_segments(segs, axis_tol_px=12, gap_tol_px=8)
        assert len(merged) == 1

    def test_collinear_gap_beyond_tolerance_kept_separate(self):
        segs = [(100, 200, 300, 200), (320, 200, 500, 200)]
        merged = merge_and_deduplicate_segments(segs, axis_tol_px=12, gap_tol_px=8)
        assert len(merged) == 2

    def test_subsumed_fragment_collapsed(self):
        segs = [(100, 200, 600, 200), (250, 205, 350, 205)]
        merged = merge_and_deduplicate_segments(segs, axis_tol_px=12, gap_tol_px=8)
        assert len(merged) == 1

    def test_mixed_orientations_independent(self):
        segs = [(100, 200, 500, 200), (300, 100, 300, 400)]
        merged = merge_and_deduplicate_segments(segs, axis_tol_px=12, gap_tol_px=8)
        assert len(merged) == 2

    def test_empty(self):
        assert merge_and_deduplicate_segments([]) == []


class TestCoaxialSpanning:
    def _tiled_walls(self):
        return [
            _wall("w_long", (100, 300, 1100, 300), is_exterior=True),
            _wall("w_a", (100, 306, 580, 306), is_exterior=True),
            _wall("w_b", (560, 306, 1100, 306), is_exterior=True),
        ]

    def test_spanning_duplicate_detected(self):
        walls = self._tiled_walls()
        idx = coaxial_spanning_wall_indices(walls, axis_tol_px=12)
        assert idx == {0}

    def test_drop_spanning(self):
        walls = self._tiled_walls()
        kept = drop_spanning_coaxial_walls(walls, axis_tol_px=12, px_per_unit=PPU)
        assert [w["id"] for w in kept] == ["w_a", "w_b"]

    def test_partial_cover_not_dropped(self):
        walls = [
            _wall("w_long", (100, 300, 1100, 300), is_exterior=False),
            _wall("w_a", (100, 306, 220, 306)),
            _wall("w_b", (240, 306, 360, 306)),
        ]
        idx = coaxial_spanning_wall_indices(walls, axis_tol_px=12)
        assert idx == set()

    def test_single_contributor_never_drops(self):
        walls = [
            _wall("w_long", (100, 300, 1100, 300)),
            _wall("w_a", (100, 306, 1100, 306)),
        ]
        # one contributor is a duplicate stroke, not a tiling — needs >= 2
        idx = coaxial_spanning_wall_indices(walls, axis_tol_px=12)
        assert idx == set()


class TestDimensionLike:
    def test_dimension_offset_dropped(self):
        walls = [
            _wall("w_wall", (100, 200, 460, 200), is_exterior=True),
            _wall("w_dim", (100, 220, 280, 220)),  # 10 ft, 20 px offset
        ]
        kept = drop_dimension_like_walls(walls, axis_tol_px=12, px_per_unit=PPU)
        assert [w["id"] for w in kept] == ["w_wall"]

    def test_long_interior_kept(self):
        walls = [
            _wall("w_wall", (100, 200, 700, 200), is_exterior=True),
            _wall("w_int", (100, 220, 700, 220)),  # 33 ft: too long to be a dim
        ]
        kept = drop_dimension_like_walls(walls, axis_tol_px=12, px_per_unit=PPU)
        assert len(kept) == 2

    def test_far_offset_kept(self):
        walls = [
            _wall("w_wall", (100, 200, 460, 200), is_exterior=True),
            _wall("w_other", (100, 400, 280, 400)),
        ]
        kept = drop_dimension_like_walls(walls, axis_tol_px=12, px_per_unit=PPU)
        assert len(kept) == 2


class TestConsolidate:
    def test_overlapping_strokes_merged(self):
        walls = [
            _wall("w_a", (100, 200, 500, 200), is_exterior=True),
            _wall("w_b", (120, 210, 520, 210), is_exterior=True),
        ]
        merged = consolidate_coaxial_wall_duplicates(
            walls, axis_tol_px=12, px_per_unit=PPU, unit_label="ft",
        )
        assert len(merged) == 1
        coords = merged[0]["px_coords"]
        lo, hi = min(coords[0], coords[2]), max(coords[0], coords[2])
        assert (lo, hi) == (100, 520)
        assert merged[0]["length_raw"] == pytest.approx(420 / PPU, abs=0.01)

    def test_exterior_interior_not_merged(self):
        walls = [
            _wall("w_ext", (100, 200, 500, 200), is_exterior=True),
            _wall("w_int", (120, 210, 520, 210), is_exterior=False),
        ]
        merged = consolidate_coaxial_wall_duplicates(
            walls, axis_tol_px=12, px_per_unit=PPU, unit_label="ft",
        )
        assert len(merged) == 2


class TestCleanupWallList:
    def _messy_walls(self):
        return [
            _wall("w1", (100, 300, 1100, 300), is_exterior=True,
                  parent_wall_id="w1", segment_count=1),
            _wall("w1.s1", (100, 306, 580, 306), is_exterior=True,
                  parent_wall_id="wp", segment_count=2),
            _wall("w1.s2", (560, 306, 1100, 306), is_exterior=True,
                  parent_wall_id="wp", segment_count=2),
            _wall("w_int", (300, 500, 800, 500)),
            _wall("w_dim", (300, 520, 480, 520)),
        ]

    def test_stats_keys(self):
        _, stats = cleanup_wall_list(
            self._messy_walls(), dedup_axis_tol_px(PPU), PPU, "ft",
        )
        assert set(stats) == {
            "duplicate_exterior_strokes", "dimension_like", "spanning",
            "coaxial_merge", "exterior_span", "spanning_final",
            "coaxial_merge_final",
        }

    def test_removes_spanning_and_dimension_walls(self):
        cleaned, stats = cleanup_wall_list(
            self._messy_walls(), dedup_axis_tol_px(PPU), PPU, "ft",
        )
        ids = {w["id"] for w in cleaned}
        assert "w_dim" not in ids        # dimension-like dropped
        assert "w1" not in ids           # spanning duplicate dropped
        assert {"w1.s1", "w1.s2", "w_int"} <= ids
        assert sum(v for k, v in stats.items() if not k.endswith("_skipped")) == 2

    def test_audit_trail_records_drops(self):
        _, _, audit = cleanup_wall_list(
            self._messy_walls(), dedup_axis_tol_px(PPU), PPU, "ft", audit=True,
        )
        dropped = {(d["id"], d["pass"]) for d in audit["drops"]}
        assert ("w_dim", "dimension_like") in dropped
        assert ("w1", "duplicate_exterior_strokes") in dropped or ("w1", "spanning") in dropped

    def test_length_guard_skips_pathological_pass(self):
        walls = [
            _wall(f"w{i}", (100, 200 + i * 5, 900, 200 + i * 5))
            for i in range(10)
        ]
        stats: dict[str, int] = {}
        audit: dict = {}
        out = _run_cleanup_pass(
            walls, "test_pass", lambda w: w[:1], stats, audit, max_drop_frac=0.40,
        )
        assert len(out) == 10
        assert stats["test_pass_skipped"] == 9
        assert audit["skipped"][0]["pass"] == "test_pass"

    def test_second_pass_is_stable(self):
        cleaned, _ = cleanup_wall_list(
            self._messy_walls(), dedup_axis_tol_px(PPU), PPU, "ft",
        )
        cleaned2, stats2 = cleanup_wall_list(
            [dict(w) for w in cleaned], dedup_axis_tol_px(PPU), PPU, "ft",
        )
        assert len(cleaned2) == len(cleaned)
        assert sum(v for k, v in stats2.items() if not k.endswith("_skipped")) == 0
