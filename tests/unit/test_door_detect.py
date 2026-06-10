"""Unit tests for the collinear-gap door detector (Arqen/door_detect.py)."""

import numpy as np
import pytest

from door_detect import detect_doors, ink_mask_from_image

pytestmark = pytest.mark.unit

PPU = 18.0  # px per ft, matches the synth plans
SHAPE = (400, 800)


def _wall(wid, coords, is_exterior=False):
    return {"id": wid, "px_coords": list(coords), "is_exterior": is_exterior}


def _draw_pair(mask, horiz, axis, lo, hi, gap=9, thick=3):
    """Draw the two parallel strokes of a wall onto mask."""
    a = int(axis - gap // 2)
    b = a + gap
    if horiz:
        mask[a - thick + 1:a + 1, lo:hi] = 255
        mask[b:b + thick, lo:hi] = 255
    else:
        mask[lo:hi, a - thick + 1:a + 1] = 255
        mask[lo:hi, b:b + thick] = 255


class TestCollinearGap:
    def test_basic_horizontal_gap_found(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 200, 100, 400)
        _draw_pair(mask, True, 200, 440, 700)
        walls = [
            _wall("w1", [100, 200, 400, 200]),
            _wall("w2", [440, 200, 700, 200]),
        ]
        doors = detect_doors(walls, mask, None, PPU)
        assert len(doors) == 1
        d = doors[0]
        assert d["host_wall_id"] == "w1"  # longer flank
        assert d["center_px"][0] == pytest.approx(420, abs=2)
        assert d["center_px"][1] == pytest.approx(200, abs=2)
        assert d["width_raw"] == pytest.approx(40 / PPU, abs=0.1)
        assert d["evidence"] == "gap"
        assert d["is_exterior"] is False

    def test_vertical_gap_found(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, False, 300, 50, 180)
        _draw_pair(mask, False, 300, 220, 380)
        walls = [
            _wall("w1", [300, 50, 300, 180]),
            _wall("w2", [300, 220, 300, 380]),
        ]
        doors = detect_doors(walls, mask, None, PPU)
        assert len(doors) == 1
        assert doors[0]["center_px"][1] == pytest.approx(200, abs=2)

    def test_zero_gap_subsegment_boundary_skipped(self):
        # Exterior per-room sub-segments share an endpoint: gap 0 is no door.
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 200, 100, 700)
        walls = [
            _wall("w1.s1", [100, 200, 400, 200], is_exterior=True),
            _wall("w1.s2", [400, 200, 700, 200], is_exterior=True),
        ]
        assert detect_doors(walls, mask, None, PPU) == []

    def test_too_wide_gap_skipped(self):
        # > 5 ft (e.g. a garage opening) is not a standard door.
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 200, 100, 300)
        _draw_pair(mask, True, 200, 420, 700)  # 120 px = 6.7 ft gap
        walls = [
            _wall("w1", [100, 200, 300, 200]),
            _wall("w2", [420, 200, 700, 200]),
        ]
        assert detect_doors(walls, mask, None, PPU) == []

    def test_non_collinear_walls_not_paired(self):
        # Parallel walls far apart on the perpendicular axis: no candidate.
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 150, 100, 400)
        _draw_pair(mask, True, 250, 440, 700)
        walls = [
            _wall("w1", [100, 150, 400, 150]),
            _wall("w2", [440, 250, 700, 250]),
        ]
        assert detect_doors(walls, mask, None, PPU) == []


class TestOpenGapVerification:
    def test_ink_filled_gap_skipped(self):
        # A dedup/cleanup split leaves wall ink across the "gap": not a door.
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 200, 100, 700)  # continuous ink
        walls = [
            _wall("w1", [100, 200, 400, 200]),
            _wall("w2", [440, 200, 700, 200]),
        ]
        assert detect_doors(walls, mask, None, PPU) == []


class TestSillDiscriminator:
    def test_gap_with_parallel_sill_skipped(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 200, 100, 400)
        _draw_pair(mask, True, 200, 440, 700)
        ink = np.zeros(SHAPE, dtype=np.uint8)
        ink[199:202, 400:440] = 255  # thin stroke across the gap = window sill
        walls = [
            _wall("w1", [100, 200, 400, 200]),
            _wall("w2", [440, 200, 700, 200]),
        ]
        assert detect_doors(walls, mask, ink, PPU) == []

    def test_perpendicular_door_leaf_not_a_sill(self):
        # An open door leaf drawn perpendicular to the wall must not be
        # mistaken for a sill.
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 200, 100, 400)
        _draw_pair(mask, True, 200, 440, 700)
        ink = np.zeros(SHAPE, dtype=np.uint8)
        ink[160:240, 402:405] = 255  # leaf at the hinge edge
        walls = [
            _wall("w1", [100, 200, 400, 200]),
            _wall("w2", [440, 200, 700, 200]),
        ]
        doors = detect_doors(walls, mask, ink, PPU)
        assert len(doors) == 1


class TestEmit:
    def test_ids_sequential_and_deduped(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 100, 100, 300)
        _draw_pair(mask, True, 100, 340, 600)
        _draw_pair(mask, False, 500, 150, 250)
        _draw_pair(mask, False, 500, 290, 390)
        walls = [
            _wall("w1", [100, 100, 300, 100]),
            _wall("w2", [340, 100, 600, 100]),
            _wall("w3", [500, 150, 500, 250]),
            _wall("w4", [500, 290, 500, 390]),
        ]
        doors = detect_doors(walls, mask, None, PPU)
        assert [d["id"] for d in doors] == ["d1", "d2"]

    def test_exterior_flag_from_flanking_walls(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 200, 100, 400)
        _draw_pair(mask, True, 200, 440, 700)
        walls = [
            _wall("w1.s1", [100, 200, 400, 200], is_exterior=True),
            _wall("w1.s2", [440, 200, 700, 200], is_exterior=True),
        ]
        doors = detect_doors(walls, mask, None, PPU)
        assert len(doors) == 1
        assert doors[0]["is_exterior"] is True

    def test_diagonal_walls_ignored(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        walls = [
            _wall("w1", [100, 100, 300, 300]),
            _wall("w2", [340, 340, 600, 600]),
        ]
        assert detect_doors(walls, mask, None, PPU) == []


class TestInkMask:
    def test_ink_mask_from_rgb(self):
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        img[10:20, 10:40] = 0
        mask = ink_mask_from_image(img)
        assert mask[15, 25] == 255
        assert mask[40, 40] == 0
