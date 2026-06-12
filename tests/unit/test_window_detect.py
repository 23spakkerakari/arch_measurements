"""Unit tests for the collinear-gap window detector (Arqen/window_detect.py)."""

import numpy as np
import pytest

from window_detect import detect_windows, ink_mask_from_image

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


def _sill_gap_setup(gap_px=72):
    """Exterior horizontal wall pair with a sill across a gap."""
    mask = np.zeros(SHAPE, dtype=np.uint8)
    gap_lo = 400
    gap_hi = gap_lo + gap_px
    _draw_pair(mask, True, 200, 100, gap_lo)
    _draw_pair(mask, True, 200, gap_hi, gap_hi + 300)
    ink = np.zeros(SHAPE, dtype=np.uint8)
    ink[199:202, gap_lo:gap_hi] = 255
    walls = [
        _wall("w1", [100, 200, gap_lo, 200], is_exterior=True),
        _wall("w2", [gap_hi, 200, gap_hi + 300, 200], is_exterior=True),
    ]
    return mask, ink, walls


class TestInSegmentOpening:
    def test_opening_within_single_exterior_wall(self):
        """Windows often fall inside one long segment, not between two segments."""
        mask = np.zeros(SHAPE, dtype=np.uint8)
        gap_px = 72
        gap_lo, gap_hi = 300, 300 + gap_px
        _draw_pair(mask, True, 200, 100, gap_lo)
        _draw_pair(mask, True, 200, gap_hi, 700)
        ink = np.zeros(SHAPE, dtype=np.uint8)
        ink[199:202, gap_lo:gap_hi] = 255
        walls = [_wall("w1", [100, 200, 700, 200], is_exterior=True)]
        windows = detect_windows(walls, mask, ink, PPU)
        assert len(windows) == 1
        assert windows[0]["width_raw"] == pytest.approx(gap_px / PPU, abs=0.1)
        assert windows[0]["host_wall_id"] == "w1"


class TestSillGap:
    def test_basic_horizontal_sill_gap_found(self):
        mask, ink, walls = _sill_gap_setup(gap_px=72)
        windows = detect_windows(walls, mask, ink, PPU)
        assert len(windows) == 1
        w = windows[0]
        assert w["host_wall_id"] in ("w1", "w2")
        assert w["width_raw"] == pytest.approx(72 / PPU, abs=0.1)
        assert w["evidence"] == "sill"
        assert w["is_exterior"] is True
        assert w["id"] == "win1"

    def test_vertical_sill_gap_found(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        gap_px = 72
        _draw_pair(mask, False, 300, 50, 180)
        _draw_pair(mask, False, 300, 180 + gap_px, 380)
        ink = np.zeros(SHAPE, dtype=np.uint8)
        ink[50:180 + gap_px, 299:302] = 255
        walls = [
            _wall("w1", [300, 50, 300, 180], is_exterior=True),
            _wall("w2", [300, 180 + gap_px, 300, 380], is_exterior=True),
        ]
        windows = detect_windows(walls, mask, ink, PPU)
        assert len(windows) == 1
        assert windows[0]["width_raw"] == pytest.approx(gap_px / PPU, abs=0.1)

    def test_gap_without_sill_skipped(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 200, 100, 400)
        _draw_pair(mask, True, 200, 472, 772)
        walls = [
            _wall("w1", [100, 200, 400, 200], is_exterior=True),
            _wall("w2", [472, 200, 772, 200], is_exterior=True),
        ]
        ink = np.zeros(SHAPE, dtype=np.uint8)
        assert detect_windows(walls, mask, ink, PPU) == []

    def test_interior_wall_with_sill_skipped(self):
        mask, ink, walls = _sill_gap_setup()
        walls = [
            _wall("w1", walls[0]["px_coords"], is_exterior=False),
            _wall("w2", walls[1]["px_coords"], is_exterior=False),
        ]
        assert detect_windows(walls, mask, ink, PPU) == []

    def test_merged_subsegments_still_find_window(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        gap_px = 72
        _draw_pair(mask, True, 200, 100, 300)
        _draw_pair(mask, True, 200, 300, 500)
        _draw_pair(mask, True, 200, 500 + gap_px, 700)
        ink = np.zeros(SHAPE, dtype=np.uint8)
        ink[199:202, 500:500 + gap_px] = 255
        walls = [
            _wall("w1", [100, 200, 300, 200], is_exterior=True),
            _wall("w2", [300, 200, 500, 200], is_exterior=True),
            _wall("w3", [500 + gap_px, 200, 700, 200], is_exterior=True),
        ]
        windows = detect_windows(walls, mask, ink, PPU)
        assert len(windows) == 1
        assert windows[0]["width_raw"] == pytest.approx(gap_px / PPU, abs=0.15)


class TestWidthRange:
    def test_too_narrow_gap_skipped(self):
        # 30 px ~ 1.67 ft, below 2.0 ft window minimum
        mask, ink, walls = _sill_gap_setup(gap_px=30)
        assert detect_windows(walls, mask, ink, PPU) == []

    def test_too_wide_gap_skipped(self):
        # 150 px ~ 8.3 ft, above 8.0 ft window maximum
        mask, ink, walls = _sill_gap_setup(gap_px=150)
        assert detect_windows(walls, mask, ink, PPU) == []


class TestOpenGapVerification:
    def test_ink_filled_gap_skipped(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        _draw_pair(mask, True, 200, 100, 772)  # continuous ink
        ink = np.zeros(SHAPE, dtype=np.uint8)
        ink[199:202, 400:472] = 255
        walls = [
            _wall("w1", [100, 200, 400, 200], is_exterior=True),
            _wall("w2", [472, 200, 772, 200], is_exterior=True),
        ]
        assert detect_windows(walls, mask, ink, PPU) == []


class TestDoorDedup:
    def test_near_door_center_skipped(self):
        mask, ink, walls = _sill_gap_setup()
        cx = (400 + 472) / 2.0
        doors = [{"center_px": [cx, 200.0]}]
        assert detect_windows(walls, mask, ink, PPU, doors=doors) == []


class TestEmit:
    def test_ids_sequential_and_deduped(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        gap_px = 72
        _draw_pair(mask, True, 100, 100, 300)
        _draw_pair(mask, True, 100, 300 + gap_px, 600)
        _draw_pair(mask, False, 500, 150, 250)
        _draw_pair(mask, False, 500, 250 + gap_px, 390)
        ink = np.zeros(SHAPE, dtype=np.uint8)
        ink[99:102, 300:300 + gap_px] = 255
        ink[150:250 + gap_px, 499:502] = 255
        walls = [
            _wall("w1", [100, 100, 300, 100], is_exterior=True),
            _wall("w2", [300 + gap_px, 100, 600, 100], is_exterior=True),
            _wall("w3", [500, 150, 500, 250], is_exterior=True),
            _wall("w4", [500, 250 + gap_px, 500, 390], is_exterior=True),
        ]
        windows = detect_windows(walls, mask, ink, PPU)
        assert [w["id"] for w in windows] == ["win1", "win2"]

    def test_diagonal_walls_ignored(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        ink = np.zeros(SHAPE, dtype=np.uint8)
        walls = [
            _wall("w1", [100, 100, 300, 300], is_exterior=True),
            _wall("w2", [340, 340, 600, 600], is_exterior=True),
        ]
        assert detect_windows(walls, mask, ink, PPU) == []

    def test_no_ink_mask_returns_empty(self):
        mask, _, walls = _sill_gap_setup()
        assert detect_windows(walls, mask, None, PPU) == []


class TestInkMask:
    def test_ink_mask_from_rgb(self):
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        img[10:20, 10:40] = 0
        mask = ink_mask_from_image(img)
        assert mask[15, 25] == 255
        assert mask[40, 40] == 0
