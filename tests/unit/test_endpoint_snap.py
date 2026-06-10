"""Stage [12b] Endpoint snapping — preprocess.snap_wall_endpoints."""

import pytest

from preprocess import pixel_length, snap_wall_endpoints

pytestmark = pytest.mark.unit

PPU = 18.0  # snap_tol = max(12, 2.0*18) = 36 px


def _wall(wid, coords, **extra):
    length = pixel_length(*coords) / PPU
    record = {
        "id": wid,
        "name": f"Wall {wid}",
        "facing": "North",
        "length": f"{length:.2f} ft",
        "length_raw": round(length, 2),
        "angle_deg": 0.0,
        "px_coords": list(coords),
        "is_exterior": False,
    }
    record.update(extra)
    return record


def _coords(walls, wid):
    return next(w["px_coords"] for w in walls if w["id"] == wid)


class TestCornerSnap:
    def test_l_corner_closes_both_endpoints(self):
        # H wall stops 15 px short of the V wall axis; V stops 10 px short of H.
        walls = [
            _wall("h", (100, 200, 485, 200)),
            _wall("v", (500, 210, 500, 600)),
        ]
        out, moved = snap_wall_endpoints(walls, PPU, "ft")
        assert moved == 2
        assert _coords(out, "h") == [100, 200, 500, 200]
        assert _coords(out, "v") == [500, 200, 500, 600]

    def test_overshoot_trimmed_back_to_corner(self):
        walls = [
            _wall("h", (100, 200, 515, 200)),   # runs 15 px past the corner
            _wall("v", (500, 200, 500, 600)),
        ]
        out, _ = snap_wall_endpoints(walls, PPU, "ft")
        assert _coords(out, "h") == [100, 200, 500, 200]

    def test_lengths_updated_after_snap(self):
        walls = [
            _wall("h", (100, 200, 485, 200)),
            _wall("v", (500, 210, 500, 600)),
        ]
        out, _ = snap_wall_endpoints(walls, PPU, "ft")
        h = next(w for w in out if w["id"] == "h")
        assert h["length_raw"] == round(400 / PPU, 2)
        assert h["length"] == f"{400 / PPU:.2f} ft"


class TestTJunction:
    def test_endpoint_extends_to_perpendicular_body(self):
        walls = [
            _wall("v", (500, 100, 500, 600)),
            _wall("h", (520, 300, 900, 300)),   # abuts v's body, 20 px short
        ]
        out, moved = snap_wall_endpoints(walls, PPU, "ft")
        assert moved == 1
        assert _coords(out, "h") == [500, 300, 900, 300]
        assert _coords(out, "v") == [500, 100, 500, 600]  # body never moves

    def test_nearest_of_two_targets_wins(self):
        walls = [
            _wall("v1", (480, 100, 480, 600)),
            _wall("v2", (500, 100, 500, 600)),
            _wall("h", (508, 300, 900, 300)),
        ]
        out, _ = snap_wall_endpoints(walls, PPU, "ft")
        assert _coords(out, "h")[0] == 500


class TestNeverBridge:
    def test_collinear_doorway_gap_not_bridged(self):
        # Two collinear H walls 20 px apart: a doorway, not a corner.
        walls = [
            _wall("h1", (100, 200, 300, 200)),
            _wall("h2", (320, 200, 600, 200)),
        ]
        out, moved = snap_wall_endpoints(walls, PPU, "ft")
        assert moved == 0
        assert _coords(out, "h1") == [100, 200, 300, 200]
        assert _coords(out, "h2") == [320, 200, 600, 200]

    def test_target_beyond_tolerance_ignored(self):
        walls = [
            _wall("h", (100, 200, 455, 200)),   # 45 px short > snap_tol 36
            _wall("v", (500, 100, 500, 600)),
        ]
        out, moved = snap_wall_endpoints(walls, PPU, "ft")
        assert moved == 0
        assert _coords(out, "h") == [100, 200, 455, 200]

    def test_endpoint_far_from_span_not_snapped(self):
        # Axis distance is fine but the endpoint is nowhere near the wall's span.
        walls = [
            _wall("h", (100, 200, 490, 200)),
            _wall("v", (500, 400, 500, 800)),   # span starts 200 px below
        ]
        out, moved = snap_wall_endpoints(walls, PPU, "ft")
        assert moved == 0


class TestSafety:
    def test_diagonal_walls_untouched(self):
        walls = [
            _wall("d", (100, 100, 300, 290)),
            _wall("v", (310, 100, 310, 600)),
        ]
        out, moved = snap_wall_endpoints(walls, PPU, "ft")
        assert moved == 0
        assert _coords(out, "d") == [100, 100, 300, 290]

    def test_tiny_wall_not_degenerated(self):
        # Snapping would shrink the 20 px stub below the minimum length.
        walls = [
            _wall("stub", (485, 200, 505, 200)),
            _wall("v", (500, 100, 500, 600)),
        ]
        out, _ = snap_wall_endpoints(walls, PPU, "ft")
        x1, _, x2, _ = _coords(out, "stub")
        assert abs(x2 - x1) >= 4

    def test_direction_preserved(self):
        # Reversed-direction segment (x2 < x1) keeps its orientation.
        walls = [
            _wall("h", (485, 200, 100, 200)),
            _wall("v", (500, 210, 500, 600)),
        ]
        out, _ = snap_wall_endpoints(walls, PPU, "ft")
        assert _coords(out, "h") == [500, 200, 100, 200]

    def test_empty_list(self):
        out, moved = snap_wall_endpoints([], PPU, "ft")
        assert out == [] and moved == 0
