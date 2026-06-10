"""Stage [9] Room split — room_wall_split.py on synthetic masks."""

import cv2
import numpy as np
import pytest

from room_wall_split import (
    build_room_label_map,
    find_interior_segments,
    inward_normal,
    segment_traces_exterior,
    split_exterior_walls_by_room,
    walk_wall_and_split_by_room,
)

pytestmark = pytest.mark.unit

PPU = 18.0
IMAGE_SHAPE = (600, 900)
# Building outer faces (100,100)-(800,500); wall thickness 18 px (1 ft)
OUTER = (100, 100, 800, 500)
INNER = (118, 118, 782, 482)
PARTITION_X = 452  # centerline; strokes at 448 / 457


def _contour():
    x0, y0, x1, y1 = OUTER
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
    return pts.reshape(-1, 1, 2)


def _wall_mask(door=True):
    h, w = IMAGE_SHAPE
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, OUTER[:2], OUTER[2:], 255, 3)
    cv2.rectangle(mask, INNER[:2], INNER[2:], 255, 3)
    cv2.line(mask, (448, INNER[1]), (448, INNER[3]), 255, 3)
    cv2.line(mask, (457, INNER[1]), (457, INNER[3]), 255, 3)
    if door:
        mask[280:316, 444:462] = 0  # 2 ft door gap in the partition
    return mask


EXTERIOR_SEGS = [
    (100, 109, 800, 109),   # north centerline
    (791, 100, 791, 500),   # east
    (800, 491, 100, 491),   # south
    (109, 500, 109, 100),   # west
]
PARTITION_SEG = (PARTITION_X, 118, PARTITION_X, 482)


def _label_map(door=True):
    return build_room_label_map(
        _contour(),
        [PARTITION_SEG],
        _wall_mask(door=door),
        IMAGE_SHAPE,
        wall_thickness_px=9,
        min_room_area_px=int(25 * PPU ** 2),
        endpoint_extend_px=18,
        close_kernel_px=45,  # 2.5 ft doorway close
    )


class TestBuildRoomLabelMap:
    def test_two_rooms_detected(self):
        labels, rooms = _label_map()
        assert len(rooms) == 2
        assert labels.max() == 2

    def test_rooms_left_and_right_of_partition(self):
        _, rooms = _label_map()
        centroids_x = sorted(r["centroid_px"][0] for r in rooms)
        assert centroids_x[0] < PARTITION_X < centroids_x[1]

    def test_door_gap_sealed_by_close(self):
        # Partition detected as two pieces stopping at a 3 ft door gap:
        # with the doorway close kernel the gap is sealed (2 rooms),
        # without it the rooms merge through the door (1 room).
        mask = _wall_mask(door=False)
        mask[280:334, 444:462] = 0  # 3 ft door gap in partition ink
        pieces = [
            (PARTITION_X, INNER[1], PARTITION_X, 280),
            (PARTITION_X, 334, PARTITION_X, INNER[3]),
        ]
        kwargs = dict(
            wall_thickness_px=9, min_room_area_px=int(25 * PPU ** 2),
            endpoint_extend_px=8,
        )
        _, rooms_closed = build_room_label_map(
            _contour(), pieces, mask, IMAGE_SHAPE, close_kernel_px=45, **kwargs,
        )
        _, rooms_open = build_room_label_map(
            _contour(), pieces, mask, IMAGE_SHAPE, close_kernel_px=1, **kwargs,
        )
        assert len(rooms_closed) == 2
        assert len(rooms_open) == 1

    def test_room_records_shape(self):
        _, rooms = _label_map()
        for r in rooms:
            assert set(r) >= {"id", "area_px", "centroid_px", "bbox_px"}
            assert r["area_px"] > 0


class TestInwardNormal:
    def test_north_wall_points_down(self):
        nx, ny = inward_normal(100, 109, 800, 109, _contour())
        assert ny > 0.9

    def test_west_wall_points_right(self):
        nx, ny = inward_normal(109, 500, 109, 100, _contour())
        assert nx > 0.9


class TestWalkWall:
    def test_north_wall_split_into_two_rooms(self):
        labels, _ = _label_map()
        runs = walk_wall_and_split_by_room(
            EXTERIOR_SEGS[0], labels, _contour(),
            probe_offsets_px=[13, 22, 36, 54],
            min_segment_px=4 * PPU,
        )
        room_runs = [r for r in runs if r[2] > 0]
        assert len(room_runs) == 2
        assert room_runs[0][2] != room_runs[1][2]

    def test_west_wall_single_room(self):
        labels, _ = _label_map()
        runs = walk_wall_and_split_by_room(
            EXTERIOR_SEGS[3], labels, _contour(),
            probe_offsets_px=[13, 22, 36, 54],
            min_segment_px=4 * PPU,
        )
        room_labels = {r[2] for r in runs if r[2] > 0}
        assert len(room_labels) == 1


class TestInteriorSegmentFilters:
    def test_partition_kept(self):
        kept = find_interior_segments(
            [PARTITION_SEG], EXTERIOR_SEGS, _contour(), near_tol=15,
        )
        assert kept == [PARTITION_SEG]

    def test_outside_segment_dropped(self):
        kept = find_interior_segments(
            [(20, 50, 90, 50)], EXTERIOR_SEGS, _contour(), near_tol=15,
        )
        assert kept == []

    def test_exterior_tracing_segment_dropped(self):
        tracing = (120, 112, 780, 112)
        assert segment_traces_exterior(tracing, EXTERIOR_SEGS, near_tol=15)
        kept = find_interior_segments(
            [tracing], EXTERIOR_SEGS, _contour(), near_tol=15,
        )
        assert kept == []

    def test_floating_interior_segment_dropped(self):
        # Inside the footprint but not touching any exterior wall
        floating = (300, 250, 600, 250)
        kept = find_interior_segments(
            [floating], EXTERIOR_SEGS, _contour(), near_tol=15,
        )
        assert kept == []


class TestSplitExteriorWallsByRoom:
    def _run(self):
        return split_exterior_walls_by_room(
            EXTERIOR_SEGS,
            wall_pair_mask=_wall_mask(),
            contour=_contour(),
            footprint_bbox=[100, 100, 800, 500],
            image_shape=(*IMAGE_SHAPE, 3),
            px_per_unit=PPU,
            unit_label="ft",
            interior_segments=[PARTITION_SEG],
        )

    def test_two_rooms_with_areas(self):
        rooms, _ = self._run()
        assert len(rooms) == 2
        for r in rooms:
            assert r["area_raw"] > 25.0
            assert r["area"].endswith("ft²")

    def test_sub_segment_metadata(self):
        _, subs = self._run()
        assert subs
        for s in subs:
            assert s["is_exterior"] is True
            assert s["parent_wall_id"].startswith("w")
            assert s["id"].startswith(s["parent_wall_id"] + ".s")
            assert s["segment_index"] >= 1
            assert s["facing"] in {"North", "South", "East", "West"}

    def test_north_wall_split_by_room(self):
        _, subs = self._run()
        north_subs = [s for s in subs if s["parent_wall_id"] == "w1"]
        room_ids = {s["room_id"] for s in north_subs if s["room_id"]}
        assert len(north_subs) == 2
        assert len(room_ids) == 2

    def test_sub_lengths_sum_to_parent(self):
        _, subs = self._run()
        north_subs = [s for s in subs if s["parent_wall_id"] == "w1"]
        total = sum(s["length_raw"] for s in north_subs)
        parent_len = 700 / PPU
        assert total == pytest.approx(parent_len, abs=0.5)

    def test_empty_input(self):
        rooms, subs = split_exterior_walls_by_room(
            [], _wall_mask(), _contour(), [100, 100, 800, 500],
            (*IMAGE_SHAPE, 3), PPU, "ft",
        )
        assert rooms == [] and subs == []
