"""Stage [11] Measurement and facing — preprocess.py geometry helpers."""

import numpy as np
import pytest

from preprocess import (
    angle_to_facing,
    assign_segment_facings,
    bbox_edge_facing,
    build_wall_adjacency,
    classify_interior_facing,
    measure_walls,
    outward_facing,
    pixel_length,
    point_in_footprint,
    vector_to_facing,
    wall_angle_deg,
)

pytestmark = pytest.mark.unit

# Square footprint, image coords (y down)
CONTOUR = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.int32)
BBOX = [100, 100, 500, 400]
EDGES = {
    "north": (100, 100, 500, 100),
    "east": (500, 100, 500, 400),
    "south": (500, 400, 100, 400),
    "west": (100, 400, 100, 100),
}


class TestAngles:
    def test_wall_angle_up_is_zero(self):
        assert wall_angle_deg(0, 0, 0, -10) == pytest.approx(0.0)

    def test_wall_angle_right_is_90(self):
        assert wall_angle_deg(0, 0, 10, 0) == pytest.approx(90.0)

    def test_wall_angle_down_is_180(self):
        assert wall_angle_deg(0, 0, 0, 10) == pytest.approx(180.0)

    @pytest.mark.parametrize("angle,facing", [
        (0, "North"), (44.9, "North"), (315, "North"), (359, "North"),
        (45, "East"), (134.9, "East"),
        (135, "South"), (224.9, "South"),
        (225, "West"), (314.9, "West"),
    ])
    def test_angle_to_facing_boundaries(self, angle, facing):
        assert angle_to_facing(angle) == facing

    @pytest.mark.parametrize("nx,ny,facing", [
        (0, -1, "North"), (1, 0, "East"), (0, 1, "South"), (-1, 0, "West"),
    ])
    def test_vector_to_facing(self, nx, ny, facing):
        assert vector_to_facing(nx, ny) == facing


class TestFootprintProbes:
    def test_point_inside(self):
        assert point_in_footprint(300, 250, CONTOUR)

    def test_point_outside(self):
        assert not point_in_footprint(50, 50, CONTOUR)

    def test_outward_facing_per_edge(self):
        assert outward_facing(*EDGES["north"], CONTOUR) == "North"
        assert outward_facing(*EDGES["east"], CONTOUR) == "East"
        assert outward_facing(*EDGES["south"], CONTOUR) == "South"
        assert outward_facing(*EDGES["west"], CONTOUR) == "West"

    def test_outward_facing_interior_returns_none(self):
        assert outward_facing(200, 250, 400, 250, CONTOUR) is None

    def test_bbox_edge_facing(self):
        assert bbox_edge_facing(*EDGES["north"], BBOX, edge_tol_px=12) == "North"
        assert bbox_edge_facing(*EDGES["west"], BBOX, edge_tol_px=12) == "West"
        assert bbox_edge_facing(200, 250, 400, 250, BBOX, edge_tol_px=12) is None


class TestAdjacency:
    def test_corner_link(self):
        segs = [(0, 0, 100, 0), (100, 0, 100, 100)]
        adj = build_wall_adjacency(segs, corner_tol_px=10)
        assert adj[0] == [1]
        assert adj[1] == [0]

    def test_t_junction_link(self):
        segs = [(100, 0, 100, 100), (0, 50, 100, 50)]
        adj = build_wall_adjacency(segs, corner_tol_px=10)
        assert 1 in adj[0]

    def test_distant_segments_not_linked(self):
        segs = [(0, 0, 100, 0), (0, 500, 100, 500)]
        adj = build_wall_adjacency(segs, corner_tol_px=10)
        assert adj == [[], []]


class TestFacingAssignment:
    def test_rectangle_facings(self):
        segs = [EDGES["north"], EDGES["east"], EDGES["south"], EDGES["west"]]
        facings = assign_segment_facings(segs, CONTOUR, BBOX, px_per_unit=18.0)
        assert facings == ["North", "East", "South", "West"]

    def test_interior_partition_inherits_neighbor(self):
        # Vertical partition T-junctioned into north and south walls
        segs = [
            EDGES["north"], EDGES["east"], EDGES["south"], EDGES["west"],
            (300, 100, 300, 400),
        ]
        facings = assign_segment_facings(segs, CONTOUR, BBOX, px_per_unit=18.0)
        assert facings[4] in ("East", "West")

    def test_classify_interior_horizontal_from_neighbor(self):
        segs = [(100, 200, 300, 200), (100, 100, 300, 100)]
        adjacency = [[1], [0]]
        facings = [None, "North"]
        out = classify_interior_facing(0, segs[0], adjacency, facings, BBOX)
        assert out == "North"


class TestMeasureWalls:
    def test_basic_record(self):
        walls = measure_walls(
            [(100, 100, 500, 100)], px_per_unit=18.0, unit_label="ft",
            contour=CONTOUR, footprint_bbox=BBOX,
        )
        assert len(walls) == 1
        w = walls[0]
        assert w["id"] == "w1"
        assert w["facing"] == "North"
        assert w["length_raw"] == pytest.approx(400 / 18.0, abs=0.01)
        assert w["length"].endswith(" ft")
        assert w["px_coords"] == [100, 100, 500, 100]

    def test_pixel_length(self):
        assert pixel_length(0, 0, 3, 4) == pytest.approx(5.0)
