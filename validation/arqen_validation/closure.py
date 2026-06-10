"""Space-boundary closure metrics.

Three complementary measures of whether detected geometry encloses space:

1. ``wall_network_closure`` (prediction-only) — every wall endpoint should
   terminate at another wall (corner or T-junction). Dangling endpoints mean
   the wall network cannot enclose space.
2. ``room_boundary_closure`` (vs ground truth) — fraction of each GT room
   perimeter that lies within tolerance of a predicted wall. A room whose
   perimeter coverage is >= ``closed_threshold`` is *closed*.
3. ``interior_coverage`` (prediction-only) — fraction of the detected
   footprint interior accounted for by detected room cells.
"""

from __future__ import annotations

import math
from typing import Sequence

from .geometry import normalize_bbox, normalize_segment

DEFAULT_CLOSED_THRESHOLD = 0.95
DEFAULT_SAMPLES_PER_ROOM = 200
_MAX_REPORTED_DANGLING = 25


def point_to_segment_distance(px: float, py: float, seg: Sequence[float]) -> float:
    """Euclidean distance from a point to a finite segment."""
    x1, y1, x2, y2 = (float(v) for v in seg[:4])
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def derive_tolerance_px(prediction_raw: dict | None, default: float = 12.0) -> float:
    """Closure tolerance: ~2 ft of slop, derived from calibration when known."""
    px_per_ft = None
    if prediction_raw:
        px_per_ft = prediction_raw.get("px_per_ft")
    if px_per_ft:
        try:
            return max(default, 2.0 * float(px_per_ft))
        except (TypeError, ValueError):
            pass
    return default


def _wall_segments(walls: list[dict]) -> list[tuple[str | None, tuple[float, float, float, float]]]:
    segs = []
    for wall in walls or []:
        coords = wall.get("px_coords")
        if not coords or len(coords) < 4:
            continue
        segs.append((wall.get("id"), normalize_segment(coords)))
    return segs


def wall_network_closure(walls: list[dict], tol_px: float) -> dict:
    """Endpoint connectivity of the predicted wall network.

    An endpoint is *closed* when it lies within ``tol_px`` of another wall's
    endpoint or body (T-junction). Reports the dangling-endpoint rate.
    """
    segs = _wall_segments(walls)
    if not segs:
        return {
            "wall_count": 0,
            "endpoint_count": 0,
            "closed_endpoints": 0,
            "dangling_endpoints": 0,
            "closure_rate": None,
            "dangling_endpoint_rate": None,
            "dangling": [],
        }

    closed = 0
    dangling: list[dict] = []
    for i, (wall_id, seg) in enumerate(segs):
        x1, y1, x2, y2 = seg
        for endpoint in ((x1, y1), (x2, y2)):
            ex, ey = endpoint
            is_closed = False
            for j, (_, other) in enumerate(segs):
                if i == j:
                    continue
                if point_to_segment_distance(ex, ey, other) <= tol_px:
                    is_closed = True
                    break
            if is_closed:
                closed += 1
            elif len(dangling) < _MAX_REPORTED_DANGLING:
                dangling.append({"wall_id": wall_id, "endpoint_px": [round(ex, 1), round(ey, 1)]})

    endpoint_count = 2 * len(segs)
    dangling_count = endpoint_count - closed
    return {
        "wall_count": len(segs),
        "endpoint_count": endpoint_count,
        "closed_endpoints": closed,
        "dangling_endpoints": dangling_count,
        "closure_rate": round(closed / endpoint_count, 4),
        "dangling_endpoint_rate": round(dangling_count / endpoint_count, 4),
        "dangling": dangling,
    }


def _room_perimeter(room: dict) -> list[tuple[float, float]] | None:
    """Closed perimeter polyline (vertex list, implicitly closed) for a room."""
    polygon = room.get("polygon_px")
    if polygon and len(polygon) >= 3:
        return [(float(p[0]), float(p[1])) for p in polygon]
    bbox = room.get("bbox_px")
    if bbox and len(bbox) >= 4:
        x0, y0, x1, y1 = normalize_bbox(bbox)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return None
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return None


def _sample_perimeter(vertices: list[tuple[float, float]], n_samples: int) -> list[tuple[float, float]]:
    edges = []
    total = 0.0
    count = len(vertices)
    for i in range(count):
        a = vertices[i]
        b = vertices[(i + 1) % count]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length > 0:
            edges.append((a, b, length))
            total += length
    if total <= 0:
        return []

    samples: list[tuple[float, float]] = []
    for a, b, length in edges:
        n_edge = max(2, int(round(n_samples * length / total)))
        for k in range(n_edge):
            t = (k + 0.5) / n_edge
            samples.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return samples


def room_boundary_closure(
    gt_rooms: list[dict],
    pred_walls: list[dict],
    tol_px: float,
    closed_threshold: float = DEFAULT_CLOSED_THRESHOLD,
    samples_per_room: int = DEFAULT_SAMPLES_PER_ROOM,
) -> dict:
    """How much of each ground-truth room perimeter is backed by predicted walls."""
    segs = [seg for _, seg in _wall_segments(pred_walls)]
    per_room: list[dict] = []
    closed_count = 0
    coverages: list[float] = []

    for room in gt_rooms or []:
        vertices = _room_perimeter(room)
        if not vertices:
            continue
        samples = _sample_perimeter(vertices, samples_per_room)
        if not samples:
            continue
        covered = 0
        for sx, sy in samples:
            for seg in segs:
                if point_to_segment_distance(sx, sy, seg) <= tol_px:
                    covered += 1
                    break
        coverage = covered / len(samples)
        is_closed = coverage >= closed_threshold
        if is_closed:
            closed_count += 1
        coverages.append(coverage)
        per_room.append({
            "room_id": room.get("id"),
            "boundary_coverage": round(coverage, 4),
            "closed": is_closed,
        })

    room_count = len(per_room)
    return {
        "room_count": room_count,
        "closed_rooms": closed_count,
        "closure_rate": round(closed_count / room_count, 4) if room_count else None,
        "mean_boundary_coverage": round(sum(coverages) / room_count, 4) if room_count else None,
        "closed_threshold": closed_threshold,
        "per_room": per_room,
    }


def _polygon_area(points: Sequence) -> float:
    """Shoelace area for an [[x, y], ...] vertex list."""
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    count = len(points)
    for i in range(count):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[(i + 1) % count][0]), float(points[(i + 1) % count][1])
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def interior_coverage(prediction_raw: dict | None) -> dict | None:
    """Fraction of footprint interior covered by detected room cells."""
    if not prediction_raw:
        return None

    footprint_area = 0.0
    polygon = prediction_raw.get("footprint_polygon_px")
    if polygon:
        footprint_area = _polygon_area(polygon)
    if footprint_area <= 0 and prediction_raw.get("footprint_bbox_px"):
        x0, y0, x1, y1 = normalize_bbox(prediction_raw["footprint_bbox_px"])
        footprint_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if footprint_area <= 0:
        return None

    rooms_area = 0.0
    for room in prediction_raw.get("rooms", []) or []:
        area_px = room.get("area_px")
        if area_px:
            rooms_area += float(area_px)
        elif room.get("bbox_px"):
            x0, y0, x1, y1 = normalize_bbox(room["bbox_px"])
            rooms_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)

    return {
        "rooms_area_px": round(rooms_area, 1),
        "footprint_area_px": round(footprint_area, 1),
        "coverage": round(min(rooms_area / footprint_area, 1.5), 4),
    }


def compute_closure(
    gt_normalized: dict,
    pred_normalized: dict,
    prediction_raw: dict | None = None,
    tol_px: float | None = None,
    closed_threshold: float = DEFAULT_CLOSED_THRESHOLD,
) -> dict:
    """Full closure report combining all three measures."""
    tolerance = tol_px if tol_px is not None else derive_tolerance_px(prediction_raw)
    pred_walls = pred_normalized.get("walls", [])
    return {
        "tolerance_px": round(tolerance, 2),
        "wall_network": wall_network_closure(pred_walls, tolerance),
        "room_boundary": room_boundary_closure(
            gt_normalized.get("rooms", []),
            pred_walls,
            tolerance,
            closed_threshold=closed_threshold,
        ),
        "interior_coverage": interior_coverage(prediction_raw),
    }
