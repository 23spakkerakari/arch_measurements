"""Geometry helpers for validation scoring."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np


def _as_xy_pairs(points: Sequence) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected Nx2 coordinates, got shape {arr.shape}")
    return arr


def bbox_from_points(points: Sequence) -> list[float]:
    pts = _as_xy_pairs(points)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [float(x0), float(y0), float(x1), float(y1)]


def normalize_bbox(bbox: Sequence[float]) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def bbox_area(bbox: Sequence[float]) -> float:
    x0, y0, x1, y1 = normalize_bbox(bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = normalize_bbox(a)
    bx0, by0, bx1, by1 = normalize_bbox(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def polygon_iou(
    poly_a: Sequence,
    poly_b: Sequence,
    canvas_size: tuple[int, int],
) -> float:
    w, h = canvas_size
    if w <= 0 or h <= 0:
        return 0.0
    pa = np.round(_as_xy_pairs(poly_a)).astype(np.int32)
    pb = np.round(_as_xy_pairs(poly_b)).astype(np.int32)
    if len(pa) < 3 or len(pb) < 3:
        return bbox_iou(bbox_from_points(pa), bbox_from_points(pb))

    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_a, [pa], 1)
    cv2.fillPoly(mask_b, [pb], 1)
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return inter / union if union > 0 else 0.0


def normalize_segment(coords: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in coords[:4])
    if (x2, y2) < (x1, y1):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return x1, y1, x2, y2


def segment_length(coords: Sequence[float]) -> float:
    x1, y1, x2, y2 = normalize_segment(coords)
    return math.hypot(x2 - x1, y2 - y1)


def segment_angle_deg(coords: Sequence[float]) -> float:
    x1, y1, x2, y2 = normalize_segment(coords)
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def segment_overlap_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """1D overlap IoU for nearly colinear axis-aligned or diagonal segments."""
    ax1, ay1, ax2, ay2 = normalize_segment(a)
    bx1, by1, bx2, by2 = normalize_segment(b)

    angle_a = segment_angle_deg(a)
    angle_b = segment_angle_deg(b)
    angle_delta = abs(angle_a - angle_b)
    angle_delta = min(angle_delta, 180.0 - angle_delta)
    if angle_delta > 12.0:
        return 0.0

    ac = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    bc = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    perp_dist = abs((bc[0] - ac[0]) * math.sin(math.radians(angle_a))
                    - (bc[1] - ac[1]) * math.cos(math.radians(angle_a)))
    max_len = max(segment_length(a), segment_length(b), 1.0)
    if perp_dist > max(12.0, 0.08 * max_len):
        return 0.0

    ux, uy = math.cos(math.radians(angle_a)), math.sin(math.radians(angle_a))
    a_proj = sorted([ax1 * ux + ay1 * uy, ax2 * ux + ay2 * uy])
    b_proj = sorted([bx1 * ux + by1 * uy, bx2 * ux + by2 * uy])
    inter = max(0.0, min(a_proj[1], b_proj[1]) - max(a_proj[0], b_proj[0]))
    union = max(a_proj[1], b_proj[1]) - min(a_proj[0], b_proj[0])
    union = max(union, inter, 1e-6)
    return inter / union


def point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def center_from_bbox(bbox: Sequence[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = normalize_bbox(bbox)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def object_center(obj: dict) -> tuple[float, float] | None:
    if obj.get("center_px"):
        c = obj["center_px"]
        return float(c[0]), float(c[1])
    if obj.get("centroid_px"):
        c = obj["centroid_px"]
        return float(c[0]), float(c[1])
    if obj.get("bbox_px"):
        return center_from_bbox(obj["bbox_px"])
    if obj.get("px_coords"):
        x1, y1, x2, y2 = normalize_segment(obj["px_coords"])
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if obj.get("polygon_px"):
        pts = _as_xy_pairs(obj["polygon_px"])
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return None


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().upper().split())


def value_within_tolerance(
    predicted: float | None,
    expected: float | None,
    rel_tol: float = 0.05,
    abs_tol: float = 0.25,
) -> bool:
    if predicted is None or expected is None:
        return False
    if abs(expected) < abs_tol:
        return abs(predicted - expected) <= abs_tol
    return abs(predicted - expected) <= max(abs_tol, rel_tol * abs(expected))
