"""Greedy bipartite matchers per geometry category."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .closure import point_to_segment_distance
from .geometry import (
    bbox_iou,
    normalize_text,
    object_center,
    point_distance,
    polygon_iou,
    segment_overlap_iou,
    value_within_tolerance,
)


@dataclass
class MatchResult:
    category: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float | None
    matches: list[dict] = field(default_factory=list)
    missing_objects: list[dict] = field(default_factory=list)
    false_positive_objects: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "counts": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
            },
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_iou": round(self.mean_iou, 4) if self.mean_iou is not None else None,
            "matches": self.matches,
            "missing_objects": self.missing_objects,
            "false_positives": self.false_positive_objects,
        }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def greedy_match(
    category: str,
    ground_truth: list[dict],
    predicted: list[dict],
    score_fn: Callable[[dict, dict], float],
    threshold: float,
) -> MatchResult:
    if not ground_truth and not predicted:
        return MatchResult(category, 0, 0, 0, 1.0, 1.0, 1.0, None)

    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(ground_truth):
        for pi, pred in enumerate(predicted):
            score = score_fn(gt, pred)
            if score >= threshold:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[dict] = []
    ious: list[float] = []

    for score, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        gt = ground_truth[gi]
        pred = predicted[pi]
        matches.append({
            "ground_truth_id": gt.get("id"),
            "predicted_id": pred.get("id"),
            "score": round(score, 4),
            "ground_truth": gt,
            "predicted": pred,
        })
        ious.append(score)

    tp = len(matches)
    fp = len(predicted) - len(matched_pred)
    fn = len(ground_truth) - len(matched_gt)
    precision, recall, f1 = _prf(tp, fp, fn)

    missing = [ground_truth[i] for i in range(len(ground_truth)) if i not in matched_gt]
    false_pos = [predicted[i] for i in range(len(predicted)) if i not in matched_pred]

    return MatchResult(
        category=category,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=(sum(ious) / len(ious)) if ious else None,
        matches=matches,
        missing_objects=missing,
        false_positive_objects=false_pos,
    )


def room_score(gt: dict, pred: dict, canvas_size: tuple[int, int]) -> float:
    if gt.get("polygon_px") and pred.get("polygon_px"):
        return polygon_iou(gt["polygon_px"], pred["polygon_px"], canvas_size)
    gt_bbox = gt.get("bbox_px")
    pred_bbox = pred.get("bbox_px")
    if gt_bbox and pred_bbox:
        return bbox_iou(gt_bbox, pred_bbox)
    gt_c = object_center(gt)
    pred_c = object_center(pred)
    if gt_c and pred_c:
        dist = point_distance(gt_c, pred_c)
        return max(0.0, 1.0 - dist / 80.0)
    return 0.0


def wall_score(gt: dict, pred: dict) -> float:
    gt_coords = gt.get("px_coords")
    pred_coords = pred.get("px_coords")
    if not gt_coords or not pred_coords:
        return 0.0
    return segment_overlap_iou(gt_coords, pred_coords)


def _segments_of(walls: list[dict]) -> list[tuple[float, float, float, float]]:
    segs = []
    for w in walls or []:
        c = w.get("px_coords")
        if c and len(c) >= 4:
            segs.append(tuple(float(v) for v in c[:4]))
    return segs


def _covered_fraction(
    seg: tuple[float, float, float, float],
    others: list[tuple[float, float, float, float]],
    tol_px: float,
    samples: int = 64,
) -> float:
    """Fraction of seg's length lying within tol_px of any other segment."""
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    if not others:
        return 0.0
    covered = 0
    for k in range(samples):
        t = (k + 0.5) / samples
        px, py = x1 + t * dx, y1 + t * dy
        if any(point_to_segment_distance(px, py, o) <= tol_px for o in others):
            covered += 1
    return covered / samples


def wall_coverage_metrics(
    gt_walls: list[dict],
    pred_walls: list[dict],
    tol_px: float,
) -> dict:
    """Length-weighted span coverage between GT and predicted wall sets.

    Complements the strict 1:1 greedy match, which under-counts when the
    pipeline legitimately emits per-room sub-segments of one GT wall (or one
    merged run covering several GT walls). Recall = fraction of total GT wall
    length within tol_px of any predicted wall; precision = fraction of total
    predicted length within tol_px of any GT wall.
    """
    gt_segs = _segments_of(gt_walls)
    pred_segs = _segments_of(pred_walls)

    def _aggregate(subject: list, others: list) -> float | None:
        total = covered = 0.0
        for seg in subject:
            length = math.hypot(seg[2] - seg[0], seg[3] - seg[1])
            if length <= 0:
                continue
            total += length
            covered += length * _covered_fraction(seg, others, tol_px)
        return (covered / total) if total else None

    recall = _aggregate(gt_segs, pred_segs)
    precision = _aggregate(pred_segs, gt_segs)
    if precision is None and recall is None:
        precision = recall = 1.0  # nothing expected, nothing predicted
    else:
        precision = 1.0 if precision is None else precision
        recall = 1.0 if recall is None else recall
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tolerance_px": round(float(tol_px), 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def opening_score(gt: dict, pred: dict, center_tol_px: float = 40.0) -> float:
    gt_bbox = gt.get("bbox_px")
    pred_bbox = pred.get("bbox_px")
    if gt_bbox and pred_bbox:
        return bbox_iou(gt_bbox, pred_bbox)
    gt_c = object_center(gt)
    pred_c = object_center(pred)
    if gt_c and pred_c:
        dist = point_distance(gt_c, pred_c)
        if dist <= center_tol_px:
            return 1.0 - dist / center_tol_px
    return 0.0


def label_score(gt: dict, pred: dict, canvas_size: tuple[int, int]) -> float:
    gt_text = normalize_text(gt.get("text"))
    pred_text = normalize_text(pred.get("text"))
    text_score = 1.0 if gt_text and gt_text == pred_text else 0.0

    if gt.get("room_id") and pred.get("room_id") and gt["room_id"] == pred["room_id"]:
        text_score = max(text_score, 0.85)

    spatial = 0.0
    if gt.get("bbox_px") and pred.get("bbox_px"):
        spatial = bbox_iou(gt["bbox_px"], pred["bbox_px"])
    else:
        gt_c = object_center(gt)
        pred_c = object_center(pred)
        if gt_c and pred_c:
            dist = point_distance(gt_c, pred_c)
            spatial = max(0.0, 1.0 - dist / 120.0)

    if text_score >= 1.0:
        return max(text_score, spatial)
    return 0.7 * text_score + 0.3 * spatial


def dimension_score(gt: dict, pred: dict, rel_tol: float = 0.05) -> float:
    gt_val = gt.get("value_raw")
    pred_val = pred.get("value_raw")
    if gt_val is None or pred_val is None:
        return 0.0
    if not value_within_tolerance(pred_val, gt_val, rel_tol=rel_tol):
        return 0.0

    gt_c = object_center(gt)
    pred_c = object_center(pred)
    if gt_c and pred_c:
        dist = point_distance(gt_c, pred_c)
        if dist > 150.0:
            return 0.5
        return max(0.75, 1.0 - dist / 300.0)
    return 0.9
