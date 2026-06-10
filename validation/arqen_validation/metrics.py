"""Aggregate category metrics into a validation report."""

from __future__ import annotations

from .matchers import MatchResult


def macro_average(results: list[MatchResult]) -> dict:
    if not results:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = sum(r.precision for r in results) / len(results)
    recall = sum(r.recall for r in results) / len(results)
    f1 = sum(r.f1 for r in results) / len(results)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def micro_average(results: list[MatchResult]) -> dict:
    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def build_report(case_id: str, category_results: list[MatchResult]) -> dict:
    by_category = {r.category: r.to_dict() for r in category_results}
    iou_values = [r.mean_iou for r in category_results if r.mean_iou is not None]

    return {
        "case_id": case_id,
        "summary": {
            "macro": macro_average(category_results),
            "micro": micro_average(category_results),
            "mean_iou": round(sum(iou_values) / len(iou_values), 4) if iou_values else None,
        },
        "categories": by_category,
    }
