"""Automated geometry scoring against ground-truth floor plans."""

from .score import score_case, score_prediction

__all__ = ["score_case", "score_prediction"]
