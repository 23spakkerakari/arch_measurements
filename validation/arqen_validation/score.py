"""Score extracted geometry against ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .closure import compute_closure
from .matchers import (
    dimension_score,
    greedy_match,
    label_score,
    opening_score,
    room_score,
    wall_score,
)
from .metrics import build_report
from .normalize import CATEGORIES, extract_wall_windows_from_prediction, normalize_document

DEFAULT_THRESHOLDS = {
    "rooms": 0.50,
    "walls": 0.55,
    "doors": 0.45,
    "windows": 0.45,
    "labels": 0.70,
    "dimensions": 0.75,
}


def _canvas_size(doc: dict) -> tuple[int, int]:
    size = doc.get("image_size_px") or [4096, 4096]
    return int(size[0]), int(size[1])


def _prepare_prediction(raw: dict[str, Any]) -> dict[str, Any]:
    pred = normalize_document(raw)
    if not pred.get("windows") and raw.get("walls"):
        pred["windows"] = extract_wall_windows_from_prediction(raw)
    return pred


def score_prediction(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    *,
    case_id: str | None = None,
    thresholds: dict[str, float] | None = None,
    closure_tolerance_px: float | None = None,
) -> dict:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    gt = normalize_document(ground_truth)
    pred = _prepare_prediction(prediction)
    canvas = _canvas_size(gt if gt.get("image_size_px") else pred)
    case = case_id or gt.get("id") or pred.get("id") or "unknown"

    results = []

    results.append(greedy_match(
        "rooms",
        gt.get("rooms", []),
        pred.get("rooms", []),
        lambda g, p: room_score(g, p, canvas),
        thresholds["rooms"],
    ))

    results.append(greedy_match(
        "walls",
        gt.get("walls", []),
        pred.get("walls", []),
        wall_score,
        thresholds["walls"],
    ))

    results.append(greedy_match(
        "doors",
        gt.get("doors", []),
        pred.get("doors", []),
        opening_score,
        thresholds["doors"],
    ))

    results.append(greedy_match(
        "windows",
        gt.get("windows", []),
        pred.get("windows", []),
        opening_score,
        thresholds["windows"],
    ))

    results.append(greedy_match(
        "labels",
        gt.get("labels", []),
        pred.get("labels", []),
        lambda g, p: label_score(g, p, canvas),
        thresholds["labels"],
    ))

    results.append(greedy_match(
        "dimensions",
        gt.get("dimensions", []),
        pred.get("dimensions", []),
        dimension_score,
        thresholds["dimensions"],
    ))

    report = build_report(case, results)
    report["closure"] = compute_closure(
        gt, pred, prediction_raw=prediction, tol_px=closure_tolerance_px,
    )
    return report


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def score_case(
    case_dir: Path,
    prediction_path: Path | None = None,
    *,
    output_path: Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict:
    case_dir = Path(case_dir)
    manifest_path = case_dir / "manifest.json"
    gt_path = case_dir / "ground_truth.json"

    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth: {gt_path}")

    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    ground_truth = load_json(gt_path)
    if manifest.get("image_size_px") and not ground_truth.get("image_size_px"):
        ground_truth["image_size_px"] = manifest["image_size_px"]
    if manifest.get("id") and not ground_truth.get("id"):
        ground_truth["id"] = manifest["id"]

    if prediction_path is None:
        prediction_path = case_dir / "prediction.json"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing prediction JSON: {prediction_path}. "
            "Run the pipeline first or pass --prediction."
        )

    prediction = load_json(prediction_path)
    report = score_prediction(
        ground_truth,
        prediction,
        case_id=manifest.get("id") or case_dir.name,
        thresholds=thresholds,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def score_all_cases(
    cases_root: Path,
    *,
    output_dir: Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict:
    cases_root = Path(cases_root)
    case_dirs = sorted(
        p for p in cases_root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "ground_truth.json").exists()
    )

    reports = []
    for case_dir in case_dirs:
        pred_path = case_dir / "prediction.json"
        if not pred_path.exists():
            continue
        report = score_case(case_dir, pred_path, thresholds=thresholds)
        reports.append(report)
        if output_dir:
            out = Path(output_dir) / f"{case_dir.name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

    return {
        "case_count": len(reports),
        "cases": [r["case_id"] for r in reports],
        "reports": reports,
    }
