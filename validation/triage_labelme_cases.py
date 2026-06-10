#!/usr/bin/env python3
"""Classify LabelMe validation cases by failure mode for targeted fixes."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _annotation_span_px(gt: dict) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for room in gt.get("rooms") or []:
        if room.get("bbox_px"):
            x0, y0, x1, y1 = room["bbox_px"]
            xs.extend([x0, x1])
            ys.extend([y0, y1])
    for wall in gt.get("walls") or []:
        c = wall.get("px_coords") or []
        if len(c) >= 4:
            xs.extend([c[0], c[2]])
            ys.extend([c[1], c[3]])
    if not xs:
        size = gt.get("image_size_px") or [0, 0]
        return 0.85 * max(size[0], size[1])
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _calibration_codes(pred: dict | None) -> set[str]:
    if not pred:
        return set()
    cal = pred.get("calibration") or {}
    return {i.get("code", "") for i in cal.get("issues") or []}


def classify_case(
    case_id: str,
    manifest: dict,
    gt: dict,
    report: dict | None,
    pred: dict | None,
) -> dict:
    """Return metrics + bucket tags for one case."""
    cats = (report or {}).get("categories") or {}
    rooms = cats.get("rooms") or {}
    walls = cats.get("walls") or {}
    doors = cats.get("doors") or {}
    wall_cov = walls.get("coverage") or {}

    gt_rooms = len(gt.get("rooms") or [])
    gt_walls = len(gt.get("walls") or [])
    pred_rooms = len((pred or {}).get("rooms") or [])
    pred_walls = len((pred or {}).get("walls") or [])

    span_px = _annotation_span_px(gt)
    inferred = manifest.get("inferred_px_per_ft")
    cal_codes = _calibration_codes(pred)

    tags: list[str] = []

    if (
        gt_walls < 15
        and pred_walls > 0
        and gt_walls < 0.3 * pred_walls
    ):
        tags.append("sparse_gt")

    if gt_rooms > 0 and pred_rooms > 1.5 * gt_rooms and rooms.get("recall", 1) < 0.5:
        tags.append("phantom")

    if (
        inferred == 72.0
        and span_px > 3000
    ) or cal_codes & {"footprint_span_low", "footprint_span_high"}:
        tags.append("calibration_suspect")

    wall_strict_r = walls.get("recall", 0) or 0
    wall_cov_r = wall_cov.get("recall", 0) or 0
    if wall_cov_r < 0.45 and wall_strict_r < 0.5 and "sparse_gt" not in tags:
        tags.append("missing_interior")

    if (
        wall_strict_r >= 0.8
        and wall_cov_r >= 0.7
        and (walls.get("precision") or 0) < 0.35
    ):
        tags.append("scoring_artifact")

    if not tags:
        tags.append("mixed")

    return {
        "case_id": case_id,
        "bucket": tags[0],
        "tags": tags,
        "gt_rooms": gt_rooms,
        "pred_rooms": pred_rooms,
        "gt_walls": gt_walls,
        "pred_walls": pred_walls,
        "room_recall": rooms.get("recall"),
        "room_precision": rooms.get("precision"),
        "wall_strict_recall": wall_strict_r,
        "wall_strict_precision": walls.get("precision"),
        "wall_cov_recall": wall_cov_r,
        "wall_cov_precision": wall_cov.get("precision"),
        "door_tp": doors.get("counts", {}).get("true_positives", 0),
        "door_fn": doors.get("counts", {}).get("false_negatives", 0),
        "inferred_px_per_ft": inferred,
        "span_px": round(span_px, 1),
        "calibration_issues": sorted(cal_codes),
    }


def build_triage(cases_root: Path) -> list[dict]:
    rows = []
    for case_dir in sorted(cases_root.iterdir()):
        if not case_dir.name.startswith("labelme_"):
            continue
        manifest = _load_json(case_dir / "manifest.json") or {}
        gt = _load_json(case_dir / "ground_truth.json") or {}
        report = _load_json(case_dir / "report.json")
        pred = _load_json(case_dir / "prediction.json")
        rows.append(classify_case(case_dir.name, manifest, gt, report, pred))
    return rows


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cov_recalls = [r["wall_cov_recall"] for r in rows if r["wall_cov_recall"] is not None]
    median_cov = statistics.median(cov_recalls) if cov_recalls else 0.0

    lines = [
        "# LabelMe triage",
        "",
        f"Cases: {len(rows)}. Median wall span-coverage recall: **{median_cov:.2f}**.",
        "",
        "| Case | Bucket | wall cov R | wall strict R | room R | pred/gt rooms | "
        "pred/gt walls | px/ft |",
        "|------|--------|------------|---------------|--------|---------------|"
        "---------------|-------|",
    ]
    for r in sorted(rows, key=lambda x: x.get("wall_cov_recall") or 0):
        lines.append(
            f"| {r['case_id']} | {r['bucket']} | "
            f"{r['wall_cov_recall']:.2f} | {r['wall_strict_recall']:.2f} | "
            f"{(r['room_recall'] or 0):.2f} | "
            f"{r['pred_rooms']}/{r['gt_rooms']} | "
            f"{r['pred_walls']}/{r['gt_walls']} | "
            f"{r['inferred_px_per_ft'] or '—'} |"
        )

    lines.extend(["", "## Bucket counts", ""])
    buckets: dict[str, int] = {}
    for r in rows:
        buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
    for b, n in sorted(buckets.items(), key=lambda x: -x[1]):
        lines.append(f"- **{b}**: {n}")

    lines.extend(["", "## Worst 5 (wall span-coverage recall)", ""])
    worst = sorted(rows, key=lambda x: x.get("wall_cov_recall") or 0)[:5]
    for r in worst:
        lines.append(
            f"- `{r['case_id']}` ({r['bucket']}): cov R={r['wall_cov_recall']:.2f}, "
            f"tags={r['tags']}, cal={r['calibration_issues'] or 'none'}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cases_root = ROOT / "cases"
    rows = build_triage(cases_root)
    if not rows:
        print("No labelme_* cases found.")
        return 1

    out = REPORTS / "labelme_triage.md"
    write_markdown(rows, out)
    print(f"Wrote {out} ({len(rows)} cases)")

    cov_recalls = [r["wall_cov_recall"] for r in rows]
    print(f"Median wall span-coverage recall: {statistics.median(cov_recalls):.2f}")

    print("\nWorst 5:")
    for r in sorted(rows, key=lambda x: x["wall_cov_recall"])[:5]:
        print(
            f"  {r['case_id']:<22} bucket={r['bucket']:<20} "
            f"cov_R={r['wall_cov_recall']:.2f} strict_R={r['wall_strict_recall']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
