"""Convert LabelMe JSON annotations into Arqen validation ground_truth.json.

Label conventions (case-insensitive):
  Room   -> rooms[]
  Wall   -> walls[] (polygon/rectangle band -> approximate centerline segment)
  Door   -> doors[]
  Window -> windows[]
  Text   -> labels[] (placeholder text unless shape description is set)

Fixture labels (Toilet, Shower, …) are skipped — not in the Arqen schema.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Map LabelMe label -> Arqen category. Keys are lowercased.
DEFAULT_LABEL_MAP: dict[str, str] = {
    "room": "rooms",
    "wall": "walls",
    "door": "doors",
    "window": "windows",
    "text": "labels",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(label: str) -> str:
    return _SLUG_RE.sub("_", label.strip().lower()).strip("_") or "item"


def _bbox_from_points(points: list) -> list[float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _center_from_bbox(bbox: list[float]) -> list[float]:
    x0, y0, x1, y1 = bbox
    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]


def _polygon_px(points: list) -> list[list[float]]:
    return [[float(p[0]), float(p[1])] for p in points]


def _centerline_from_polygon(points: list) -> list[float] | None:
    """Approximate a thick wall polygon/rectangle as one H/V centerline segment."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 2:
        return None

    if pts.shape[0] == 2:
        x1, y1 = float(pts[0][0]), float(pts[0][1])
        x2, y2 = float(pts[1][0]), float(pts[1][1])
        if math.hypot(x2 - x1, y2 - y1) < 3:
            return None
        return [x1, y1, x2, y2]

    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    if w < 2 and h < 2:
        return None

    ang = math.radians(angle)
    if w >= h:
        half = w / 2.0
        dx = math.cos(ang) * half
        dy = math.sin(ang) * half
    else:
        half = h / 2.0
        ang = math.radians(angle + 90.0)
        dx = math.cos(ang) * half
        dy = math.sin(ang) * half

    x1, y1 = cx - dx, cy - dy
    x2, y2 = cx + dx, cy + dy
    if math.hypot(x2 - x1, y2 - y1) < 3:
        return None

    # Snap near-axis segments so wall overlap scoring aligns with the CV path.
    dx, dy = x2 - x1, y2 - y1
    deg = math.degrees(math.atan2(dy, dx)) % 180
    if deg < 12 or deg > 168:
        y = (y1 + y2) / 2.0
        return [min(x1, x2), y, max(x1, x2), y]
    if 78 < deg < 102:
        x = (x1 + x2) / 2.0
        return [x, min(y1, y2), x, max(y1, y2)]

    return [x1, y1, x2, y2]


def _resolve_image_path(
    labelme_path: Path,
    labelme_doc: dict,
    images_root: Path | None,
) -> Path | None:
    raw = labelme_doc.get("imagePath") or ""
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((labelme_path.parent / p).resolve())
            if images_root:
                candidates.append((images_root / p.name).resolve())
                # LabelMe often uses ..\floor-plan-cropped\NAME.png
                candidates.append((images_root / Path(raw).name).resolve())
    stem = labelme_path.stem
    if images_root:
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            candidates.append(images_root / f"{stem}{ext}")

    for c in candidates:
        if c.exists():
            return c
    return None


def convert_labelme_document(
    labelme_doc: dict,
    case_id: str,
    scale: str = "1in=1ft",
    label_map: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """Return (ground_truth, conversion_report)."""
    label_map = label_map or DEFAULT_LABEL_MAP
    w = int(labelme_doc.get("imageWidth") or 0)
    h = int(labelme_doc.get("imageHeight") or 0)

    gt: dict[str, Any] = {
        "id": case_id,
        "scale": scale,
        "image_size_px": [w, h],
        "rooms": [],
        "walls": [],
        "doors": [],
        "windows": [],
        "labels": [],
        "dimensions": [],
    }
    report = {
        "skipped_labels": {},
        "skipped_shapes": [],
        "wall_centerline_warnings": [],
    }

    counters: dict[str, int] = {cat: 0 for cat in set(label_map.values())}

    for shape in labelme_doc.get("shapes") or []:
        raw_label = str(shape.get("label") or "").strip()
        category = label_map.get(raw_label.lower())
        if not category:
            report["skipped_labels"][raw_label] = (
                report["skipped_labels"].get(raw_label, 0) + 1
            )
            continue

        points = shape.get("points") or []
        if not points:
            report["skipped_shapes"].append({"label": raw_label, "reason": "no points"})
            continue

        shape_type = shape.get("shape_type") or ""
        counters[category] += 1
        idx = counters[category]
        obj_id = f"{category[:-1]}_{idx}" if category.endswith("s") else f"{category}_{idx}"
        if category == "rooms":
            obj_id = f"R{idx}"
        elif category == "walls":
            obj_id = f"w{idx}"

        if category == "rooms":
            entry: dict[str, Any] = {"id": obj_id}
            if shape_type == "rectangle" and len(points) == 2:
                entry["bbox_px"] = _bbox_from_points(points)
            else:
                entry["polygon_px"] = _polygon_px(points)
                entry["bbox_px"] = _bbox_from_points(points)
            gt["rooms"].append(entry)

        elif category == "walls":
            seg = _centerline_from_polygon(points)
            if seg is None:
                report["skipped_shapes"].append({
                    "label": raw_label, "reason": "degenerate wall centerline",
                })
                continue
            if shape_type == "polygon" and len(points) > 4:
                report["wall_centerline_warnings"].append(obj_id)
            gt["walls"].append({"id": obj_id, "px_coords": seg})

        elif category in ("doors", "windows"):
            bbox = _bbox_from_points(points)
            gt[category].append({
                "id": obj_id,
                "bbox_px": bbox,
                "center_px": _center_from_bbox(bbox),
            })

        elif category == "labels":
            text = (shape.get("description") or "").strip() or raw_label
            bbox = _bbox_from_points(points)
            gt["labels"].append({
                "id": obj_id,
                "text": text,
                "bbox_px": bbox,
                "center_px": _center_from_bbox(bbox),
            })

    return gt, report


_SPAN_HYPOTHESES_FT = (30.0, 50.0, 70.0, 100.0, 150.0)
# Plausible building footprint span for scoring hypotheses (ft).
_SPAN_PLAUSIBLE_LO = 15.0
_SPAN_PLAUSIBLE_HI = 400.0
_PX_PER_FT_MIN = 12.0


def _annotation_span_px(gt: dict, image_width: int, image_height: int) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for room in gt.get("rooms") or []:
        if room.get("bbox_px"):
            x0, y0, x1, y1 = room["bbox_px"]
            xs.extend([x0, x1])
            ys.extend([y0, y1])
        elif room.get("polygon_px"):
            for x, y in room["polygon_px"]:
                xs.append(x)
                ys.append(y)
    for wall in gt.get("walls") or []:
        c = wall.get("px_coords") or []
        if len(c) >= 4:
            xs.extend([c[0], c[2]])
            ys.extend([c[1], c[3]])
    if xs and ys:
        return max(max(xs) - min(xs), max(ys) - min(ys))
    return 0.85 * max(image_width, image_height)


def _median_wall_length_px(gt: dict) -> float | None:
    lengths: list[float] = []
    for wall in gt.get("walls") or []:
        c = wall.get("px_coords") or []
        if len(c) < 4:
            continue
        lengths.append(math.hypot(c[2] - c[0], c[3] - c[1]))
    if len(lengths) < 3:
        return None
    lengths.sort()
    return lengths[len(lengths) // 2]


def _px_per_ft_cap(span_px: float) -> float:
    """Large dorm crops can exceed the old 72 px/ft ceiling."""
    return max(72.0, min(120.0, span_px / 25.0))


def _hypothesis_score(span_px: float, assumed_span_ft: float) -> tuple[float, float]:
    """Lower score is better. Returns (score, px_per_ft)."""
    cap = _px_per_ft_cap(span_px)
    px_per_ft = max(_PX_PER_FT_MIN, min(cap, span_px / assumed_span_ft))
    span_ft = span_px / px_per_ft
    score = 0.0
    if span_ft < _SPAN_PLAUSIBLE_LO:
        score += (_SPAN_PLAUSIBLE_LO - span_ft) * 2.0
    elif span_ft > _SPAN_PLAUSIBLE_HI:
        score += (span_ft - _SPAN_PLAUSIBLE_HI) * 2.0
    # Prefer mid-range px/ft (~24–60) for morphological filters.
    target = 40.0
    score += abs(px_per_ft - target) * 0.05
    return score, px_per_ft


def infer_crop_calibration(
    gt: dict,
    image_width: int,
    image_height: int,
    assumed_span_ft: float = 50.0,
) -> tuple[str, int, float, float]:
    """Pick scale/dpi so px_per_ft suits a LabelMe crop (not a full sheet).

    Returns (scale_str, dpi, px_per_ft, chosen_hypothesis_ft).

    Cropped PNGs are already in pixel space. We use ``1in=1ft`` with a synthetic
    DPI equal to the inferred px_per_ft so wall-pair gaps and min-length
    filters land in a sensible range. Multiple span hypotheses are scored against
    a plausible footprint band; a wall-length prior nudges the result when
    enough annotated walls exist.
    """
    span_px = _annotation_span_px(gt, image_width, image_height)

    hypotheses = list(_SPAN_HYPOTHESES_FT)
    if assumed_span_ft not in hypotheses:
        hypotheses.append(assumed_span_ft)

    best_score = float("inf")
    best_px = max(_PX_PER_FT_MIN, min(_px_per_ft_cap(span_px), span_px / 50.0))
    best_hyp = 50.0
    for hyp in hypotheses:
        score, px = _hypothesis_score(span_px, hyp)
        if score < best_score:
            best_score = score
            best_px = px
            best_hyp = hyp

    median_wall_px = _median_wall_length_px(gt)
    if median_wall_px is not None:
        # Typical interior partition ~8 ft; blend 30 % toward wall-derived px/ft.
        wall_px_per_ft = median_wall_px / 8.0
        wall_px_per_ft = max(_PX_PER_FT_MIN, min(_px_per_ft_cap(span_px), wall_px_per_ft))
        best_px = 0.7 * best_px + 0.3 * wall_px_per_ft

    px_per_ft = max(_PX_PER_FT_MIN, min(_px_per_ft_cap(span_px), best_px))
    dpi = int(round(px_per_ft))
    return "1in=1ft", dpi, px_per_ft, best_hyp


def import_labelme_case(
    labelme_path: Path,
    case_dir: Path,
    *,
    images_root: Path | None = None,
    scale: str | None = None,
    dpi: int | None = None,
    doorway_close_ft: float = 2.5,
    label_map: dict[str, str] | None = None,
    copy_image: bool = True,
    assumed_span_ft: float = 50.0,
) -> dict:
    """Create a validation case folder from one LabelMe JSON file."""
    labelme_path = Path(labelme_path)
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    doc = json.loads(labelme_path.read_text(encoding="utf-8"))
    case_id = case_dir.name

    image_src = _resolve_image_path(labelme_path, doc, images_root)
    if image_src is None:
        raise FileNotFoundError(
            f"Could not find image for {labelme_path.name}. "
            f"Set --images-root to the folder containing the PNGs."
        )

    gt, report = convert_labelme_document(
        doc, case_id, scale=(scale or "1in=1ft"), label_map=label_map,
    )

    img_w, img_h = gt["image_size_px"]
    if scale is None or dpi is None:
        auto_scale, auto_dpi, px_per_ft, hyp_ft = infer_crop_calibration(
            gt, img_w, img_h, assumed_span_ft=assumed_span_ft,
        )
        if scale is None:
            scale = auto_scale
        if dpi is None:
            dpi = auto_dpi
        report["inferred_px_per_ft"] = round(px_per_ft, 2)
        report["inferred_dpi"] = dpi
        report["calibration_hypothesis_ft"] = hyp_ft
    gt["scale"] = scale

    image_dest = case_dir / "image.png"
    if copy_image:
        import shutil
        shutil.copy2(image_src, image_dest)

    manifest = {
        "id": case_id,
        "description": (
            f"Imported from LabelMe ({labelme_path.name}). "
            "Verify scale in manifest if lengths matter."
        ),
        "image": "image.png",
        "scale": scale,
        "dpi": dpi,
        "roi": None,
        "doorway_close_ft": doorway_close_ft,
        "image_size_px": gt["image_size_px"],
        "imported_from": str(labelme_path),
        "labelme_image": str(image_src),
        "labelme_crop": True,
        # Full-image ROI disables sheet-margin blanking (crop is already tight).
        "roi": {"x0_pct": 0.0, "y0_pct": 0.0, "x1_pct": 1.0, "y1_pct": 1.0},
    }
    if report.get("inferred_px_per_ft") is not None:
        manifest["inferred_px_per_ft"] = report["inferred_px_per_ft"]
    if report.get("calibration_hypothesis_ft") is not None:
        manifest["calibration_hypothesis_ft"] = report["calibration_hypothesis_ft"]

    (case_dir / "ground_truth.json").write_text(
        json.dumps(gt, indent=2), encoding="utf-8",
    )
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    (case_dir / "labelme_conversion.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )

    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "image": str(image_src),
        "counts": {k: len(gt[k]) for k in (
            "rooms", "walls", "doors", "windows", "labels", "dimensions",
        )},
        "report": report,
    }


def recalibrate_case(case_dir: Path, *, assumed_span_ft: float = 50.0) -> dict:
    """Refresh scale/dpi in manifest from existing ground_truth (no image copy)."""
    case_dir = Path(case_dir)
    gt_path = case_dir / "ground_truth.json"
    manifest_path = case_dir / "manifest.json"
    if not gt_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing ground_truth or manifest in {case_dir}")

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    img_w, img_h = gt.get("image_size_px") or manifest.get("image_size_px") or [0, 0]

    scale, dpi, px_per_ft, hyp_ft = infer_crop_calibration(
        gt, int(img_w), int(img_h), assumed_span_ft=assumed_span_ft,
    )
    manifest["scale"] = scale
    manifest["dpi"] = dpi
    manifest["inferred_px_per_ft"] = round(px_per_ft, 2)
    manifest["calibration_hypothesis_ft"] = hyp_ft
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    conv_path = case_dir / "labelme_conversion.json"
    if conv_path.exists():
        report = json.loads(conv_path.read_text(encoding="utf-8"))
    else:
        report = {}
    report["inferred_px_per_ft"] = round(px_per_ft, 2)
    report["inferred_dpi"] = dpi
    report["calibration_hypothesis_ft"] = hyp_ft
    conv_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "case_id": case_dir.name,
        "scale": scale,
        "dpi": dpi,
        "inferred_px_per_ft": round(px_per_ft, 2),
        "calibration_hypothesis_ft": hyp_ft,
    }
