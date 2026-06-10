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


def infer_crop_calibration(
    gt: dict,
    image_width: int,
    image_height: int,
    assumed_span_ft: float = 50.0,
) -> tuple[str, int, float]:
    """Pick scale/dpi so px_per_ft suits a LabelMe crop (not a full sheet).

    Cropped PNGs are already in pixel space. We use ``1in=1ft`` with a synthetic
    DPI equal to the inferred px_per_ft so wall-pair gaps and min-length
    filters land in a sensible range (~12–72 px/ft).
    """
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
        span_px = max(max(xs) - min(xs), max(ys) - min(ys))
    else:
        span_px = 0.85 * max(image_width, image_height)

    px_per_ft = max(12.0, min(72.0, span_px / assumed_span_ft))
    dpi = int(round(px_per_ft))
    return "1in=1ft", dpi, px_per_ft


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
        auto_scale, auto_dpi, px_per_ft = infer_crop_calibration(
            gt, img_w, img_h, assumed_span_ft=assumed_span_ft,
        )
        if scale is None:
            scale = auto_scale
        if dpi is None:
            dpi = auto_dpi
        report["inferred_px_per_ft"] = round(px_per_ft, 2)
        report["inferred_dpi"] = dpi
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
