"""Normalize Arqen pipeline output and ground-truth files."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .geometry import bbox_from_points, normalize_bbox, normalize_segment


CATEGORIES = ("rooms", "walls", "doors", "windows", "labels", "dimensions")


def _room_bbox(room: dict) -> list[float] | None:
    if room.get("bbox_px"):
        return normalize_bbox(room["bbox_px"])
    if room.get("polygon_px"):
        return bbox_from_points(room["polygon_px"])
    return None


def _normalize_room(room: dict) -> dict:
    out = {
        "id": room.get("id"),
        "label": room.get("label"),
        "area_raw": room.get("area_raw"),
        "bbox_px": _room_bbox(room),
        "polygon_px": room.get("polygon_px"),
        "centroid_px": room.get("centroid_px"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _normalize_wall(wall: dict) -> dict:
    coords = wall.get("px_coords")
    if not coords or len(coords) < 4:
        return {}
    x1, y1, x2, y2 = normalize_segment(coords)
    out = {
        "id": wall.get("id"),
        "px_coords": [x1, y1, x2, y2],
        "facing": wall.get("facing"),
        "length_raw": wall.get("length_raw"),
        "is_exterior": wall.get("is_exterior"),
        "room_id": wall.get("room_id"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _normalize_opening(obj: dict) -> dict:
    out = {"id": obj.get("id"), "host_wall_id": obj.get("host_wall_id")}
    if obj.get("bbox_px"):
        out["bbox_px"] = normalize_bbox(obj["bbox_px"])
    if obj.get("center_px"):
        out["center_px"] = [float(obj["center_px"][0]), float(obj["center_px"][1])]
    elif out.get("bbox_px"):
        x0, y0, x1, y1 = out["bbox_px"]
        out["center_px"] = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
    return out


def _normalize_label(label: dict) -> dict:
    out = {
        "id": label.get("id"),
        "text": label.get("text") or label.get("label"),
        "room_id": label.get("room_id"),
    }
    if label.get("bbox_px"):
        out["bbox_px"] = normalize_bbox(label["bbox_px"])
    if label.get("center_px"):
        out["center_px"] = [float(label["center_px"][0]), float(label["center_px"][1])]
    return {k: v for k, v in out.items() if v is not None}


def _parse_dimension_value(obj: dict) -> float | None:
    if obj.get("value_raw") is not None:
        return float(obj["value_raw"])
    text = obj.get("text") or obj.get("value")
    if not text:
        return None
    cleaned = str(text).replace("'", "-").replace('"', "").replace("ft", "").strip()
    try:
        if "-" in cleaned:
            whole, frac = cleaned.split("-", 1)
            return float(whole) + float(frac) / 12.0
        return float(cleaned)
    except ValueError:
        return None


def _normalize_dimension(dim: dict) -> dict:
    out = {
        "id": dim.get("id"),
        "text": dim.get("text") or dim.get("value"),
        "value_raw": _parse_dimension_value(dim),
        "unit": dim.get("unit"),
    }
    if dim.get("bbox_px"):
        out["bbox_px"] = normalize_bbox(dim["bbox_px"])
    if dim.get("center_px"):
        out["center_px"] = [float(dim["center_px"][0]), float(dim["center_px"][1])]
    return {k: v for k, v in out.items() if v is not None}


def normalize_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert raw JSON (prediction or ground truth) to canonical scoring form."""
    out: dict[str, Any] = {
        "id": doc.get("id"),
        "image_size_px": doc.get("image_size_px"),
        "scale": doc.get("scale") or doc.get("detected_scale"),
    }

    rooms = [_normalize_room(r) for r in doc.get("rooms", [])]
    out["rooms"] = [r for r in rooms if r.get("bbox_px") or r.get("polygon_px")]

    walls = [_normalize_wall(w) for w in doc.get("walls", [])]
    out["walls"] = [w for w in walls if w.get("px_coords")]

    doors = [_normalize_opening(d) for d in doc.get("doors", [])]
    out["doors"] = [d for d in doors if d.get("bbox_px") or d.get("center_px")]

    windows = [_normalize_opening(w) for w in doc.get("windows", [])]
    out["windows"] = [w for w in windows if w.get("bbox_px") or w.get("center_px")]

    labels = [_normalize_label(l) for l in doc.get("labels", [])]
    out["labels"] = [l for l in labels if l.get("text")]

    dims = [_normalize_dimension(d) for d in doc.get("dimensions", [])]
    out["dimensions"] = [d for d in dims if d.get("value_raw") is not None or d.get("text")]

    return out


def extract_wall_windows_from_prediction(doc: dict[str, Any]) -> list[dict]:
    """Promote per-wall window counts into window objects for scoring."""
    windows: list[dict] = []
    for wall in doc.get("walls", []):
        count = wall.get("windows")
        if not count:
            continue
        try:
            n = int(count)
        except (TypeError, ValueError):
            continue
        for i in range(n):
            windows.append({
                "id": f"{wall.get('id', 'wall')}.win{i + 1}",
                "host_wall_id": wall.get("id"),
                "center_px": list(wall.get("px_coords", [0, 0, 0, 0])[:2]),
            })
    return windows
