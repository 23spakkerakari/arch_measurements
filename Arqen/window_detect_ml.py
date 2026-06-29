"""ML window detector: tiled YOLO inference + fusion with classical windows.

This is the *augmentation* half of the hybrid window pipeline. The classical
``detect_windows`` (sill/symbol/gap geometry) stays the backbone; this module
adds a second evidence source from a fine-tuned YOLO model and fuses the two.

Design constraints (see plan "ML Window Detection"):
- Soft dependency: if torch/ultralytics or the weights file are missing, every
  entry point returns an empty list and logs once. The runtime never crashes.
- Gated: callers only invoke this when ``ARQEN_WINDOW_ML`` is truthy.
- Schema parity: emitted dicts match the window dicts from ``analyze_page``
  (id, host_wall_id, bbox_px, center_px, width, width_raw, is_exterior,
  evidence, confidence). Coordinates are in the same frame as the input image
  (the ROI crop frame inside ``analyze_page``), so the existing roi_offset
  shift applies uniformly.

Train the model offline first:
    python Arqen/ml/export_window_dataset.py
    python Arqen/ml/train_window_yolo.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Tiling must mirror export_window_dataset.py so the model sees patches at the
# same scale it was trained on.
TILE_PX = 640
TILE_OVERLAP = 0.2
# Confidence floor for accepting an ML detection. The fine-tuned nano model has
# very high recall but over-fires at low confidence. A 12-point threshold sweep
# (validation/window_threshold_sweep.py -> validation/reports/window_sweep.json)
# showed conf=0.7 maximizes all-real F1 (0.694) while nearly halving false
# positives vs 0.5. Override with ARQEN_WINDOW_ML_CONF.
DEFAULT_CONF = 0.7
DEFAULT_IOU_NMS = 0.45
# Optional gate: drop ML detections whose center is farther than this many units
# from any detected wall. The same sweep found the gate is net-negative (it
# removes true windows on walls the classical detector missed faster than it
# removes FPs), so it is OFF by default. Override with ARQEN_WINDOW_ML_WALL_FT.
DEFAULT_WALL_GATE_FT = 0.0
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "ml" / "weights" / "window_yolo.pt"

_MODEL = None
_MODEL_TRIED = False
_WARNED = False


def _warn_once(msg: str) -> None:
    global _WARNED
    if not _WARNED:
        print(f"  [window-ml] {msg}", file=sys.stderr)
        _WARNED = True


def ml_enabled() -> bool:
    """True when the ML window path is switched on via env var."""
    return os.environ.get("ARQEN_WINDOW_ML", "0").strip().lower() in ("1", "true", "yes", "on")


def _weights_path() -> Path:
    override = os.environ.get("ARQEN_WINDOW_ML_WEIGHTS")
    return Path(override) if override else DEFAULT_WEIGHTS


def _load_model():
    """Lazy-load the YOLO model once. Returns None if unavailable."""
    global _MODEL, _MODEL_TRIED
    if _MODEL is not None or _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True

    weights = _weights_path()
    if not weights.exists():
        _warn_once(f"weights not found at {weights}; ML windows disabled")
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        _warn_once("ultralytics not installed; ML windows disabled")
        return None
    try:
        _MODEL = YOLO(str(weights))
    except Exception as e:  # noqa: BLE001
        _warn_once(f"failed to load model: {e}")
        _MODEL = None
    return _MODEL


def _tile_origins(size: int, tile: int, step: int) -> list[int]:
    if size <= tile:
        return [0]
    origins = list(range(0, size - tile + 1, step))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list[list[float]], scores: list[float], iou_thr: float) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou(boxes[i], boxes[j]) < iou_thr]
    return keep


def _raw_detections(
    image: np.ndarray, conf: float, iou_nms: float,
) -> tuple[list[list[float]], list[float]]:
    """Tile the image, run YOLO per tile, return global boxes + scores."""
    model = _load_model()
    if model is None:
        return [], []

    # ultralytics expects BGR ndarrays (cv2 convention). analyze_page works in
    # RGB, so flip channels for inference.
    if image.ndim == 3 and image.shape[2] == 3:
        bgr = image[:, :, ::-1]
    else:
        bgr = image

    h, w = bgr.shape[:2]
    step = max(1, int(round(TILE_PX * (1.0 - TILE_OVERLAP))))
    origins = [(ox, oy) for oy in _tile_origins(h, TILE_PX, step)
               for ox in _tile_origins(w, TILE_PX, step)]

    boxes: list[list[float]] = []
    scores: list[float] = []
    for ox, oy in origins:
        tw = min(TILE_PX, w - ox)
        th = min(TILE_PX, h - oy)
        tile = bgr[oy:oy + th, ox:ox + tw]
        if tile.shape[0] != TILE_PX or tile.shape[1] != TILE_PX:
            canvas = np.full((TILE_PX, TILE_PX, 3), 255, dtype=np.uint8)
            canvas[:tile.shape[0], :tile.shape[1]] = tile
            tile = canvas
        try:
            res = model.predict(tile, conf=conf, iou=iou_nms, verbose=False)
        except Exception as e:  # noqa: BLE001
            _warn_once(f"inference error: {e}")
            return [], []
        for r in res:
            xyxy = getattr(r.boxes, "xyxy", None)
            confs = getattr(r.boxes, "conf", None)
            if xyxy is None or confs is None:
                continue
            xyxy = xyxy.cpu().numpy()
            confs = confs.cpu().numpy()
            for (x0, y0, x1, y1), c in zip(xyxy, confs):
                # Drop boxes that fall entirely in the white pad region.
                if x0 >= tw or y0 >= th:
                    continue
                boxes.append([
                    float(x0) + ox, float(y0) + oy,
                    float(min(x1, tw)) + ox, float(min(y1, th)) + oy,
                ])
                scores.append(float(c))

    if not boxes:
        return [], []
    keep = _nms(boxes, scores, DEFAULT_IOU_NMS)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def _nearest_wall(center: list[float], walls: list[dict]) -> tuple[Optional[str], bool, float]:
    """Return (host_wall_id, is_exterior, distance_px) for the closest wall."""
    cx, cy = center
    best_id: Optional[str] = None
    best_ext = True
    best_d = float("inf")
    for wall in walls or []:
        coords = wall.get("px_coords")
        if not coords or len(coords) < 4:
            continue
        d = _point_segment_dist(cx, cy, *coords[:4])
        if d < best_d:
            best_d = d
            best_id = wall.get("id")
            best_ext = bool(wall.get("is_exterior", True))
    return best_id, best_ext, best_d


def _point_segment_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _conf_floor(explicit: Optional[float]) -> float:
    if explicit is not None:
        return explicit
    env = os.environ.get("ARQEN_WINDOW_ML_CONF")
    try:
        return float(env) if env else DEFAULT_CONF
    except ValueError:
        return DEFAULT_CONF


def _wall_gate_ft() -> float:
    env = os.environ.get("ARQEN_WINDOW_ML_WALL_FT")
    try:
        return float(env) if env is not None else DEFAULT_WALL_GATE_FT
    except ValueError:
        return DEFAULT_WALL_GATE_FT


def _coincides_with_door(
    box: list[float], center: list[float], doors: list[dict], tol_px: float
) -> bool:
    """True if a detection overlaps a detected door (bbox IoU) or sits on its
    center. Door swing arcs are the dominant ML false positive, and the door
    detector already found them, so we suppress windows that land on a door."""
    for d in doors or []:
        db = d.get("bbox_px")
        if db and _iou(box, db) >= 0.2:
            return True
        dc = d.get("center_px")
        if dc and _near(center, dc, tol_px):
            return True
    return False


def detect_windows_ml(
    image: np.ndarray,
    px_per_unit: float,
    unit_label: str = "ft",
    walls: Optional[list[dict]] = None,
    conf: Optional[float] = None,
    iou_nms: float = DEFAULT_IOU_NMS,
    doors: Optional[list[dict]] = None,
) -> list[dict]:
    """Detect windows with the fine-tuned YOLO model.

    Returns window dicts in the same schema as ``detect_windows`` (without the
    final ``id`` assignment, which the fuser/caller handles). Coordinates are in
    the input image frame. Safe no-op (``[]``) if the model is unavailable.

    Precision controls:
    - ``conf`` (or ARQEN_WINDOW_ML_CONF): confidence floor.
    - wall gate (ARQEN_WINDOW_ML_WALL_FT): when ``walls`` are supplied, drop
      detections whose center is farther than N units from any wall, since real
      windows sit on walls. Set 0 to disable.
    - door suppression: when ``doors`` are supplied, drop detections that
      coincide with a detected door (door swing arcs are the dominant FP).
    """
    if image is None or px_per_unit <= 0:
        return []
    conf = _conf_floor(conf)
    wall_gate_ft = _wall_gate_ft()
    wall_gate_px = wall_gate_ft * px_per_unit if (wall_gate_ft > 0 and walls) else None

    to_ft = 1.0 if unit_label == "ft" else 3.2808
    gate_px = (wall_gate_px / to_ft) if wall_gate_px else None
    door_tol_px = max(6.0, 0.5 * px_per_unit)

    boxes, scores = _raw_detections(image, conf, iou_nms)
    windows: list[dict] = []
    for (x0, y0, x1, y1), score in zip(boxes, scores):
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        # Opening width is the longer side of the box (windows are drawn as a
        # narrow band along the wall).
        width_px = max(x1 - x0, y1 - y0)
        width_units = width_px / px_per_unit
        host_id, is_ext, wall_dist = _nearest_wall([cx, cy], walls or [])
        if gate_px is not None and wall_dist > gate_px:
            continue  # floating detection not on any wall -> reject
        if doors and _coincides_with_door([x0, y0, x1, y1], [cx, cy], doors, door_tol_px):
            continue  # door swing arc misread as a window -> reject
        windows.append({
            "id": "",
            "host_wall_id": host_id,
            "bbox_px": [float(x0), float(y0), float(x1), float(y1)],
            "center_px": [cx, cy],
            "width": f"{width_units:.2f} {unit_label}",
            "width_raw": round(width_units, 2),
            "is_exterior": is_ext,
            "evidence": "ml",
            "confidence": round(score, 3),
        })
    return windows


def fuse_windows(
    classical: list[dict],
    ml: list[dict],
    px_per_unit: float,
    axis_tol_px: Optional[int] = None,
) -> list[dict]:
    """Merge classical + ML window detections.

    Strategy: classical detections are trusted first (geometry-grounded). An ML
    box that overlaps an existing kept window (bbox IoU or close center) is
    treated as the same opening and dropped, but it raises the kept window's
    confidence. Non-overlapping ML boxes are added as new windows. IDs are
    reassigned ``win1..winN`` in a stable spatial order.
    """
    if px_per_unit <= 0:
        merged = list(classical)
    else:
        if axis_tol_px is None:
            axis_tol_px = max(12, int(0.6 * px_per_unit))
        center_tol = max(6.0, 0.5 * px_per_unit)

        merged: list[dict] = list(classical)
        for cand in ml:
            cb = cand["bbox_px"]
            cc = cand["center_px"]
            dup = False
            for kept in merged:
                if _iou(cb, kept["bbox_px"]) >= 0.3 or _near(cc, kept["center_px"], center_tol):
                    dup = True
                    # ML agreement boosts confidence of the existing detection.
                    prev = float(kept.get("confidence") or 0.5)
                    boost = float(cand.get("confidence") or 0.5)
                    kept["confidence"] = round(min(0.99, max(prev, boost) + 0.1 * min(prev, boost)), 3)
                    break
            if not dup:
                merged.append(cand)

    merged.sort(key=lambda w: (w["center_px"][0], w["center_px"][1]))
    for i, w in enumerate(merged):
        w["id"] = f"win{i + 1}"
    return merged


def _near(a: list[float], b: list[float], dist: float) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) < dist
