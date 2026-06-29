"""Unit tests for the ML window detector + hybrid fusion.

These never load a real model: ``_load_model`` is monkeypatched with a fake so
the tests stay fast and dependency-free (no torch/ultralytics/weights needed).
They lock down the public contract: output schema parity, tiling/global-coord
mapping, soft-fail behavior, env gating, and fusion dedup/boost logic.
"""

from __future__ import annotations

import numpy as np
import pytest

import window_detect_ml as wml

pytestmark = pytest.mark.unit


class _Arr:
    """Mimic the ultralytics tensor surface: ``.cpu().numpy()``."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _Arr(xyxy)
        self.conf = _Arr(conf)


class _Result:
    def __init__(self, xyxy, conf):
        self.boxes = _Boxes(xyxy, conf)


class _FakeModel:
    """Returns the same detections for every tile it is given."""

    def __init__(self, xyxy, conf):
        self._xyxy = xyxy
        self._conf = conf
        self.calls = 0

    def predict(self, tile, conf=0.25, iou=0.45, verbose=False):
        self.calls += 1
        return [_Result(self._xyxy, self._conf)]


@pytest.fixture
def fake_single_box(monkeypatch):
    model = _FakeModel(xyxy=[[100, 100, 140, 160]], conf=[0.9])
    monkeypatch.setattr(wml, "_load_model", lambda: model)
    return model


def test_detect_windows_ml_schema(fake_single_box):
    img = np.full((640, 640, 3), 255, dtype=np.uint8)
    out = wml.detect_windows_ml(img, px_per_unit=10.0, unit_label="ft")

    assert len(out) == 1
    w = out[0]
    # Schema parity with analyze_page window dicts.
    for key in ("id", "host_wall_id", "bbox_px", "center_px", "width",
                "width_raw", "is_exterior", "evidence", "confidence"):
        assert key in w
    assert w["evidence"] == "ml"
    assert w["bbox_px"] == [100.0, 100.0, 140.0, 160.0]
    assert w["center_px"] == [120.0, 130.0]
    # Opening width is the longer side (60 px) / 10 px-per-ft = 6.0 ft.
    assert w["width_raw"] == pytest.approx(6.0)
    assert w["confidence"] == pytest.approx(0.9)
    assert w["host_wall_id"] is None  # no walls supplied


def test_detect_windows_ml_assigns_nearest_wall(fake_single_box):
    img = np.full((640, 640, 3), 255, dtype=np.uint8)
    walls = [
        {"id": "wA", "px_coords": [0, 130, 640, 130], "is_exterior": True},
        {"id": "wB", "px_coords": [0, 600, 640, 600], "is_exterior": False},
    ]
    out = wml.detect_windows_ml(img, px_per_unit=10.0, walls=walls)
    assert out[0]["host_wall_id"] == "wA"  # center y=130 sits on wA
    assert out[0]["is_exterior"] is True


def test_detect_windows_ml_tiles_large_image(fake_single_box):
    # A wider image yields multiple tile origins; per-tile boxes map to global
    # coords and NMS collapses exact duplicates from overlap regions.
    img = np.full((640, 1500, 3), 255, dtype=np.uint8)
    out = wml.detect_windows_ml(img, px_per_unit=10.0)
    assert fake_single_box.calls > 1  # actually tiled
    assert len(out) >= 1


def test_door_coincidence_filter(fake_single_box):
    # The fake model emits a box at center [120,130]. A door at the same spot
    # should suppress it (door swing arc misread as window).
    img = np.full((640, 640, 3), 255, dtype=np.uint8)
    doors = [{"bbox_px": [100, 100, 140, 160], "center_px": [120.0, 130.0]}]
    out = wml.detect_windows_ml(img, px_per_unit=10.0, doors=doors)
    assert out == []
    # A door elsewhere must not suppress the detection.
    far_doors = [{"bbox_px": [400, 400, 440, 460], "center_px": [420.0, 430.0]}]
    out2 = wml.detect_windows_ml(img, px_per_unit=10.0, doors=far_doors)
    assert len(out2) == 1


def test_soft_fail_without_model(monkeypatch):
    monkeypatch.setattr(wml, "_load_model", lambda: None)
    img = np.full((640, 640, 3), 255, dtype=np.uint8)
    assert wml.detect_windows_ml(img, px_per_unit=10.0) == []


def test_invalid_inputs_return_empty(fake_single_box):
    img = np.full((640, 640, 3), 255, dtype=np.uint8)
    assert wml.detect_windows_ml(None, px_per_unit=10.0) == []
    assert wml.detect_windows_ml(img, px_per_unit=0.0) == []


def test_ml_enabled_env(monkeypatch):
    monkeypatch.delenv("ARQEN_WINDOW_ML", raising=False)
    assert wml.ml_enabled() is False
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("ARQEN_WINDOW_ML", truthy)
        assert wml.ml_enabled() is True
    monkeypatch.setenv("ARQEN_WINDOW_ML", "0")
    assert wml.ml_enabled() is False


def _win(id_, cx, cy, half=15, conf=None, evidence="sill"):
    w = {
        "id": id_,
        "host_wall_id": None,
        "bbox_px": [cx - half, cy - half, cx + half, cy + half],
        "center_px": [float(cx), float(cy)],
        "width": "3.00 ft",
        "width_raw": 3.0,
        "is_exterior": True,
        "evidence": evidence,
    }
    if conf is not None:
        w["confidence"] = conf
    return w


def test_fuse_adds_non_overlapping_ml():
    classical = [_win("win1", 100, 100)]
    ml = [dict(_win("", 400, 400, conf=0.8, evidence="ml"))]
    out = wml.fuse_windows(classical, ml, px_per_unit=10.0)
    assert len(out) == 2
    # Stable spatial order + reassigned ids.
    assert [w["id"] for w in out] == ["win1", "win2"]
    centers = sorted(w["center_px"] for w in out)
    assert centers == [[100.0, 100.0], [400.0, 400.0]]


def test_fuse_dedups_overlapping_ml_and_boosts_confidence():
    classical = [_win("win1", 100, 100, conf=0.5)]
    ml = [dict(_win("", 102, 101, conf=0.9, evidence="ml"))]
    out = wml.fuse_windows(classical, ml, px_per_unit=10.0)
    assert len(out) == 1  # ML overlapped -> not added as new
    assert out[0]["evidence"] == "sill"  # classical kept as backbone
    assert out[0]["confidence"] > 0.5  # confirmed by ML -> boosted


def test_fuse_empty_ml_is_noop_with_reindex():
    classical = [_win("winX", 300, 50), _win("winY", 10, 10)]
    out = wml.fuse_windows(classical, [], px_per_unit=10.0)
    assert [w["id"] for w in out] == ["win1", "win2"]
    assert out[0]["center_px"] == [10.0, 10.0]  # sorted by x then y
