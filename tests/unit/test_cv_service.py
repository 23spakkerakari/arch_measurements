"""HTTP layer — cv_service.py Flask endpoints (skipped if flask missing)."""

import base64

import cv2
import numpy as np
import pytest

flask = pytest.importorskip("flask")

import cv_service  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture()
def client():
    cv_service.app.config["TESTING"] = True
    with cv_service.app.test_client() as c:
        yield c


def _b64_png(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}


class TestCvAnalyzeValidation:
    def test_missing_fields_rejected(self, client):
        resp = client.post("/cv-analyze", json={})
        assert resp.status_code == 400
        assert "required" in resp.get_json()["error"]

    def test_bad_base64_rejected(self, client):
        resp = client.post(
            "/cv-analyze",
            json={"imageBase64": "data:image/png;base64,!!!", "scale": "1/4in=1ft"},
        )
        assert resp.status_code == 400

    def test_blank_image_returns_error_payload(self, client):
        blank = np.full((400, 400, 3), 255, np.uint8)
        resp = client.post(
            "/cv-analyze",
            json={"imageBase64": _b64_png(blank), "scale": "1/4in=1ft", "dpi": 150},
        )
        assert resp.status_code == 200
        assert resp.get_json()["error"] == "No building footprint found"


class TestDecodeImage:
    def test_decode_roundtrip(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img[:, :, 0] = 255  # red channel in RGB
        decoded = cv_service._decode_image(_b64_png(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
        assert decoded is not None
        assert decoded.shape == (10, 10, 3)
        assert decoded[0, 0, 0] == 255

    def test_decode_garbage_returns_none(self):
        assert cv_service._decode_image("!!!") is None
