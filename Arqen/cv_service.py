"""
cv_service.py — Flask HTTP wrapper around the Arqen OpenCV wall-detection pipeline.

Deployed on Render (or any Python host) and called by the Cloudflare Pages Function
at functions/api/cv-analyze.js, which proxies /api/cv-analyze on arqenlabs.ai.

Environment variables:
  PORT            — TCP port to bind (Render sets this automatically)
  SERVICE_SECRET  — Optional shared secret; if set, requests must include
                    X-Service-Secret header with the same value.
"""

import base64
import json
import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request

from preprocess import analyze_page  # noqa: E402 — same directory as this file

app = Flask(__name__)

SECRET = os.environ.get("SERVICE_SECRET", "")

# When ARQEN_DEBUG_DUMP=1, each /cv-analyze request is saved to
# debug_runs/<timestamp>/ (image.png + request.json) for offline replay
# with debug_pipeline.py.
DEBUG_DUMP = os.environ.get("ARQEN_DEBUG_DUMP", "") == "1"


def _dump_request(image: np.ndarray, scale: str, dpi: int, roi) -> None:
    try:
        run_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "debug_runs",
            time.strftime("%Y%m%d-%H%M%S"),
        )
        os.makedirs(run_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(run_dir, "image.png"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        with open(os.path.join(run_dir, "request.json"), "w") as f:
            json.dump({"scale": scale, "dpi": dpi, "roi": roi}, f, indent=2)
        print(f"  [debug-dump] saved request to {run_dir}", flush=True)
    except Exception as e:
        print(f"  [debug-dump] failed: {e}", flush=True)


def _check_secret():
    if not SECRET:
        return None
    if request.headers.get("X-Service-Secret") != SECRET:
        return jsonify({"error": "Unauthorized"}), 401
    return None


def _decode_image(b64: str) -> np.ndarray | None:
    """Decode a base64 image string (with or without data-URL prefix) to an RGB ndarray."""
    if ";base64," in b64:
        b64 = b64.split(";base64,")[1]
    try:
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/cv-analyze", methods=["POST"])
def cv_analyze():
    auth_err = _check_secret()
    if auth_err:
        return auth_err

    data = request.get_json(silent=True) or {}
    img_b64 = data.get("imageBase64", "")
    scale = data.get("scale", "")
    if not img_b64 or not scale:
        return jsonify({"error": "imageBase64 and scale are required"}), 400

    image = _decode_image(img_b64)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    dpi = int(data.get("dpi", 150))
    roi_raw = data.get("roi")
    roi = roi_raw if isinstance(roi_raw, dict) else None
    doorway_close_ft = float(data.get("doorway_close_ft", 2.5))

    if DEBUG_DUMP:
        _dump_request(image, scale, dpi, roi)

    result = analyze_page(
        image, scale, dpi, roi=roi, doorway_close_ft=doorway_close_ft,
    )

    # Convert the wall_pair_mask temp file to an inline base64 data-URL so the
    # browser can render the debug overlay without a second round-trip.  The
    # temp file is deleted immediately after encoding.
    mask_path = result.pop("mask_cache_path", None)
    if mask_path:
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, buf = cv2.imencode(".png", mask)
                result["mask_base64"] = (
                    "data:image/png;base64,"
                    + base64.b64encode(buf.tobytes()).decode()
                )
        except Exception:
            pass
        finally:
            try:
                os.unlink(mask_path)
            except Exception:
                pass

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
