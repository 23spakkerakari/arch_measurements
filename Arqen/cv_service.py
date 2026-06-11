"""
cv_service.py — Flask HTTP wrapper around the Arqen OpenCV wall-detection pipeline.

Deployed on Render (or any Python host) and called by the Cloudflare Pages Function
at functions/api/cv-analyze.js, which proxies /api/cv-analyze on arqenlabs.ai.

Environment variables:
  PORT            — TCP port to bind (Render sets this automatically)
  SERVICE_SECRET  — Optional shared secret; if set, requests must include
                    X-Service-Secret header with the same value.
  MAX_ANALYSIS_PX — Longest image side sent to OpenCV (default 2400). Larger
                    uploads are downscaled in-place; DPI is scaled so geometry
                    stays calibrated. Lower this if Render free tier OOMs.
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
MAX_ANALYSIS_PX = int(os.environ.get("MAX_ANALYSIS_PX", "2400"))


def _roi_crop_size(roi, w: int, h: int) -> tuple[int, int]:
    """Pixel size of the ROI crop analyze_page will process (full image if no ROI)."""
    if not isinstance(roi, dict):
        return w, h
    try:
        x0 = min(max(float(roi.get("x0_pct", 0.0)), 0.0), 1.0)
        y0 = min(max(float(roi.get("y0_pct", 0.0)), 0.0), 1.0)
        x1 = min(max(float(roi.get("x1_pct", 1.0)), 0.0), 1.0)
        y1 = min(max(float(roi.get("y1_pct", 1.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        return w, h
    crop_w = max(1, int(round((x1 - x0) * w)))
    crop_h = max(1, int(round((y1 - y0) * h)))
    return crop_w, crop_h


def _cap_image_for_memory(
    image: np.ndarray, dpi: int, roi=None,
) -> tuple[np.ndarray, int]:
    """Downscale very large rasters so room-split fits in Render free-tier RAM.

    The cap is sized against the ROI crop, not the full sheet: analyze_page
    crops to the user ROI before the memory-heavy stages, so a plan drawing
    occupying e.g. 60 % of a 4900 px sheet should keep ~MAX_ANALYSIS_PX of
    real resolution instead of being downscaled for title-block area that is
    cropped away anyway. Wall strokes survive at the higher working
    resolution, which directly improves wall recall.
    """
    h, w = image.shape[:2]
    crop_w, crop_h = _roi_crop_size(roi, w, h)
    longest = max(crop_h, crop_w)
    if longest <= MAX_ANALYSIS_PX:
        return image, dpi

    scale = MAX_ANALYSIS_PX / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    new_dpi = max(1, int(round(dpi * scale)))
    print(
        f"  [cv_service] downscaled {w}x{h} -> {new_w}x{new_h}, "
        f"dpi {dpi} -> {new_dpi} (MAX_ANALYSIS_PX={MAX_ANALYSIS_PX})",
        flush=True,
    )
    return resized, new_dpi


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

    dpi_raw = data.get("dpi", 150)
    try:
        dpi = int(dpi_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "dpi must be an integer"}), 400
    if dpi < 1 or dpi > 1200:
        return jsonify({"error": "dpi must be between 1 and 1200"}), 400

    roi_raw = data.get("roi")
    roi = roi_raw if isinstance(roi_raw, dict) else None
    doorway_close_ft = float(data.get("doorway_close_ft", 2.5))

    if DEBUG_DUMP:
        _dump_request(image, scale, dpi, roi)

    image, dpi = _cap_image_for_memory(image, dpi, roi=roi)

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
