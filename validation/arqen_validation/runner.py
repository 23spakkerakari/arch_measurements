"""Run the Arqen pipeline on validation cases and summarize results.

Shared by ``capture_baseline.py``, ``compare_to_baseline.py``, and the
integration test suite so every consumer runs cases identically.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .closure import interior_coverage, wall_network_closure
from .normalize import normalize_document

VALIDATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VALIDATION_ROOT.parent
ARQEN_DIR = REPO_ROOT / "Arqen"


def ensure_arqen_on_path() -> None:
    p = str(ARQEN_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_manifest(case_dir: Path) -> dict:
    manifest_path = Path(case_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {case_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve(case_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (Path(case_dir) / path).resolve()
    return path


def load_case_image(case_dir: Path, manifest: dict) -> np.ndarray:
    """Load the case input as an RGB ndarray (PNG/JPG or rasterized PDF page)."""
    import cv2

    if manifest.get("image"):
        img_path = _resolve(case_dir, manifest["image"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image: {img_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pdf_value = manifest.get("pdf") or manifest.get("pdf_path") or "plan.pdf"
    pdf_path = _resolve(case_dir, pdf_value)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found for case: {pdf_path}")

    import fitz  # PyMuPDF

    dpi = int(manifest.get("dpi", 300))
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(int(manifest.get("page", 0)))
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        return img.copy()
    finally:
        doc.close()


def run_case_pipeline(
    case_dir: Path,
    manifest: dict | None = None,
    write_prediction: bool = True,
) -> dict:
    """Run ``analyze_page`` per the case manifest. Returns the prediction dict
    augmented with ``runtime_s``. Cleans up the temp mask cache file."""
    ensure_arqen_on_path()
    from preprocess import analyze_page  # noqa: PLC0415

    case_dir = Path(case_dir)
    if manifest is None:
        manifest = load_manifest(case_dir)

    image = load_case_image(case_dir, manifest)

    t0 = time.time()
    result = analyze_page(
        image,
        manifest["scale"],
        int(manifest.get("dpi", 300)),
        roi=manifest.get("roi"),
        doorway_close_ft=float(manifest.get("doorway_close_ft", 2.5)),
        crop_mode=bool(manifest.get("crop_mode"))
        or bool(manifest.get("labelme_crop"))
        or bool(manifest.get("roi")),
    )
    runtime = round(time.time() - t0, 2)

    mask_path = result.pop("mask_cache_path", None)
    if mask_path:
        try:
            os.unlink(mask_path)
        except OSError:
            pass

    result["runtime_s"] = runtime

    if write_prediction and "error" not in result:
        out_path = case_dir / "prediction.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def parse_area(total_area: str | None) -> float | None:
    """'8702.1 ft²' -> 8702.1"""
    if not total_area:
        return None
    try:
        return float(str(total_area).split()[0])
    except (ValueError, IndexError):
        return None


def structural_summary(prediction: dict, closure_tol_px: float | None = None) -> dict:
    """Ground-truth-free metrics describing a pipeline run."""
    if "error" in prediction:
        return {"error": prediction["error"]}

    walls = prediction.get("walls", [])
    rooms = prediction.get("rooms", [])
    exterior = [w for w in walls if w.get("is_exterior")]
    interior = [w for w in walls if not w.get("is_exterior")]

    normalized = normalize_document(prediction)
    if closure_tol_px is None:
        closure_tol_px = max(12.0, 2.0 * float(prediction.get("px_per_ft") or 0) or 12.0)
    network = wall_network_closure(normalized.get("walls", []), closure_tol_px)
    coverage = interior_coverage(prediction)

    return {
        "error": None,
        "wall_count": len(walls),
        "exterior_wall_count": len(exterior),
        "interior_wall_count": len(interior),
        "room_count": len(rooms),
        "total_area_raw": parse_area(prediction.get("total_area")),
        "px_per_ft": prediction.get("px_per_ft"),
        "polygon_vertices": prediction.get("polygon_vertices"),
        "wall_network_closure_rate": network.get("closure_rate"),
        "dangling_endpoints": network.get("dangling_endpoints"),
        "interior_coverage": (coverage or {}).get("coverage"),
        "runtime_s": prediction.get("runtime_s"),
    }


def environment_info() -> dict:
    import platform

    import cv2

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    try:
        import fitz

        info["pymupdf"] = getattr(fitz, "__doc__", "") or "installed"
    except ImportError:
        info["pymupdf"] = None
    return info
