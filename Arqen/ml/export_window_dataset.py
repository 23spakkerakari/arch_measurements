#!/usr/bin/env python3
"""Export a tiled YOLO window-detection dataset from validation cases.

The LabelMe-derived validation cases (``validation/cases/labelme_fp_*``) carry
per-plan ``windows[].bbox_px`` ground truth on very large sheets
(~2827x3719 px). A single window is a tiny, repeated symbol, so we tile each
sheet into overlapping patches and remap window boxes into tile coordinates.
This turns ~20 plans into thousands of training patches with a single
``window`` class.

The split is done at the *plan* level (not the tile level) so tiles from one
plan never leak between train and val.

Usage:
    python Arqen/ml/export_window_dataset.py
    python Arqen/ml/export_window_dataset.py --tile 640 --overlap 0.2 --val-frac 0.2
    python Arqen/ml/export_window_dataset.py --out Arqen/ml/dataset

Output layout (YOLO-detection format):
    <out>/
      images/train/<case>__r<row>_c<col>.png
      images/val/...
      labels/train/<case>__r<row>_c<col>.txt   # class cx cy w h  (normalized)
      labels/val/...
      dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPO_ROOT / "validation" / "cases"
DEFAULT_OUT = Path(__file__).resolve().parent / "dataset"

# A window box must keep at least this fraction of its area inside a tile to be
# emitted as a label for that tile (avoids sliver boxes at tile borders).
MIN_BOX_KEEP_FRAC = 0.35


def _resolve_image(case_dir: Path, manifest: dict) -> Path | None:
    img = manifest.get("image")
    if not img:
        return None
    p = Path(img)
    if not p.is_absolute():
        p = (case_dir / p).resolve()
    return p if p.exists() else None


def _iter_tile_origins(size: int, tile: int, step: int) -> list[int]:
    """Tile-origin positions covering ``size`` px, always including the far edge."""
    if size <= tile:
        return [0]
    origins = list(range(0, size - tile + 1, step))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


def _clip_box_to_tile(
    box: list[float], ox: int, oy: int, tile: int
) -> tuple[float, float, float, float] | None:
    """Clip an absolute [x0,y0,x1,y1] box to a tile; return tile-local box."""
    x0, y0, x1, y1 = box
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    orig_area = max(1e-6, (x1 - x0) * (y1 - y0))

    cx0 = max(x0, ox)
    cy0 = max(y0, oy)
    cx1 = min(x1, ox + tile)
    cy1 = min(y1, oy + tile)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    keep_frac = ((cx1 - cx0) * (cy1 - cy0)) / orig_area
    if keep_frac < MIN_BOX_KEEP_FRAC:
        return None
    return (cx0 - ox, cy0 - oy, cx1 - ox, cy1 - oy)


def _collect_cases(cases_root: Path, prefix: str) -> list[Path]:
    return sorted(
        p for p in cases_root.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
        and (p / "ground_truth.json").exists()
        and (p / "manifest.json").exists()
    )


def export(
    cases_root: Path,
    out_dir: Path,
    tile: int,
    overlap: float,
    val_frac: float,
    prefix: str,
    neg_frac: float,
    seed: int,
    empty_bg: bool = False,
    exclude: set[str] | None = None,
) -> dict:
    step = max(1, int(round(tile * (1.0 - overlap))))
    rng = random.Random(seed)
    exclude = exclude or set()

    cases = [c for c in _collect_cases(cases_root, prefix) if c.name not in exclude]
    if not cases:
        raise SystemExit(f"No '{prefix}*' cases with ground_truth+manifest under {cases_root}")

    # Plan-level split: shuffle case order deterministically, hold out val_frac.
    order = list(cases)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_frac))) if len(order) > 1 else 0
    val_cases = set(p.name for p in order[:n_val])

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "train_cases": [], "val_cases": [],
        "tiles": {"train": 0, "val": 0},
        "boxes": {"train": 0, "val": 0},
        "skipped_cases": [],
    }

    for case_dir in cases:
        manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        gt = json.loads((case_dir / "ground_truth.json").read_text(encoding="utf-8"))
        windows = [w.get("bbox_px") for w in gt.get("windows", []) if w.get("bbox_px")]
        img_path = _resolve_image(case_dir, manifest)
        if img_path is None:
            stats["skipped_cases"].append({"case": case_dir.name, "reason": "image not found"})
            continue
        if not windows and not empty_bg:
            stats["skipped_cases"].append({"case": case_dir.name, "reason": "no GT windows"})
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            stats["skipped_cases"].append({"case": case_dir.name, "reason": "decode failed"})
            continue
        h, w = bgr.shape[:2]

        split = "val" if case_dir.name in val_cases else "train"
        stats[f"{split}_cases"].append(case_dir.name)

        # Windowless plans are pure hard-negative sources (door arcs, fixtures,
        # callouts). Keep a larger share of their non-empty tiles so the model
        # learns those symbols are NOT windows.
        bg_only = not windows
        keep_empty_prob = min(1.0, neg_frac * 4) if bg_only else neg_frac

        for oy in _iter_tile_origins(h, tile, step):
            for ox in _iter_tile_origins(w, tile, step):
                tw = min(tile, w - ox)
                th = min(tile, h - oy)
                labels: list[str] = []
                for box in windows:
                    clipped = _clip_box_to_tile(box, ox, oy, tile)
                    if clipped is None:
                        continue
                    lx0, ly0, lx1, ly1 = clipped
                    cx = ((lx0 + lx1) / 2.0) / tw
                    cy = ((ly0 + ly1) / 2.0) / th
                    bw = (lx1 - lx0) / tw
                    bh = (ly1 - ly0) / th
                    labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                if not labels:
                    # Skip blank (near-white) tiles entirely; only keep empties
                    # that actually contain drawing ink, sampled by keep prob.
                    tile_probe = bgr[oy:oy + th, ox:ox + tw]
                    ink = float((tile_probe < 200).mean())
                    if ink < 0.01 or rng.random() > keep_empty_prob:
                        continue

                stem = f"{case_dir.name}__r{oy}_c{ox}"
                tile_img = bgr[oy:oy + th, ox:ox + tw]
                # Pad partial edge tiles up to full tile size (white background).
                if tile_img.shape[0] != tile or tile_img.shape[1] != tile:
                    canvas = np.full((tile, tile, 3), 255, dtype=np.uint8)
                    canvas[:tile_img.shape[0], :tile_img.shape[1]] = tile_img
                    tile_img = canvas
                    # Re-normalize labels against full tile size, not cropped size.
                    fixed: list[str] = []
                    for box in windows:
                        clipped = _clip_box_to_tile(box, ox, oy, tile)
                        if clipped is None:
                            continue
                        lx0, ly0, lx1, ly1 = clipped
                        fixed.append(
                            f"0 {((lx0 + lx1) / 2.0) / tile:.6f} "
                            f"{((ly0 + ly1) / 2.0) / tile:.6f} "
                            f"{(lx1 - lx0) / tile:.6f} {(ly1 - ly0) / tile:.6f}"
                        )
                    labels = fixed

                cv2.imwrite(str(out_dir / "images" / split / f"{stem}.png"), tile_img)
                (out_dir / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(labels), encoding="utf-8"
                )
                stats["tiles"][split] += 1
                stats["boxes"][split] += len(labels)

    dataset_yaml = (
        f"# Auto-generated by export_window_dataset.py\n"
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: window\n"
    )
    (out_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    (out_dir / "export_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases-root", default=str(CASES_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output dataset dir")
    parser.add_argument("--tile", type=int, default=640, help="Tile size in px")
    parser.add_argument("--overlap", type=float, default=0.2, help="Tile overlap fraction (0-1)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of plans held out for val")
    parser.add_argument("--prefix", default="labelme_", help="Case folder prefix to include")
    parser.add_argument("--neg-frac", type=float, default=0.15,
                        help="Fraction of window-free tiles to keep as background")
    parser.add_argument("--empty-bg", action="store_true",
                        help="Include zero-window plans as hard-negative background")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Case folder names to exclude entirely (held-out eval)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    stats = export(
        cases_root=Path(args.cases_root),
        out_dir=Path(args.out),
        tile=args.tile,
        overlap=args.overlap,
        val_frac=args.val_frac,
        prefix=args.prefix,
        neg_frac=args.neg_frac,
        seed=args.seed,
        empty_bg=args.empty_bg,
        exclude=set(args.exclude),
    )

    print("Dataset export complete:")
    print(f"  train: {len(stats['train_cases'])} plans, "
          f"{stats['tiles']['train']} tiles, {stats['boxes']['train']} boxes")
    print(f"  val:   {len(stats['val_cases'])} plans, "
          f"{stats['tiles']['val']} tiles, {stats['boxes']['val']} boxes")
    if stats["skipped_cases"]:
        print(f"  skipped: {stats['skipped_cases']}")
    print(f"  val plans: {sorted(stats['val_cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
