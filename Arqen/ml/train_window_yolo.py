#!/usr/bin/env python3
"""Fine-tune a small YOLO detector for floor-plan windows.

This is an OFFLINE step. It needs the training-only deps:
    pip install -r requirements-train.txt

It expects the tiled dataset produced by ``export_window_dataset.py``:
    python Arqen/ml/export_window_dataset.py

Then:
    python Arqen/ml/train_window_yolo.py
    python Arqen/ml/train_window_yolo.py --model yolov8n.pt --epochs 100 --imgsz 640

The best checkpoint is copied to ``Arqen/ml/weights/window_yolo.pt`` — the only
artifact the runtime needs. Training runs/logs land under ``Arqen/ml/runs/``.

If you have no local GPU, training still runs on CPU (slow) or use Colab:
upload the ``dataset/`` folder, ``pip install ultralytics``, run the same call.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = ML_DIR / "dataset" / "dataset.yaml"
DEFAULT_PROJECT = ML_DIR / "runs"
WEIGHTS_OUT = ML_DIR / "weights" / "window_yolo.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="dataset.yaml path")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Pretrained base model (e.g. yolo11n.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25,
                        help="Early-stop patience (epochs without val improvement)")
    parser.add_argument("--device", default=None, help="cuda device id, or 'cpu'")
    parser.add_argument("--name", default="window_yolo", help="Run name under runs/")
    parser.add_argument("--out-weights", default=str(WEIGHTS_OUT),
                        help="Destination for the best checkpoint")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: dataset not found at {data_path}\n"
              f"Run: python Arqen/ml/export_window_dataset.py", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install -r requirements-train.txt",
              file=sys.stderr)
        return 2

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        project=str(DEFAULT_PROJECT),
        name=args.name,
        exist_ok=True,
        # Floor-plan symbols are rotation/scale-stable; keep augmentation gentle
        # so we don't invent windows that never occur in real drawings.
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.5,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
    )

    save_dir = Path(getattr(results, "save_dir", DEFAULT_PROJECT / args.name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"WARNING: best.pt not found under {save_dir}", file=sys.stderr)
        return 1

    out_weights = Path(args.out_weights)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_weights)
    print(f"\nBest checkpoint copied to: {out_weights}")
    print("Point the runtime at it via ARQEN_WINDOW_ML_WEIGHTS, or replace "
          "window_yolo.pt to make it the default (set ARQEN_WINDOW_ML=1 to use).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
