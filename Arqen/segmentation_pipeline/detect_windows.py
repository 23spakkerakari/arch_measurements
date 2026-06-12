#!/usr/bin/env python3
"""
detect_windows.py — Detect window bars along exterior wall segments.

Windows on architectural drawings appear as short, bold (heavy-lineweight)
black bars interrupting the thin double-line exterior wall.  This detector
finds them by comparing the density of dark ink in a perpendicular cross-
section band at every pixel position along the wall: a plain wall shows
two thin parallel lines (modest darkness); a window bar fills the full
wall width with heavy ink, raising the cross-section darkness well above
the wall-line baseline.

The detector re-rasterizes the PDF at the same DPI used by preprocess.py /
room_wall_split.py, so no pixel data needs to be stored in the upstream JSON.

Usage
-----
  python detect_windows.py <plan.pdf> \\
      --json walls.json --scale "1/4in=1ft" \\
      [--page 1] [--output windows.json] [--debug-img debug_win.png]

  # tune sensitivity
  python detect_windows.py plan.pdf --json walls.json --scale "1/4in=1ft" \\
      --band 0.35 --ratio 1.8 --min-win 1.5 --max-win 6.5
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import pdf_to_images
from scale_parse import parse_scale


# ── Vectorised darkness profile ───────────────────────────────────────────────

def build_profile(gray_inv: np.ndarray, x1, y1, x2, y2, band_half: int) -> np.ndarray:
    """
    For each pixel position along the wall, compute the mean inverted-grey
    value summed over a ±band_half wide perpendicular slice.

    gray_inv : uint8 image, 0 = white paper, 255 = heavy black ink
    Returns  : float32 array of length = round(wall length in pixels)
    """
    h, w = gray_inv.shape
    length = math.hypot(x2 - x1, y2 - y1)
    n = max(int(round(length)), 2)

    ts = np.linspace(0.0, 1.0, n)
    cx = x1 + ts * (x2 - x1)   # (N,)
    cy = y1 + ts * (y2 - y1)   # (N,)

    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    nx, ny = -uy, ux            # perpendicular unit vector

    ds = np.arange(-band_half, band_half + 1, dtype=np.float32)  # (B,)

    px_c = np.round(cx[:, None] + ds[None, :] * nx).astype(np.int32)  # (N, B)
    py_c = np.round(cy[:, None] + ds[None, :] * ny).astype(np.int32)  # (N, B)
    np.clip(px_c, 0, w - 1, out=px_c)
    np.clip(py_c, 0, h - 1, out=py_c)

    return gray_inv[py_c, px_c].mean(axis=1).astype(np.float32)


# ── Run finding ───────────────────────────────────────────────────────────────

def _find_runs(mask: np.ndarray):
    """Return [(start, end)] of contiguous True runs (end is inclusive)."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


# ── Per-wall detection ────────────────────────────────────────────────────────

def detect_windows_for_wall(
    gray_inv: np.ndarray,
    wall: dict,
    px_per_ft: float,
    band_half_px: int,
    darkness_ratio: float,
    min_win_ft: float,
    max_win_ft: float,
    endpoint_margin: float = 0.05,
) -> list[dict]:
    """
    Return a list of window dicts for one wall segment.

    endpoint_margin : fraction of wall length at each end to ignore —
                      corners are often darker than plain wall runs.
    """
    x1, y1, x2, y2 = wall["px_coords"]
    wall_len_px = math.hypot(x2 - x1, y2 - y1)
    if wall_len_px < 8:
        return []

    profile = build_profile(gray_inv, x1, y1, x2, y2, band_half_px)
    n = len(profile)

    # Suppress endpoint region to avoid corner/junction artifacts
    margin_px = max(1, int(round(endpoint_margin * n)))
    profile[:margin_px] = 0.0
    profile[n - margin_px:] = 0.0

    median_dark = float(np.median(profile[profile > 0])) if (profile > 0).any() else 0.0
    if median_dark < 8.0:
        # Wall barely registers — no meaningful baseline to threshold against.
        # Basement sheets with recesses only will typically fall here.
        return []

    threshold = median_dark * darkness_ratio
    min_px = max(2, int(round(min_win_ft * px_per_ft)))
    max_px = int(round(max_win_ft * px_per_ft))

    runs = _find_runs(profile >= threshold)
    runs = [(s, e) for s, e in runs if min_px <= (e - s + 1) <= max_px]

    # Merge runs separated by < 0.75 ft (likely same window interrupted by a stud line)
    merge_gap_px = max(1, int(round(0.75 * px_per_ft)))
    merged = []
    for run in runs:
        if merged and (run[0] - merged[-1][1]) <= merge_gap_px:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(list(run))
    runs = [tuple(r) for r in merged]
    runs = [(s, e) for s, e in runs if min_px <= (e - s + 1) <= max_px]

    windows = []
    for s, e in runs:
        t_s = s / (n - 1) if n > 1 else 0.0
        t_e = e / (n - 1) if n > 1 else 1.0
        t_m = (t_s + t_e) / 2.0

        wx1 = int(round(x1 + t_s * (x2 - x1)))
        wy1 = int(round(y1 + t_s * (y2 - y1)))
        wx2 = int(round(x1 + t_e * (x2 - x1)))
        wy2 = int(round(y1 + t_e * (y2 - y1)))

        width_ft = (e - s + 1) / px_per_ft

        windows.append({
            "pos_along_wall": round(t_m, 3),
            "t_start":  round(t_s, 3),
            "t_end":    round(t_e, 3),
            "width_raw": round(width_ft, 2),
            "width":    f"{width_ft:.2f} ft",
            "px_coords": [wx1, wy1, wx2, wy2],
            "peak_darkness": round(float(profile[s:e+1].max()), 1),
            "baseline_darkness": round(median_dark, 1),
        })

    return windows


# ── Debug visualisation ───────────────────────────────────────────────────────

def save_debug_image(
    image: np.ndarray, data: dict, out_path: str,
    px_per_ft: float, band_half_px: int,
):
    vis = image.copy()
    for wall in data.get("walls", []):
        x1, y1, x2, y2 = wall["px_coords"]
        cv2.line(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)

        for win in wall.get("windows", []):
            wx1, wy1, wx2, wy2 = win["px_coords"]
            # Draw the along-wall span in cyan
            cv2.line(vis, (wx1, wy1), (wx2, wy2), (0, 255, 255), 5)

            # Draw perpendicular end-ticks so the bar reads clearly
            seg_len = math.hypot(wx2 - wx1, wy2 - wy1) or 1.0
            ux = (wx2 - wx1) / seg_len
            uy = (wy2 - wy1) / seg_len
            nx, ny = -uy, ux
            tick = band_half_px
            for pt in [(wx1, wy1), (wx2, wy2)]:
                p1 = (int(pt[0] + tick * nx), int(pt[1] + tick * ny))
                p2 = (int(pt[0] - tick * nx), int(pt[1] - tick * ny))
                cv2.line(vis, p1, p2, (0, 255, 255), 3)

            # Small label: window width
            mid_x = int((wx1 + wx2) / 2) + 6
            mid_y = int((wy1 + wy2) / 2) - 6
            cv2.putText(
                vis, win["width"], (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1, cv2.LINE_AA,
            )

    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Debug image → {out_path}", file=sys.stderr)


# ── Top-level runner ──────────────────────────────────────────────────────────

def detect_windows(
    image: np.ndarray,
    data: dict,
    px_per_ft: float,
    band_half_ft: float = 0.35,
    darkness_ratio: float = 1.8,
    min_win_ft: float = 1.5,
    max_win_ft: float = 6.5,
) -> tuple[dict, int]:
    """
    Run window detection over all walls in *data*.
    Returns (updated_data_dict, band_half_px_used).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray_inv = 255 - gray

    band_half_px = max(4, int(round(band_half_ft * px_per_ft)))

    updated_walls = []
    total = 0

    for wall in data.get("walls", []):
        wins = detect_windows_for_wall(
            gray_inv, wall, px_per_ft,
            band_half_px, darkness_ratio, min_win_ft, max_win_ft,
        )
        wall_id = wall.get("id", "?")
        for k, w in enumerate(wins, 1):
            w["id"] = f"{wall_id}.win{k}"
            w["parent_wall_id"] = wall_id
        updated_walls.append({**wall, "windows": wins})
        total += len(wins)

    result = {**data, "walls": updated_walls, "window_count": total}
    return result, band_half_px


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Detect window bars along exterior wall segments."
    )
    ap.add_argument("pdf", help="Architectural plan PDF")
    ap.add_argument("--json",    required=True,
                    help="Wall JSON produced by preprocess.py or room_wall_split.py")
    ap.add_argument("--scale",   required=True,
                    help='Drawing scale, e.g. "1/4in=1ft"')
    ap.add_argument("--page",    type=int, default=1,
                    help="PDF page to analyse (1-indexed, default 1)")
    ap.add_argument("--dpi",     type=int, default=300,
                    help="Rasterisation DPI — must match the upstream run (default 300)")
    ap.add_argument("--band",    type=float, default=0.35,
                    help="Half-width of perpendicular sampling band in feet (default 0.35 ≈ 4 in)")
    ap.add_argument("--ratio",   type=float, default=1.8,
                    help="Darkness multiple above median to flag a window (default 1.8)")
    ap.add_argument("--min-win", type=float, default=1.5,
                    help="Minimum window width in feet (default 1.5)")
    ap.add_argument("--max-win", type=float, default=6.5,
                    help="Maximum window width in feet (default 6.5)")
    ap.add_argument("--output",    default="windows.json",
                    help="Output JSON path (default: windows.json)")
    ap.add_argument("--debug-img", default=None,
                    help="Save annotated debug PNG to this path")
    args = ap.parse_args()

    pdf_path  = Path(args.pdf)
    json_path = Path(args.json)

    for p in (pdf_path, json_path):
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "walls" not in data:
        print("Error: JSON has no 'walls' key — run preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    cal = parse_scale(args.scale, args.dpi, output_unit="ft")
    px_per_ft = cal["px_per_unit"]

    print(f"Rasterising page {args.page} at {args.dpi} DPI …", file=sys.stderr)
    images = pdf_to_images(str(pdf_path), dpi=args.dpi)
    idx = args.page - 1
    if idx < 0 or idx >= len(images):
        print(f"Error: page {args.page} out of range (PDF has {len(images)} pages)",
              file=sys.stderr)
        sys.exit(1)
    image = images[idx]

    print(f"Detecting windows  (band ±{args.band} ft, ratio ×{args.ratio}) …",
          file=sys.stderr)
    result, band_half_px = detect_windows(
        image, data, px_per_ft,
        band_half_ft=args.band,
        darkness_ratio=args.ratio,
        min_win_ft=args.min_win,
        max_win_ft=args.max_win,
    )

    # Per-wall summary
    print("", file=sys.stderr)
    for wall in result["walls"]:
        wc = len(wall.get("windows", []))
        if wc:
            widths = ", ".join(w["width"] for w in wall["windows"])
            print(f"  {wall['id']:10s}  {wc} window(s)  [{widths}]", file=sys.stderr)

    print(f"\nTotal windows detected: {result['window_count']}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Written → {args.output}", file=sys.stderr)

    if args.debug_img:
        save_debug_image(image, result, args.debug_img, px_per_ft, band_half_px)


if __name__ == "__main__":
    main()
