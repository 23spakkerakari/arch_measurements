"""
debug_pipeline.py — Stage-by-stage debug harness for the wall-detection pipeline.

Runs the same steps as preprocess.analyze_page but saves an annotated overlay
PNG after every stage, and tracks named "probe boxes" (pixel rects around walls
known to go missing) so each missing wall can be traced to the exact stage that
drops it.

Two input modes:
  --image <png> [--roi x0,y0,x1,y1]   Full pipeline from the original plan image
                                      (e.g. a debug_runs/<ts>/image.png capture).
  --mask <png>                        Start from a cached wall_pair_mask (the
                                      arqen_mask_*.png temp file saved by
                                      analyze_page).  The image-level stages are
                                      skipped; the filtered/footprint mask is
                                      emulated by applying the same margin
                                      blanking + grid stripping to the mask.

Usage:
  python debug_pipeline.py --mask debug_runs/trdi_wall_pair_mask.png \
      --px-per-unit 18 --out debug_runs/diag \
      --probe "top:0.05,0.02,0.95,0.12" \
      --probe "west:0.02,0.10,0.12,0.90" \
      --probe "topright:0.70,0.08,0.95,0.16" \
      --probe "bottomleft:0.06,0.92,0.42,1.0" \
      --probe "bottomctr:0.50,0.90,0.76,1.0"
"""

import argparse
import json
import sys
from pathlib import Path

# pylint: disable=no-member
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import preprocess as pp  # noqa: E402


# ─── overlay helpers ─────────────────────────────────────────────────────────

PROBE_COLOR = (255, 0, 255)  # magenta


def _canvas_from_mask(mask: np.ndarray) -> np.ndarray:
    """White canvas with mask ink drawn light grey, BGR."""
    canvas = np.full((*mask.shape, 3), 255, np.uint8)
    canvas[mask > 0] = (190, 190, 190)
    return canvas


def _draw_probes(canvas: np.ndarray, probes: dict) -> None:
    for name, (x0, y0, x1, y1) in probes.items():
        cv2.rectangle(canvas, (x0, y0), (x1, y1), PROBE_COLOR, 4)
        cv2.putText(canvas, name, (x0 + 8, max(30, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, PROBE_COLOR, 3)


def _save(canvas: np.ndarray, out_dir: Path, name: str, probes: dict) -> None:
    _draw_probes(canvas, probes)
    path = out_dir / f"{name}.png"
    cv2.imwrite(str(path), canvas)
    print(f"  saved {path}")


def _save_mask_stage(mask: np.ndarray, out_dir: Path, name: str, probes: dict) -> None:
    canvas = np.full((*mask.shape, 3), 255, np.uint8)
    canvas[mask > 0] = (0, 0, 0)
    _save(canvas, out_dir, name, probes)


def _save_segments_stage(
    segments: list,
    base_mask: np.ndarray,
    out_dir: Path,
    name: str,
    probes: dict,
    color=(0, 0, 255),
    labeled: bool = False,
) -> None:
    canvas = _canvas_from_mask(base_mask)
    for i, (x1, y1, x2, y2) in enumerate(segments):
        cv2.line(canvas, (x1, y1), (x2, y2), color, 6)
        if labeled:
            cv2.putText(canvas, str(i), ((x1 + x2) // 2, (y1 + y2) // 2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    _save(canvas, out_dir, name, probes)


FATE_COLORS = {
    "accepted": (0, 160, 0),
    "rejected-duplicate": (200, 120, 0),
    "rejected-no-pair": (0, 0, 255),
    "rejected-non-orthogonal": (160, 160, 160),
    "filtered-by-roi": (180, 0, 180),
    "dropped-at-dedup": (0, 0, 255),
}


# ─── probe bookkeeping ───────────────────────────────────────────────────────

class ProbeTracker:
    """Records, per probe box, which stages still contain mask ink / segments."""

    def __init__(self, probes: dict):
        self.probes = probes
        self.rows: list[dict] = []

    def check_mask(self, stage: str, mask: np.ndarray) -> None:
        row = {"stage": stage, "kind": "mask"}
        for name, (x0, y0, x1, y1) in self.probes.items():
            row[name] = int(cv2.countNonZero(mask[y0:y1, x0:x1]))
        self.rows.append(row)

    def check_segments(self, stage: str, segments: list) -> None:
        row = {"stage": stage, "kind": "segments"}
        for name, box in self.probes.items():
            hits = [s for s in segments if self._seg_in_box(s, box)]
            row[name] = len(hits)
        self.rows.append(row)

    @staticmethod
    def _seg_in_box(seg: tuple, box: tuple) -> bool:
        """True if a meaningful portion of the segment lies inside the box."""
        x0, y0, x1, y1 = box
        sx1, sy1, sx2, sy2 = seg
        # Clip the segment to the box; require >= 40px of clipped length.
        if abs(sx2 - sx1) >= abs(sy2 - sy1):  # horizontal
            if not (y0 <= (sy1 + sy2) / 2 <= y1):
                return False
            lo, hi = max(min(sx1, sx2), x0), min(max(sx1, sx2), x1)
            return hi - lo >= 40
        if not (x0 <= (sx1 + sx2) / 2 <= x1):
            return False
        lo, hi = max(min(sy1, sy2), y0), min(max(sy1, sy2), y1)
        return hi - lo >= 40

    def report(self) -> str:
        names = list(self.probes.keys())
        widths = {n: max(len(n), 8) for n in names}
        lines = []
        header = f"{'stage':<38} {'kind':<9}" + "".join(
            f" {n:>{widths[n]}}" for n in names)
        lines.append(header)
        lines.append("-" * len(header))
        for row in self.rows:
            line = f"{row['stage']:<38} {row['kind']:<9}" + "".join(
                f" {row[n]:>{widths[n]}}" for n in names)
            lines.append(line)
        return "\n".join(lines)


# ─── main harness ────────────────────────────────────────────────────────────

def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    px_per_unit = args.px_per_unit

    # ── Stage 0: obtain masks ────────────────────────────────────────────────
    if args.image:
        image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        if args.roi:
            x0f, y0f, x1f, y1f = [float(v) for v in args.roi.split(",")]
            roi = {"x0_pct": x0f, "y0_pct": y0f, "x1_pct": x1f, "y1_pct": y1f}
            image, _, _, _ = pp._crop_to_roi(image, roi)
        mask_full = pp._extract_wall_lines(
            image, apply_margins=False, px_per_unit=px_per_unit)
        if args.no_margins:
            mask_filtered = mask_full
        else:
            mask_filtered = pp._extract_wall_lines(
                image, apply_margins=True, px_per_unit=px_per_unit)
    else:
        mask_full = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask_full is None:
            sys.exit(f"could not read mask: {args.mask}")
        if args.no_margins:
            mask_filtered = mask_full
        else:
            # Emulate the filtered (margins-blanked) mask used for footprint
            # detection: blank exclusion zones, then strip grid lines at 0.65.
            h, w = mask_full.shape
            excl = pp._build_exclusion_mask(h, w)
            masked = mask_full.copy()
            masked[excl > 0] = 0
            mask_filtered = pp._strip_spanning_grid_lines(masked, span_frac=0.98)

    h, w = mask_full.shape
    print(f"mask size: {w}x{h}, px_per_unit={px_per_unit}")

    # Resolve probe boxes (given as fractions) to pixels.
    probes = {}
    for spec in args.probe:
        name, coords = spec.split(":")
        x0f, y0f, x1f, y1f = [float(v) for v in coords.split(",")]
        probes[name] = (int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h))
    tracker = ProbeTracker(probes)

    # ── Stage 1: wall-pair masks ─────────────────────────────────────────────
    tracker.check_mask("1-wall_pair_mask_full", mask_full)
    tracker.check_mask("1-wall_pair_mask_filtered", mask_filtered)
    _save_mask_stage(mask_full, out_dir, "01_mask_full", dict(probes))
    _save_mask_stage(mask_filtered, out_dir, "02_mask_filtered", dict(probes))

    # Exclusion zones overlay.
    excl = pp._build_exclusion_mask(h, w)
    canvas = _canvas_from_mask(mask_full)
    canvas[excl > 0] = (
        canvas[excl > 0] * 0.5 + np.array((0, 0, 255)) * 0.5).astype(np.uint8)
    _save(canvas, out_dir, "03_exclusion_zones", dict(probes))
    tracker.check_mask("1-mask_after_exclusion", cv2.bitwise_and(
        mask_full, cv2.bitwise_not(excl)))

    # ── Stage 2: footprint morphology (preprocess downscale path) ────────────
    small_h, small_w = h // pp.DOWNSCALE, w // pp.DOWNSCALE
    small = cv2.resize(mask_filtered, (small_w, small_h), interpolation=cv2.INTER_AREA)
    _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
    small = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                       iterations=2)
    close_k_size = max(25, int(12 * px_per_unit / pp.DOWNSCALE))
    close_k_size = min(close_k_size, min(small_w, small_h) // 6)
    close_k_size = max(close_k_size, 25)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k_size, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_k_size))
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kh, iterations=1)
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kv, iterations=1)
    binary = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    tracker.check_mask("2-binary_for_footprint", binary)
    _save_mask_stage(binary, out_dir, "04_binary_footprint_input", dict(probes))

    component_mask = pp.find_footprint(binary, use_exclusion=not args.no_margins)
    if component_mask is None:
        print("!! find_footprint returned None — stopping")
        print(tracker.report())
        return
    tracker.check_mask("2-footprint_component", component_mask)
    _save_mask_stage(component_mask, out_dir, "05_footprint_component", dict(probes))

    contour = pp.find_footprint_contour(component_mask)
    polygon = pp.simplify_polygon(contour, epsilon_factor=args.epsilon)

    # ── Stage 3: polygon segments ────────────────────────────────────────────
    min_seg_px = max(60, int(12 * px_per_unit))
    print(f"min_seg_px={min_seg_px}")
    from extract_wall_segments_class import extract_wall_segments
    segments = extract_wall_segments(polygon, min_length_px=min_seg_px)
    tracker.check_segments("3-polygon_segments_raw", segments)

    xs_poly, ys_poly = polygon[:, 0], polygon[:, 1]
    pad = int(min(w, h) * 0.03)
    poly_roi = {
        "x0_pct": max(0.0, float(xs_poly.min() - pad) / w),
        "y0_pct": max(0.0, float(ys_poly.min() - pad) / h),
        "x1_pct": min(1.0, float(xs_poly.max() + pad) / w),
        "y1_pct": min(1.0, float(ys_poly.max() + pad) / h),
    }
    print(f"poly_roi: x {poly_roi['x0_pct']:.3f}-{poly_roi['x1_pct']:.3f}, "
          f"y {poly_roi['y0_pct']:.3f}-{poly_roi['y1_pct']:.3f}")

    canvas = _canvas_from_mask(mask_full)
    cv2.polylines(canvas, [polygon.reshape(-1, 1, 2)], True, (0, 160, 0), 5)
    cv2.rectangle(canvas,
                  (int(poly_roi["x0_pct"] * w), int(poly_roi["y0_pct"] * h)),
                  (int(poly_roi["x1_pct"] * w), int(poly_roi["y1_pct"] * h)),
                  (200, 120, 0), 4)
    _save(canvas, out_dir, "06_polygon_and_poly_roi", dict(probes))

    filter_roi = pp._expand_poly_roi(poly_roi, mask_full, w, h) if args.no_margins else poly_roi
    max_span_frac = 1.0 if args.no_margins else 0.95
    segments = pp._filter_wall_segments(segments, w, h, roi=filter_roi, max_span_frac=max_span_frac)
    tracker.check_segments("3-polygon_after_filter", segments)
    segments = pp.snap_segments_to_walls(segments, mask_full, px_per_unit=px_per_unit)
    tracker.check_segments("3-polygon_after_snap", segments)
    _save_segments_stage(segments, mask_full, out_dir,
                         "07_polygon_segments", dict(probes))
    exterior_segs = list(segments)

    # ── Stage 4: Hough supplement with fates ─────────────────────────────────
    pair_gap_range = pp.wall_pair_gap_range(px_per_unit)
    hough_min_px = max(60, int(6 * px_per_unit))
    print(f"pair_gap_range={pair_gap_range}, hough_min_px={hough_min_px}")
    fates: list = []
    hough_segs = pp._hough_supplement(
        mask_full, segments,
        min_length_px=hough_min_px,
        dedup_tol_px=pp.dedup_axis_tol_px(px_per_unit),
        pair_gap_range=pair_gap_range,
        fates=fates,
    )
    tracker.check_segments("4-hough_accepted", hough_segs)
    for fate_name in ("rejected-duplicate", "rejected-no-pair", "rejected-non-orthogonal"):
        segs_f = [s for s, f in fates if f == fate_name]
        tracker.check_segments(f"4-hough_{fate_name}", segs_f)

    canvas = _canvas_from_mask(mask_full)
    for seg, fate in fates:
        cv2.line(canvas, seg[:2], seg[2:], FATE_COLORS[fate], 4)
    _save(canvas, out_dir, "08_hough_fates", dict(probes))
    counts = {}
    for _, f in fates:
        counts[f] = counts.get(f, 0) + 1
    print(f"hough fates: {counts}")

    if hough_segs:
        before = list(hough_segs)
        hough_segs = pp._filter_wall_segments(hough_segs, w, h, roi=filter_roi,
                                              max_span_frac=max_span_frac)
        roi_dropped = [s for s in before if s not in hough_segs]
        tracker.check_segments("4-hough_after_roi_filter", hough_segs)
        tracker.check_segments("4-hough_dropped_by_roi", roi_dropped)
        hough_segs = pp.snap_segments_to_walls(hough_segs, mask_full, px_per_unit=px_per_unit)
        tracker.check_segments("4-hough_after_snap", hough_segs)
        segments = segments + hough_segs

    tracker.check_segments("5-combined_before_dedup", segments)
    _save_segments_stage(segments, mask_full, out_dir,
                         "09_combined_before_dedup", dict(probes))

    # ── Stage 5: dedup ───────────────────────────────────────────────────────
    axis_tol_px = pp.dedup_axis_tol_px(px_per_unit)
    gap_tol_px = max(5, int(0.3 * px_per_unit))
    merged = pp.merge_and_deduplicate_segments(
        segments, axis_tol_px=axis_tol_px, gap_tol_px=gap_tol_px)
    tracker.check_segments("5-after_dedup", merged)
    print(f"dedup: {len(segments)} -> {len(merged)} "
          f"(axis_tol={axis_tol_px}, gap_tol={gap_tol_px})")
    _save_segments_stage(merged, mask_full, out_dir,
                         "10_after_dedup", dict(probes), labeled=True)

    # ── Stage 6: length filter (roi mode drops < 8 ft) ───────────────────────
    walls = pp.measure_walls(merged, px_per_unit, "ft")
    survivors = [tuple(wd["px_coords"]) for wd in walls if wd["length_raw"] >= 8.0]
    tracker.check_segments("6-after_min_length_8ft", survivors)
    _save_segments_stage(survivors, mask_full, out_dir,
                         "11_final_walls", dict(probes), labeled=True)

    # ── Stage 7: room map + exterior wall splits ─────────────────────────────
    import room_wall_split as rws  # noqa: E402

    fp_bbox = [
        int(xs_poly.min()), int(ys_poly.min()),
        int(xs_poly.max()), int(ys_poly.max()),
    ]
    room_debug = out_dir / "rooms"
    rooms, ext_sub_walls = rws.split_exterior_walls_by_room(
        exterior_segs,
        wall_pair_mask=mask_full,
        contour=contour,
        footprint_bbox=fp_bbox,
        image_shape=(h, w, 3),
        px_per_unit=px_per_unit,
        unit_label="ft",
        doorway_close_ft=args.doorway_close,
        debug_dir=str(room_debug),
    )
    print(f"room split: {len(rooms)} rooms, {len(ext_sub_walls)} exterior sub-segments")

    interior_walls = [{**w, "is_exterior": False} for w in walls if w["length_raw"] >= 8.0]
    combined = ext_sub_walls + interior_walls
    before_cleanup = len(combined)
    combined, cleanup_stats = pp.cleanup_wall_list(
        combined, axis_tol_px, px_per_unit, "ft",
    )
    parts = ", ".join(f"{k}={v}" for k, v in cleanup_stats.items() if v)
    print(f"wall cleanup: {before_cleanup} -> {len(combined)} ({parts or 'no change'})")
    tracker.check_segments(
        "7-after_wall_cleanup",
        [tuple(w["px_coords"]) for w in combined],
    )

    if (room_debug / "room_labels_color.png").exists():
        color_vis = cv2.imread(str(room_debug / "room_labels_color.png"))
        if color_vis is not None:
            _save(color_vis, out_dir, "12_room_mask", dict(probes))
    if (room_debug / "cut_layer.png").exists():
        cut = cv2.imread(str(room_debug / "cut_layer.png"), cv2.IMREAD_GRAYSCALE)
        if cut is not None:
            _save_mask_stage(cut, out_dir, "13_cut_layer", dict(probes))

    canvas = _canvas_from_mask(mask_full)
    palette = [
        (0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 200, 255),
        (255, 0, 255), (0, 128, 255), (128, 0, 255), (0, 255, 128),
    ]
    for w in ext_sub_walls:
        rid = w.get("room_id") or ""
        idx = int(rid[1:]) if rid.startswith("R") and rid[1:].isdigit() else 0
        color = palette[idx % len(palette)]
        coords = w["px_coords"]
        cv2.line(canvas, coords[:2], coords[2:], color, 6)
        mid = ((coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2)
        cv2.putText(canvas, w["id"], mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    _save(canvas, out_dir, "14_exterior_splits", dict(probes))

    print()
    print(tracker.report())
    with open(out_dir / "probe_report.json", "w") as f:
        json.dump(tracker.rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", help="Original plan image (PNG)")
    ap.add_argument("--roi", help="ROI as fractions x0,y0,x1,y1 (image mode)")
    ap.add_argument("--mask", help="Cached wall_pair_mask PNG (mask mode)")
    ap.add_argument("--px-per-unit", type=float, required=True)
    ap.add_argument("--epsilon", type=float, default=0.006,
                    help="approxPolyDP epsilon factor (0.006 = roi mode default)")
    ap.add_argument("--out", default="debug_runs/diag")
    ap.add_argument("--no-margins", action="store_true",
                    help="Mirror the fixed ROI path: no margin blanking or "
                         "exclusion zones (user-cropped input)")
    ap.add_argument("--probe", action="append", default=[],
                    help='Probe box "name:x0,y0,x1,y1" in mask fractions')
    ap.add_argument("--doorway-close", type=float, default=2.5,
                    help="Seal interior walls across doorways up to N feet")
    args = ap.parse_args()
    if not args.image and not args.mask:
        ap.error("--image or --mask required")
    run(args)


if __name__ == "__main__":
    main()
