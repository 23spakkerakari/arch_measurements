# Baseline Metrics — Current Implementation

Captured **2026-06-10T14:49:13Z** on Python 3.14.2 / OpenCV 4.11.0 / NumPy 2.4.0 (Windows 11).
Source of truth: `validation/baselines/baseline.json`. Reproduce with:

```bash
python validation/capture_baseline.py        # re-capture (overwrites baseline)
python validation/compare_to_baseline.py     # regression gate vs this baseline
python -m pytest -m unit                     # 193 unit tests (~2 s)
python -m pytest -m integration              # 30 integration tests (~25 s)
```

The CV path is deterministic for fixed inputs, so these numbers are exactly
reproducible on this environment (e.g. `mcginnies_pdf` total area 8702.1 ft²
matches the historical `Arqen/out.json` run bit-for-bit).

## Cases

| Case | Input | Ground truth | Notes |
|------|-------|--------------|-------|
| `synth_two_room` | Generated 60x40 ft rectangle, 1 partition, door, 2 windows | Exact (generated) | True accuracy measurable |
| `synth_l_shape` | Generated 70x50 ft L-shape, 1 partition, door | Exact (generated) | Tests notch geometry |
| `capture_153430` | Web-app capture (TRDI plan), 3/8"=1ft @ 144 DPI, ROI | None yet | Structural + snapshot only |
| `capture_165134` | Web-app capture (room-split plan), 3/8"=1ft @ 144 DPI, ROI | None yet | Structural + snapshot only |
| `mcginnies_pdf` | `Arqen/test.pdf` full sheet, 1in=16ft @ 150 DPI | None yet | Largest/slowest case |
| `demo_minimal` | Static scorer fixture | Hand-written | Verifies the scorer itself |

## Structural metrics (no ground truth needed)

| Case | Walls (ext/int) | Rooms | Area (ft²) | Wall-network closure | Dangling ends | Interior coverage | Runtime |
|------|-----------------|-------|------------|----------------------|---------------|-------------------|---------|
| `synth_two_room` | 7 (5/2) | 2 | 2957.9 | 0.857 | 2 | 0.722 | 1.2 s |
| `synth_l_shape` | 10 (8/2) | 2 | 3045.3 | 0.900 | 2 | 0.713 | 1.1 s |
| `capture_153430` | 39 (19/20) | 6 | 2797.0 | 0.615 | 30 | 0.605 | 3.1 s |
| `capture_165134` | 35 (17/18) | 6 | 2623.2 | 0.657 | 24 | 0.619 | 4.2 s |
| `mcginnies_pdf` | 126 (58/68) | 40 | 8702.1 | 0.758 | 61 | 0.654 | 11.0 s |

Synthetic `total_area` and interior coverage are distorted by a known issue:
the footprint polygon swallows the adjacent dimension-string strip, inflating
the area denominator (~2,500 ft² true outer area for `synth_l_shape`). Rooms
themselves are exact — see the accuracy table. Tracked as the footprint /
annotation separation item in `docs/improvement_proposals.md`.

## Accuracy vs exact ground truth (synthetic cases)

Precision / Recall / F1 at the default matching thresholds:

| Category | synth_two_room | synth_l_shape | Interpretation |
|----------|----------------|---------------|----------------|
| Rooms | **1.00** / **1.00** / 1.00 | **1.00** / **1.00** / 1.00 | Exact room detection (IoU 0.97), including the L-shaped room polygon; phantom cells eliminated |
| Walls | 0.57 / **0.80** / 0.67 | 0.60 / **0.86** / 0.71 | All exterior axes land on the GT centerlines; remaining FP/FN are per-room sub-segmentation vs full-run GT in 1:1 matching (see M2) |
| Doors | 0.00 / **0.00** / 0.00 | 0.00 / **0.00** / 0.00 | **Not detected** — no door geometry in the CV path |
| Windows | 0.00 / **0.00** / 0.00 | — (none drawn) | **Not detected** — windows only bridged morphologically |
| Dimensions | 0.00 / **0.00** / 0.00 | 0.00 / **0.00** / 0.00 | **Not extracted** — dimension strings are deliberately filtered as annotation ink; no OCR |
| Labels | — (none drawn) | — | No OCR in CV path (LLM-only in web app) |

### Space boundary closure

| Measure | synth_two_room | synth_l_shape | capture_153430 | capture_165134 | mcginnies_pdf |
|---------|----------------|---------------|----------------|----------------|---------------|
| Wall-network closure rate | 0.857 | 0.900 | 0.615 | 0.657 | 0.758 |
| GT-room boundary closure rate (>=95% perimeter) | 1.00 | 1.00 | n/a | n/a | n/a |
| Mean GT-room boundary coverage | 1.00 | 1.00 | n/a | n/a | n/a |
| Interior coverage (room cells / footprint) | 0.722 | 0.713 | 0.605 | 0.619 | 0.654 |

## Honest read of the baseline

1. **Doors, windows, dimensions are at 0% recall by construction** — the CV
   path has no detectors for them. These categories cannot improve without new
   capability (see `docs/improvement_proposals.md` P1/P2).
2. **Wall recall is held down by representation mismatch**, not only missed
   ink: predictions are per-room sub-segments while GT is full wall runs, so
   greedy 1:1 matching counts fragments as false positives. Aggregate-overlap
   scoring would read higher; the per-object numbers are kept as the honest,
   stricter baseline.
3. **Wall-network closure 0.62-0.76 on real plans** — endpoints still stop
   short of the wall they visually meet; corner snapping (proposal #2 in the
   ranked list) is the next lever.
4. **Interior coverage 0.60-0.72** — partly real (corridors lost to the
   `min_room_ft2` filter, wall-band erosion, doorway over-closing) and partly
   the inflated footprint denominator noted above.
5. **Rooms are exact on synthetic plans** (P=R=1.0, boundary closure 1.0);
   on captures the room map still merges areas through unsealed door gaps.
6. **Against `docs/project_context.md` targets** (>95% rooms/walls, >90%
   windows/doors): rooms 100% on synth, walls 80-86% recall,
   windows/doors 0%. The gap is measured and tracked now.

## Tolerances enforced by the gate

From `validation/arqen_validation/compare.py`: wall/room counts +/-max(2, 10%),
total area +/-5%, px_per_ft +/-0.01, wall-network closure drop <=0.05,
interior coverage +/-0.10, per-category F1 and recall drop <=0.02.

## Baseline history

| Date (UTC) | Change |
|------------|--------|
| 2026-06-10T04:07 | Initial baseline of the untouched pipeline. |
| 2026-06-10T14:49 | Accepted after the phantom-suppression + snap-validation iteration (proposal #1): pair-validated snapping with annotation hop, stroke-partner stats, exterior-envelope filters for rooms and Hough interiors, span trim/clamp, dedup span-overlap fix. Synth rooms 0.5 → 1.0 precision, both synth cases exact; wall-network closure +0.07…+0.18 and dangling endpoints −2…−28 on every real plan; mcginnies recovers 21 walls (105 → 126) at stable room count 40. Synth `total_area`/coverage shifts are the footprint-strip artifact described above, accepted knowingly. |
