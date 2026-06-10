# Baseline Metrics — Current Implementation

Captured **2026-06-10T04:07:52Z** on Python 3.14.2 / OpenCV 4.11.0 / NumPy 2.4.0 (Windows 11).
Source of truth: `validation/baselines/baseline.json`. Reproduce with:

```bash
python validation/capture_baseline.py        # re-capture (overwrites baseline)
python validation/compare_to_baseline.py     # regression gate vs this baseline
python -m pytest -m unit                     # 169 unit tests (~2 s)
python -m pytest -m integration              # 27 integration tests (~20 s)
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
| `synth_two_room` | 9 (6/3) | 4 | 2970.7 | 0.833 | 3 | 0.851 | 1.0 s |
| `synth_l_shape` | 11 (11/0) | 2 | 2301.9 | 0.773 | 5 | 0.768 | 0.9 s |
| `capture_153430` | 35 (15/20) | 8 | 2797.0 | 0.543 | 32 | 0.640 | 2.6 s |
| `capture_165134` | 35 (14/21) | 8 | 2623.2 | 0.557 | 31 | 0.603 | 3.2 s |
| `mcginnies_pdf` | 105 (54/51) | 40 | 8702.1 | 0.576 | 89 | 0.717 | 7.6 s |

## Accuracy vs exact ground truth (synthetic cases)

Precision / Recall / F1 at the default matching thresholds:

| Category | synth_two_room | synth_l_shape | Interpretation |
|----------|----------------|---------------|----------------|
| Rooms | 0.50 / **1.00** / 0.67 | 0.50 / **0.50** / 0.50 | All true rooms found in the rectangle case but extra phantom cells appear (4 detected vs 2 real); the L-shape merges/splits one room |
| Walls | 0.22 / **0.40** / 0.29 | 0.55 / **0.86** / 0.67 | Exterior walls found, but per-room sub-segmentation fragments them vs full-run GT, hurting 1:1 matching; the two-room interior partition is found split |
| Doors | 0.00 / **0.00** / 0.00 | 0.00 / **0.00** / 0.00 | **Not detected** — no door geometry in the CV path |
| Windows | 0.00 / **0.00** / 0.00 | — (none drawn) | **Not detected** — windows only bridged morphologically |
| Dimensions | 0.00 / **0.00** / 0.00 | 0.00 / **0.00** / 0.00 | **Not extracted** — dimension strings are deliberately filtered as annotation ink; no OCR |
| Labels | — (none drawn) | — | No OCR in CV path (LLM-only in web app) |

### Space boundary closure

| Measure | synth_two_room | synth_l_shape | capture_153430 | capture_165134 | mcginnies_pdf |
|---------|----------------|---------------|----------------|----------------|---------------|
| Wall-network closure rate | 0.833 | 0.773 | 0.543 | 0.557 | 0.576 |
| GT-room boundary closure rate (>=95% perimeter) | 0.50 | 0.00 | n/a | n/a | n/a |
| Mean GT-room boundary coverage | 0.88 | 0.86 | n/a | n/a | n/a |
| Interior coverage (room cells / footprint) | 0.851 | 0.768 | 0.640 | 0.603 | 0.717 |

## Honest read of the baseline

1. **Doors, windows, dimensions are at 0% recall by construction** — the CV
   path has no detectors for them. These categories cannot improve without new
   capability (see `docs/improvement_proposals.md` P1/P2).
2. **Wall recall is held down by representation mismatch**, not only missed
   ink: predictions are per-room sub-segments while GT is full wall runs, so
   greedy 1:1 matching counts fragments as false positives. Aggregate-overlap
   scoring would read higher; the per-object numbers are kept as the honest,
   stricter baseline.
3. **Wall-network closure ~0.54-0.58 on real plans** — roughly 4 in 10 wall
   endpoints don't terminate at another wall, so space enclosure relies
   heavily on the morphological room map rather than the wall graph.
4. **Interior coverage 0.60-0.72 on real plans** — a quarter to a third of the
   footprint isn't assigned to any room cell (corridors lost to the
   `min_room_ft2` filter, wall-band erosion, doorway over-closing).
5. **Phantom rooms on synthetic plans** (4 vs 2) — window sill ink and
   partition endpoint extension create spurious cells; precision 0.5.
6. **Against `docs/project_context.md` targets** (>95% rooms/walls, >90%
   windows/doors): rooms ~50-100% recall, walls 40-86% recall,
   windows/doors 0%. The gap is measured and tracked now.

## Tolerances enforced by the gate

From `validation/arqen_validation/compare.py`: wall/room counts +/-max(2, 10%),
total area +/-5%, px_per_ft +/-0.01, wall-network closure drop <=0.05,
interior coverage +/-0.10, per-category F1 and recall drop <=0.02.
