# Baseline Metrics — Current Implementation

Captured **2026-06-10** (post LabelMe crop wall-recall pass) on Python 3.14.2 / OpenCV 4.11.0 / NumPy 2.4.0 (Windows 11).
Source of truth: `validation/baselines/baseline.json`. Reproduce with:

```bash
python validation/capture_baseline.py        # re-capture (overwrites baseline)
python validation/compare_to_baseline.py     # regression gate vs this baseline
python -m pytest -m unit                     # 237 unit tests (~2 s)
python -m pytest -m integration              # 41 integration tests (~15 s)
```

The CV path is deterministic for fixed inputs, so these numbers are exactly
reproducible on this environment (e.g. `mcginnies_pdf` total area 8702.1 ft²
matches the historical `Arqen/out.json` run bit-for-bit).

## Cases

| Case | Input | Ground truth | Notes |
|------|-------|--------------|-------|
| `synth_two_room` | Generated 60x40 ft rectangle, 1 partition, door, 2 windows | Exact (generated) | True accuracy measurable |
| `synth_l_shape` | Generated 70x50 ft L-shape, 1 partition, door | Exact (generated) | Tests notch geometry |
| `synth_corridor` | Generated 60x40 ft, 4 ft corridor, 24 ft² closet, 3 doors | Exact (generated) | Tests corridor/small-room recovery |
| `capture_153430` | Web-app capture (TRDI plan), 3/8"=1ft @ 144 DPI, ROI | None yet | Structural + snapshot only |
| `capture_165134` | Web-app capture (room-split plan), 3/8"=1ft @ 144 DPI, ROI | None yet | Structural + snapshot only |
| `mcginnies_pdf` | `Arqen/test.pdf` full sheet, 1in=16ft @ 150 DPI | None yet | Largest/slowest case |
| `demo_minimal` | Static scorer fixture | Hand-written | Verifies the scorer itself |

## Structural metrics (no ground truth needed)

| Case | Walls (ext/int) | Rooms | Area (ft²) | Wall-network closure | Dangling ends | Interior coverage | Runtime |
|------|-----------------|-------|------------|----------------------|---------------|-------------------|---------|
| `synth_two_room` | 7 (5/2) | 2 | 2957.9 | 0.857 | 2 | 0.724 | 0.5 s |
| `synth_l_shape` | 10 (8/2) | 2 | 3045.3 | 0.900 | 2 | 0.713 | 0.5 s |
| `synth_corridor` | 12 (5/7) | 4 | 2957.9 | 0.750 | 6 | 0.679 | 0.5 s |
| `capture_153430` | 38 (18/20) | 6 | 2797.0 | 0.605 | 30 | 0.646 | 1.5 s |
| `capture_165134` | 34 (16/18) | 7 | 2623.2 | 0.632 | 25 | 0.662 | 1.9 s |
| `mcginnies_pdf` | 145 (59/86) | 41 | 8702.1 | 0.797 | 59 | 0.665 | 4.2 s |

The synth cases' remaining dangling endpoints (2 per partition with a drawn
door opening) are genuinely open geometry, correctly not bridged — these gaps
are now *classified* as doors by `Arqen/door_detect.py` rather than closed.
`synth_corridor`'s 12 walls include the 3 closet partitions recovered by the
short-partition pass; `mcginnies_pdf`'s +18 interior walls (68 → 86) are
short bathroom/closet partitions in the dorm wings recovered by the same pass
at stable room count 41.

Synthetic `total_area` and interior coverage are distorted by a known issue:
the footprint polygon swallows the adjacent dimension-string strip, inflating
the area denominator (~2,500 ft² true outer area for `synth_l_shape`). Rooms
themselves are exact — see the accuracy table. Tracked as the footprint /
annotation separation item in `docs/improvement_proposals.md`.

## Accuracy vs exact ground truth (synthetic cases)

Precision / Recall / F1 at the default matching thresholds:

| Category | synth_two_room | synth_l_shape | synth_corridor | Interpretation |
|----------|----------------|---------------|----------------|----------------|
| Rooms | **1.00** / **1.00** / 1.00 | **1.00** / **1.00** / 1.00 | **1.00** / **1.00** / 1.00 | Exact room detection, including the 4 ft corridor and the 24 ft² closet (below the compact 25 ft² floor — recovered via the corridor floor) |
| Walls (strict 1:1) | 0.43 / **0.60** / 0.50 | 0.60 / **0.86** / 0.71 | 0.56 / **0.62** / 0.59 | Reported but not gated: per-room sub-segments of one GT wall read as FP under greedy 1:1 matching |
| Walls (span coverage) | 1.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 1.00 / **0.97** / 0.99 | Length-weighted coverage (M2): every GT wall length is traced by predictions and vice versa; this is the gated wall metric |
| Doors | **1.00** / **1.00** / 1.00 | **1.00** / **1.00** / 1.00 | **1.00** / **1.00** / 1.00 | Geometric gap detection (#7): all 5 synth doors found, including the closet door behind the recovered short partitions; the 2 sill windows produce no door FPs |
| Windows | 0.00 / **0.00** / 0.00 | — (none drawn) | — (none drawn) | **Not detected** — windows only bridged morphologically |
| Dimensions | 0.00 / **0.00** / 0.00 | 0.00 / **0.00** / 0.00 | 0.00 / **0.00** / 0.00 | **Not extracted** — dimension strings are deliberately filtered as annotation ink; no OCR |
| Labels | — (none drawn) | — | — | No OCR in CV path (LLM-only in web app) |

### Space boundary closure

| Measure | synth_two_room | synth_l_shape | synth_corridor | capture_153430 | capture_165134 | mcginnies_pdf |
|---------|----------------|---------------|----------------|----------------|----------------|---------------|
| Wall-network closure rate | 0.857 | 0.900 | 0.750 | 0.605 | 0.632 | 0.797 |
| GT-room boundary closure rate (>=95% perimeter) | 1.00 | 1.00 | 0.50 | n/a | n/a | n/a |
| Mean GT-room boundary coverage | 1.00 | 1.00 | 0.90 | n/a | n/a | n/a |
| Interior coverage (room cells / footprint) | 0.724 | 0.713 | 0.679 | 0.646 | 0.662 | 0.665 |

The closet partitions formerly missing from `walls[]` (probabilistic Hough
skips sub-6-ft runs on a full sheet) are now recovered by the short-partition
pass in `analyze_page`: directional morphological opening finds 2.5–12 ft
double-stroke runs, and a candidate is accepted only when it T-junctions into
an already-accepted wall (iterated) and is not parallel-adjacent to a wall at
dimension-line offset. `synth_corridor` closure dipped 0.778 → 0.750 because
the recovered partitions add genuinely open door endpoints — the door
detector classifies those gaps instead.

## Honest read of the baseline

1. **Doors are now detected geometrically** (proposal #7): collinear wall
   gaps of 1.5–5 ft, verified ink-free, with a sill discriminator against
   windows. Synth P/R 1.0/1.0 (5/5 doors, 0 FPs). On the 20 LabelMe real-plan
   crops recall is 0.03 (9/308, precision 1.00) — door recall there is bounded
   by wall recall on those crops (both flanking walls must be detected), not
   by the gap classifier. **Windows and dimensions remain at 0% recall by
   construction** (see `docs/improvement_proposals.md` #8).
2. **Strict 1:1 wall numbers are a representation artifact**: predictions
   are per-room sub-segments while GT is full wall runs. The span-coverage
   metric (M2) shows the true picture (1.0/1.0 on synth) and is what the
   gate enforces; the strict numbers remain reported for diagnosis.
3. **Wall-network closure 0.62-0.78 on real plans** — corner/T-junction
   endpoint snapping (proposal #2) landed; the remaining dangling endpoints
   are genuine openings (doors, garage fronts) and gaps larger than 2 ft
   that need opening detection (#7) rather than more aggressive snapping.
4. **Interior coverage 0.65-0.72** — the recovery pass (proposal #5:
   measured-gap erosion, directional doorway close, corridor area floor,
   orphan reabsorption) lifted real plans +0.01…+0.04 and recovered one room
   each on `capture_165134` and `mcginnies_pdf`. The remaining gap is
   dominated by the inflated footprint denominator (the dimension-strip
   artifact, tracked under proposal #6) and by openings that merge or leak
   spaces (proposal #7) — not by the room-cell filters anymore.
5. **Rooms are exact on synthetic plans** (P=R=1.0 on all three, including
   the 4 ft corridor and sub-floor closet); on captures the room map still
   merges areas through unsealed door gaps.
6. **Against `docs/project_context.md` targets** (>95% rooms/walls, >90%
   windows/doors): rooms 100% on synth, walls 80-86% recall, doors 100% on
   synth but ~1% on real crops (wall-recall-bound), windows 0%. The gap is
   measured and tracked now.

## LabelMe crop accuracy (20 cases, reported not gated)

Primary progress metric: **wall span-coverage recall** (`categories.walls.coverage`
in per-case `report.json`), not strict 1:1 wall matching — predictions are
per-room sub-segments while GT is sparse thick-polygon centerlines.

| Metric | Value (2026-06-10 pass) |
|--------|-------------------------|
| Median wall span-coverage recall | **0.55** |
| Median strict wall recall | ~0.31 |
| Doors matched | 2/308 (recall 0.01) |
| Triage buckets | 15 sparse_gt, 3 missing_interior, 2 mixed |

Tools: `validation/score_labelme_cases.py`, `validation/triage_labelme_cases.py`,
`validation/import_labelme_cases.py --recalibrate`.

## Tolerances enforced by the gate

From `validation/arqen_validation/compare.py`: wall/room counts +/-max(2, 10%),
total area +/-5%, px_per_ft +/-0.01, wall-network closure drop <=0.05,
interior coverage +/-0.10, per-category F1 and recall drop <=0.02.

## Baseline history

| Date (UTC) | Change |
|------------|--------|
| 2026-06-10T04:07 | Initial baseline of the untouched pipeline. |
| 2026-06-10T14:49 | Accepted after the phantom-suppression + snap-validation iteration (proposal #1): pair-validated snapping with annotation hop, stroke-partner stats, exterior-envelope filters for rooms and Hough interiors, span trim/clamp, dedup span-overlap fix. Synth rooms 0.5 → 1.0 precision, both synth cases exact; wall-network closure +0.07…+0.18 and dangling endpoints −2…−28 on every real plan; mcginnies recovers 21 walls (105 → 126) at stable room count 40. Synth `total_area`/coverage shifts are the footprint-strip artifact described above, accepted knowingly. |
| 2026-06-10 (#5) | Accepted after interior-coverage recovery (proposal #5): measured contour-to-ink erosion (capped at the legacy 2x thickness), directional doorway close, aspect-aware corridor area floor (8 ft² at aspect >= 2.5) with a minimum cell width of one close kernel, and orphan-fragment reabsorption. New `synth_corridor` case (4 ft corridor + 24 ft² closet) added with exact GT, rooms P/R 1.0/1.0. Interior coverage +0.042/+0.044/+0.012 on the real plans; `capture_165134` 6 → 7 rooms and `mcginnies_pdf` 40 → 41 rooms (recovered, width-guarded against cavity slivers); all other gated metrics flat within tolerance. The gate passed against the previous baseline before re-capture (coverage deltas < +0.10). |
| 2026-06-10 (LabelMe wall recall) | LabelMe crop pass: multi-hypothesis `infer_crop_calibration` (raised px/ft cap, wall-length prior), `crop_mode` in `analyze_page` (3 ft Hough min, 6 ft polygon min, tighter phantom envelope on crops), triage tooling. Median wall span-coverage recall 0.55 on 20 LabelMe cases; synth + capture gates flat. Doors on LabelMe still wall-bound (2/308 after recalibration). |
| 2026-06-10 (#7) | Accepted after geometric door detection (proposal #7): `Arqen/door_detect.py` classifies 1.5–5 ft collinear wall gaps (ink-free verification + window-sill discriminator) into `doors[]`; `opening_score` now takes max(bbox IoU, px_per_ft-scaled center proximity) because annotated door boxes include swing arcs. Short-partition recovery pass (directional morphological open, 2.5–12 ft, T-junction acceptance, dimension-offset rejection) restores closet/bathroom partitions Hough misses. Synth doors 0/5 → 5/5 (P/R 1.0/1.0, no FPs from sill windows); `synth_corridor` 9 → 12 walls (closet partitions), closure 0.778 → 0.750 (new genuinely-open door endpoints); `mcginnies_pdf` 127 → 145 walls (+18 dorm bathroom partitions, rooms stable at 41, closure +0.017); captures unchanged. LabelMe 20-case batch: doors 9/308 recall (precision 1.00), bounded by wall recall on those crops. The 20 labelme cases entered the baseline with this capture. |
