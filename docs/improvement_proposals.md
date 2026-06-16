# Improvement Proposals — Ranked

Grounded in the measured baseline (`docs/baseline_metrics.md`, captured
2026-06-10) and the `Arqen/ARCHITECTURE.md` §7 roadmap. Every proposal cites
the baseline evidence it targets and the metric that must move.

**Scoring:** Gain = expected accuracy improvement on tracked metrics.
Effort = engineering size. Risk = chance of regressing currently-passing
cases (all changes are gated by `validation/compare_to_baseline.py`).

## Ranking

| # | Proposal | Expected gain | Effort | Regression risk | Gated by |
|---|----------|---------------|--------|-----------------|----------|
| 1 | ~~Phantom room-cell suppression~~ **DONE 2026-06-10** | Delivered — rooms P 0.50 → 1.00 on synth, closure +0.07…+0.18 on real plans | Low | Low | rooms P/R, room counts |
| 2 | ~~Wall endpoint corner-snapping~~ **DONE 2026-06-10** | Delivered — closure 0.62–0.78 on real plans (was 0.54–0.58); synth 0.86–0.90 | Medium | Medium | closure rate, walls F1 |
| 3 | Scale/DPI sanity guards | High (reliability) — prevents silently wrong *all* measurements on bad inputs | Low | Low | px_per_ft, error statuses |
| 4 | ~~Dedup audit trail + safer fallbacks~~ **DONE 2026-06-10** | Length-conservation guard skips passes that would drop >40% of total wall length; optional per-wall drop audit | Low | Low | walls P, cleanup stats |
| 5 | ~~Interior coverage recovery (corridors/small rooms)~~ **DONE 2026-06-10** | Delivered — coverage +0.01…+0.04 on real plans, +1 room on two of them; corridor/closet recovery exact on the new `synth_corridor` case. Remaining coverage gap is the footprint denominator (#6) and openings (#7) | Medium | Medium-high | interior coverage, rooms R |
| 6 | Footprint confidence + parameterized morphology | Medium — protects the single highest-leverage failure point (stage [4] has no fallback) | Medium | Medium | error rate, area, polygon |
| 7 | ~~Door detection (geometric)~~ **DONE 2026-06-10** | Delivered — synth doors 0/5 → 5/5 (P/R 1.0/1.0, no window FPs); LabelMe doors 9/308 with precision 1.00 (recall bounded by wall recall on those crops) | High | Medium | doors P/R/F1 |
| 8 | Window detection (geometric) | Unlocks windows 0% → ~60–80% recall | High | Medium | windows P/R/F1 |
| 9 | Dimension extraction (line geometry + existing LLM OCR) | Unlocks dimensions 0% → ~70% recall; enables scale cross-check | High | Low (additive) | dimensions P/R/F1 |
| 10 | ML hybrid wall graph | Very high (handles curved/diagonal/style variance) | Very high | High | full suite |

Measurement-side (not production code, no gate needed): **M1** annotate ground
truth for `capture_*` and `mcginnies_pdf` cases; **M2** add aggregate-overlap
wall scoring to complement strict 1:1 matching (fragmentation currently reads
as FP — walls P 0.22 on synth_two_room); **M3** fix the feet-inches parser
quirk in `validation/arqen_validation/normalize.py` (`20'-6"` parsed to 19.5). **DONE 2026-06-10**

---

## 1. Phantom room-cell suppression — IMPLEMENTED 2026-06-10

- **Baseline evidence:** `synth_two_room` detected 4 rooms where 2 exist
  (precision 0.50); phantom cells appeared between real walls and adjacent
  dimension strings; dimension lines also leaked into the wall list.
- **What landed** (differs from the original sketch — the root cause turned
  out to be annotation ink, not sill slivers):
  - `preprocess._snap_axis_position`: pair-validated snapping
    (`_stroke_partner_stats`) with a relative annotation hop for polygon
    edges riding a dimension-string bulge; legacy score taken from the
    stroke cluster, not the raw coordinate.
  - `preprocess.snap_segments_to_walls`: spans trimmed to the ink that
    justified the snap; exterior parents clamped to the exterior axis
    envelope (`clamp_segments_to_envelope`).
  - `room_wall_split.drop_rooms_outside_exterior` + 
    `drop_segments_outside_exterior`: room cells and Hough interior
    candidates outside the snapped exterior envelope are dropped.
  - `drop_duplicate_exterior_strokes` span-overlap fix and
    `segment_traces_exterior` clamped-distance fix (stepped perimeters no
    longer merged/rejected wrongly).
- **Measured impact:** synth rooms P/R 1.0/1.0 (IoU 0.97), room boundary
  closure 1.0 on both synth cases; exterior axes exactly on GT centerlines;
  wall-network closure 0.576 → 0.758 and dangling endpoints 89 → 61 on
  `mcginnies_pdf`; closure +0.07/+0.10 on the captures; rooms stable at 40
  on mcginnies. Baseline re-captured (see `docs/baseline_metrics.md`).
- **Residuals (feed later proposals):** footprint polygon still swallows the
  dimension strip (inflates `total_area` and the interior-coverage
  denominator — fold into #6); captures still merge rooms through unsealed
  door gaps (#5/#7); strict 1:1 wall matching penalizes sub-segments (M2).

## 2. Wall endpoint corner-snapping — IMPLEMENTED 2026-06-10

- **Baseline evidence:** wall-network closure 0.543 / 0.557 / 0.576 on real
  plans before endpoint snapping.
- **What landed:** `preprocess.snap_wall_endpoints` — post-cleanup pass that
  moves dangling H/V endpoints onto perpendicular wall axes within
  `max(12, 2*px_per_unit)` (~2 ft). Only perpendicular targets; collinear
  gaps (doorways) are never bridged. Wired into `analyze_page` after
  `cleanup_wall_list`.
- **Measured impact:** synth closure 0.86–0.90 (2 dangling = door openings);
  real plans 0.62–0.78 (up from 0.54–0.58); mcginnies 89 → 56 dangling
  endpoints. Span-coverage wall metric (M2) added and gated alongside strict
  1:1 matching.
- **Tests:** `tests/unit/test_endpoint_snap.py` (corner, T-junction, doorway
  no-bridge, degenerate skip, idempotence).

## 3. Scale/DPI sanity guards

- **Baseline evidence:** none of the baseline cases fail — but §6 of
  ARCHITECTURE.md ranks silent mis-scale as a top failure, and every pixel
  tolerance in the pipeline derives from `px_per_unit`. A wrong DPI yields
  confidently wrong output with no error.
- **Change:** validate `px_per_unit` against image size (a building should
  span a plausible 10–500 ft); cross-check footprint bbox in feet; return a
  structured warning/error instead of silent garbage.
- **Expected impact:** prevents the worst real-world failure mode (entire
  output wrong by a constant factor). No change on healthy inputs.
- **Risk:** rejecting valid unusual inputs — make it a warning field first.
- **Tests:** unit tests for the validator; integration asserts warning absent
  on all baseline cases.

## 4. Dedup audit trail + safer fallbacks

- **Baseline evidence:** `cleanup_wall_list` runs 7 destructive passes;
  synth_two_room still emits duplicate strokes (wall FP=7) while past bugs
  (per code comments) deleted real walls. Drops are currently summarized only
  as per-pass counts.
- **Change:** record per-wall drop reasons (`dropped_by: "dimension_like"`,
  …) behind a debug flag, and add a conservation check: if a pass removes
  > N% of total wall length, skip it and flag.
- **Expected impact:** no direct accuracy change; converts future dedup
  regressions from silent to visible, protects walls recall.
- **Risk:** minimal (observability + guard rails).
- **Tests:** unit test that the guard skips a pathological pass; baseline
  unchanged.

## 5. Interior coverage recovery — IMPLEMENTED 2026-06-10

- **Baseline evidence:** interior coverage 0.605 / 0.619 / 0.654 on real
  plans — up to 40% of the footprint belonged to no room. Causes: fixed
  `min_room_ft2=25` (kills corridors/closets), assumed-thickness erosion
  (`wall_thickness_px * 2`), square doorway-close kernel.
- **What landed** (all in `Arqen/room_wall_split.py`, signature-additive):
  - Measured erosion: `_contour_ink_gap` measures the median distance from
    the (morphologically inflated) footprint contour to real wall ink; the
    interior mask erodes by `max(thickness, gap+2)` capped at the legacy
    `2*thickness` instead of always `2*thickness`.
  - Directional doorway close: the square `close_kernel` on the cut layer is
    now two 1-D closes (H + V, OR-ed) — door gaps still seal along the wall
    axis, but concave corner pockets are no longer absorbed.
  - Aspect-aware floor: cells with bbox aspect >= 2.5 pass at
    `min_corridor_ft2=8` instead of `min_room_ft2=25`; **and** every cell
    must be at least one close-kernel (2.5 ft) wide in its narrow dimension —
    anything narrower is a boundary/cavity sliver the close would have sealed
    (this guard is what kept mcginnies at 41 rooms instead of 49 phantoms).
  - Orphan reabsorption: `_reassign_orphan_fragments` merges sub-floor
    fragments separated from exactly one kept room by <= 3 px (the opening
    artifact scale) back into that room; fragments bordering 2+ rooms are
    never merged.
- **Measured impact:** interior coverage 0.605 → 0.646, 0.619 → 0.662,
  0.654 → 0.665 on the real plans; `capture_165134` recovers a 7th room,
  `mcginnies_pdf` a 41st. New `synth_corridor` case (4 ft corridor, 24 ft²
  closet below the compact floor, 3 doors) detects all 4 rooms at P/R
  1.0/1.0. All other gated metrics flat; gate passed before re-capture.
- **Residuals:** the 0.85+ coverage target needs the footprint denominator
  fix (#6 — the dimension strip still inflates `footprint_area_px`) and
  opening detection (#7 — unsealed gaps still merge/leak space). The closet
  partitions are found by the room map but not emitted as `walls[]` (Hough
  interior filter requires an exterior T-junction) — fold into #7
  (resolved there 2026-06-10 by the short-partition recovery pass).
- **Tests:** `tests/unit/test_room_split.py` (corridor floor keeps/drops,
  directional-close corner pocket, orphan merge one/two/zero neighbors);
  `tests/integration/test_synth_plans.py::TestCorridor*` (4 rooms, sub-25 ft²
  closet recovery, corridor aspect, coverage/closure floors).

## 6. Footprint confidence + parameterized morphology

- **Baseline evidence:** all baseline cases find a footprint, but stage [4]
  is single-path: when `find_footprint` picks wrong (title block, partial
  building), everything downstream is silently wrong. The hard-coded
  exclusion fractions (top 12%, bottom 18%, …) only fit one sheet layout.
- **Change:** score top-k footprint candidates (area x compactness x
  ink-density), expose a confidence value in the output, and retry with
  relaxed morphology when confidence is low.
- **Expected impact:** fewer catastrophic failures on unseen sheet layouts;
  measurable once more real cases are annotated (M1).
- **Risk + mitigation:** candidate ranking could flip on existing cases —
  baseline compare pins footprint area to +/-5%.
- **Tests:** unit tests with decoy blobs; confidence present in output schema.

## 7. Door detection (geometric) — IMPLEMENTED 2026-06-10

- **Baseline evidence:** doors recall 0.00 by construction against a >90%
  product target; the signal already existed — `snap_wall_endpoints`
  deliberately never bridges collinear endpoint gaps because they are
  doorways.
- **What landed:**
  - `Arqen/door_detect.py`: groups axis-aligned walls by orientation + axis
    (dedup tolerance), takes each adjacent collinear pair's span gap as a
    candidate when 1.5–5 ft wide, verifies the gap band of `wall_pair_mask`
    is essentially ink-free (dedup/cleanup splits still have wall ink), and
    rejects gaps with a raw-ink stroke running parallel to the wall axis
    across most of the span (window sill). Emits `doors[]` with `id`,
    `host_wall_id` (longer flank), `bbox_px`, `center_px`, `width_raw`,
    `is_exterior`, `evidence: "gap"`.
  - Wired into `analyze_page` after `snap_wall_endpoints` (walls and mask
    share crop coordinates); door coords shifted by `roi_offset` alongside
    walls; `"doors"` added to the output dict. Purely additive.
  - Scorer fix (measurement-side, like M2): `opening_score` is now
    max(bbox IoU, center-proximity) with the center tolerance scaled to
    ~3 ft via the prediction's `px_per_ft` — LabelMe door boxes include the
    swing arc (~3x3 ft) so IoU alone under-scored correct gap-tight
    detections.
  - Short-partition recovery (folded #5 residual): probabilistic Hough
    reliably misses sub-6-ft runs on a full sheet, so closet partitions never
    reached `walls[]`. A directional-morphological-opening pass finds
    2.5–12 ft double-stroke runs; a candidate is kept only when it
    T-junctions into an already-accepted wall (iterated, so stubs chain) and
    is not parallel to a wall at dimension-line offset (1–2.5 ft) — that
    second guard is what keeps fixture/dimension scraps out and protects
    cleanup's dimension-like classification of real walls. The pass is
    purely additive: recovered shorts are deduped among themselves, never
    re-merged into the stable interior set.
- **Measured impact:** synth doors 5/5 (P/R 1.0/1.0 on all three cases,
  including the corridor closet door; the 2 sill windows produce no FPs);
  `synth_corridor` walls 9 → 12 (closet partitions now emitted);
  `mcginnies_pdf` walls 127 → 145 (+18 dorm bathroom/closet partitions,
  rooms stable at 41, closure +0.017); captures unchanged. LabelMe 20-case
  batch: doors 9/308 recall at precision 1.00 — recall there is bounded by
  wall recall on those crops (both flanking walls must exist), not by the
  gap classifier.
- **Residuals:** real-plan door recall needs wall recall on cropped images
  to rise first; swing-arc confirmation can be added later under the same
  schema (`evidence` field).
- **Tests:** `tests/unit/test_door_detect.py` (collinear gap H/V, zero-gap
  sub-segment boundary, too-wide garage opening, non-collinear pairing,
  ink-filled gap, sill vs perpendicular door leaf, dedup/ids, exterior flag,
  diagonal walls); synth integration assertions flipped from TP==0 to
  recall/precision floors (two_room 1/1, l_shape 1/1, corridor >= 2/3 with
  precision 1.0).

## 8. Window detection (geometric)

- **Baseline evidence:** windows recall 0.00 — a category the product
  promises (project_context.md targets >90%) with no detector behind it.
- **Change:** within exterior wall bands, detect the sill-line signature
  (thin single/triple stroke between wall faces, the current
  `_find_wall_pairs` already keeps this ink) and emit `windows[]`. The door
  detector's sill discriminator (`door_detect._gap_has_sill`) is the seed:
  what it rejects as "not a door" is precisely the window signature.
- **Expected impact:** windows 0% → 60–80% recall on synthetic + annotated
  cases.
- **Risk + mitigation:** purely additive output fields; walls/rooms paths
  untouched — gate ensures no drift; synth renderer already draws windows
  with exact GT.
- **Tests:** synth cases carry window GT (currently asserting TP==0 — flip
  to recall floors when detection lands).

### 8a. Window Accuracy V2 — scored gate + flank gate + recall unlock — IMPLEMENTED 2026-06-16

Three geometric phases on top of the V1 detector, all gated to never regress
synth or FP-only cases (measure with `validation/window_metrics.py`):

- **Phase 1 — multi-cue confidence:** `_window_confidence` combines sill cover,
  open-gap, bilateral break, triple-line and a dimension penalty into a
  `confidence` field (emitted on every window + candidate). Calibration finding:
  re-weighting these cues trades precision against recall along the *same*
  frontier and cannot beat the V1 binary gate, so acceptance is kept
  F1-neutral and confidence is exposed for ranking/debug.
- **Phase 2 — wall-flank gate:** `_opening_flanked_by_wall` requires
  double-stroke wall ink to continue on both sides of the gap (along-axis
  coverage ≥ 0.50). Removes phantom-segment FPs over whitespace from
  overshooting footprint polygons. Effect: labeled FP 69 → 56, fp_only FP
  92 → 88.
- **Phase 3 — interior-envelope recall unlock:** also scan interior-tagged
  walls that lie on the building envelope (axis extremes over all walls);
  these are perimeter walls the Hough supplement mis-tagged. Gated with
  `strict_open` (open band or bilateral break required) to hold precision.
- **Measured (LabelMe + synth, fresh pipeline):** labeled micro
  F1 0.277 → 0.283 (TP 85, FP 69 → 56), all-real F1 0.241 → 0.247,
  fp_only FP 92 → 88, synth stays 1.0.

### 8b. Window Symbol Recall (V3) — DELIVERED 2026-06-16 (geometric symbol detector)

The deferred Phase 4 "template matching" was reframed and delivered as a
**geometric** symbol detector after a measurement-first diagnosis (the original
deferral assumed missing host walls were the binding constraint). Re-probing
`labelme_fp_20` showed the host walls are present and scanned, but the windows
are drawn as a **regularly spaced series of glyph markers on a continuous
centreline** — the wall pair never breaks, so the opening-based strategies
structurally cannot generate candidates for them.

Implementation (`window_detect.py`, strategy `symbol_on_wall`):

- `_symbol_runs_along_wall` scans a thin centre-channel band of feature ink
  (`ink_mask` present **and** `wall_pair_mask` absent, so wall strokes and
  crossing walls drop out) for compact on-axis marker blobs. A series of
  ≥ `SYMBOL_MIN_MARKERS` (3) with spacing coeff. of variation ≤
  `SYMBOL_SPACING_CV_MAX` (0.55) is emitted as one window per marker
  (`evidence: "symbol"`).
- Precision guards: wall pair must stay continuous (not an open gap — those
  belong to the opening strategies), wall ink must flank the glyph, dimension
  strings are rejected, and on-axis ink continuing far perpendicular is rejected
  as a crossing wall (`_symbol_perp_extends`). The periodic-series requirement is
  the core guard — plain walls and FP-only plans have an empty centre channel
  (measured ≈ 0), so they do not fire.

Measured impact (`validation/window_metrics.py --run-pipeline`, vs. the V2
baseline where the symbol path emits no candidates):

- `labelme_fp_20`: recall 0.12 → **0.26** (9 → 20 TP), precision 0.75 → **0.87**.
- Labeled aggregate F1 0.283 → **0.311** (TP +10, FP flat at 57).
- FP-only **not regressed**: 98 → **90** FP (symbol windows merge-dedup with some
  opening-path FPs).
- Synth stays **1.0**; a new glyph-on-wall fixture (`render_symbol_window_plan`)
  asserts the convention is detected end-to-end.

Still bounded: genuine interior-wall / courtyard glazing (e.g. `fp_19`,
`fp_31_2`) remains out of scope (those walls are not scanned by design), and the
hatched-band convention (`fp_60`) is handled by the opening path, not this one.
NCC/ML template matching remains deferred unless the geometric detector plateaus.

## 9. Dimension extraction

- **Baseline evidence:** dimensions recall 0.00; dimension strings are
  deliberately discarded as annotation ink (`_find_wall_pairs`,
  `drop_dimension_like_walls`).
- **Change:** capture (don't just discard) dimension-line candidates
  (single-stroke lines with end ticks parallel to a nearby wall), crop the
  text region, and reuse the web app's existing LLM OCR for the value; emit
  `dimensions[]`. Use parsed values to cross-validate `px_per_unit` (synergy
  with #3).
- **Expected impact:** dimensions 0% → ~70% recall; independent scale check.
- **Risk:** additive; OCR dependency stays out of the core CV path (service
  layer composes the two).
- **Tests:** synth dimension strings carry exact GT (60', 40', 70').

## 10. ML hybrid wall graph

As ARCHITECTURE.md Phase 3: CNN/transformer wall segmentation fused with the
geometric pipeline. Highest ceiling (curved walls, style variance — §5
unsupported list), but weeks of effort, training data needs, and a wholesale
behavior change. Do **after** the geometric path is measured, hardened, and
the annotated real-plan set (M1) exists to evaluate it honestly.

---

## Per-change protocol (required for every code change)

1. **Why:** cite the baseline metric / failure the change addresses.
2. **Expected impact:** which metrics should move, in which direction, and
   which must stay flat.
3. **Implement** with new/updated unit tests for the touched stage.
4. **Run:** `python -m pytest -m unit && python -m pytest -m integration`.
5. **Gate:** `python validation/compare_to_baseline.py` — zero regressions
   beyond tolerance, or a written justification plus
   `python validation/capture_baseline.py` to accept the new baseline (commit
   the updated `baseline.json` and note the change in
   `docs/baseline_metrics.md`).
