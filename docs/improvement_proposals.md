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
| 5 | Interior coverage recovery (corridors/small rooms) | Medium — interior coverage 0.60–0.72 → 0.85+ | Medium | Medium-high | interior coverage, rooms R |
| 6 | Footprint confidence + parameterized morphology | Medium — protects the single highest-leverage failure point (stage [4] has no fallback) | Medium | Medium | error rate, area, polygon |
| 7 | Door detection (geometric) | Unlocks doors 0% → ~60–80% recall | High | Medium | doors P/R/F1 |
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

## 5. Interior coverage recovery

- **Baseline evidence:** interior coverage 0.640 / 0.603 / 0.717 on real
  plans — up to 40% of the footprint belongs to no room. Causes: fixed
  `min_room_ft2=25` (kills corridors/closets), aggressive wall-band erosion
  (`wall_thickness_px * 2`), global `doorway_close_ft`.
- **Change:** lower the area floor for high-aspect cells (corridors), scale
  erosion with actual stroke gap instead of assumed thickness, and reassign
  orphan interior pixels to the adjacent room cell.
- **Expected impact:** interior coverage → 0.85+; rooms recall up on real
  plans (more real rooms pass the filter).
- **Risk + mitigation:** more cells = more phantom-room exposure — land
  proposal #1 first; gate on room-count tolerances.
- **Tests:** corridor fixture in the synth renderer with exact GT; coverage
  floor test.

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

## 7–8. Door and window detection

- **Baseline evidence:** doors/windows recall 0.00 — categories the product
  promises (project_context.md targets >90%) with no detector behind them.
- **Change (doors):** detect doorway gaps along walls (the room-split walk
  already finds room-transition points); classify gap + swing-arc ink
  (Hough circles / arc fit) as a door; emit `doors[]` with `host_wall_id`.
- **Change (windows):** within exterior wall bands, detect the
  sill-line signature (thin single/triple stroke between wall faces, the
  current `_find_wall_pairs` already keeps this ink) and emit `windows[]`.
- **Expected impact:** doors/windows 0% → 60–80% recall on synthetic +
  annotated cases; closes the biggest gap vs product targets.
- **Risk + mitigation:** purely additive output fields; walls/rooms paths
  untouched — gate ensures no drift; synth renderer already draws windows and
  door gaps with exact GT.
- **Tests:** synth cases already carry door/window GT (currently asserting
  TP==0 — flip these to recall floors when detection lands).

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
