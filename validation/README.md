# Arqen Validation

Automated scoring of extracted floor-plan geometry against curated ground truth.

## Folder layout

Each floor plan is a **case** under `validation/cases/<case_id>/`:

```
validation/cases/<case_id>/
  plan.pdf              # Original PDF (or symlink)
  manifest.json         # Scale, DPI, ROI, metadata
  ground_truth.json     # Curated rooms, walls, doors, windows, labels, dimensions
  prediction.json       # Pipeline output (generated)
  report.json           # Scoring output (generated)
```

Copy `cases/_template/` when adding a new plan.

## Ground truth schema

`ground_truth.json` supports six object categories:

| Category | Required geometry | Notes |
|----------|-------------------|-------|
| `rooms` | `bbox_px` or `polygon_px` | Optional `label`, `area_raw` |
| `walls` | `px_coords` `[x1,y1,x2,y2]` | Optional `facing`, `length_raw`, `is_exterior` |
| `doors` | `bbox_px` or `center_px` | Optional `host_wall_id` |
| `windows` | `bbox_px` or `center_px` | Optional `host_wall_id` |
| `labels` | `text` | Optional `bbox_px`, `room_id` |
| `dimensions` | `value_raw` or `text` | Optional `bbox_px`, `unit` |

See `schema/ground_truth.schema.json` for the full JSON schema.

Predictions use the same fields. Arqen `out.json` / `analyze_page()` output is normalized automatically. Per-wall `windows` counts in LLM output are promoted to window objects for scoring.

## Metrics

For each category the scorer reports:

- **Precision** — matched predictions / all predictions
- **Recall** — matched ground-truth objects / all ground truth
- **F1 score**
- **Mean IoU** — average overlap score on matched pairs
- **Missing objects** — false negatives (in ground truth, not matched)
- **False positives** — predictions with no ground-truth match

### Space boundary closure

Every report also includes a `closure` section with three measures:

| Measure | Needs GT? | Definition |
|---------|-----------|------------|
| `wall_network.closure_rate` | No | Fraction of predicted wall endpoints terminating at another wall (corner or T-junction) within tolerance. Dangling endpoints mean the network cannot enclose space. |
| `room_boundary.closure_rate` | Yes | Fraction of GT rooms whose perimeter is >= 95% covered by predicted walls within tolerance. Per-room `boundary_coverage` is also reported. |
| `interior_coverage.coverage` | No | Sum of predicted room areas / detected footprint area. |

Tolerance defaults to `max(12px, 2.0 * px_per_ft)` (about 2 ft); override with
`score_prediction(..., closure_tolerance_px=...)`.

Matching is greedy best-pair by overlap score with category-specific thresholds:

| Category | Default threshold | IoU / score method |
|----------|-------------------|--------------------|
| rooms | 0.50 | polygon IoU, else bbox IoU |
| walls | 0.55 | colinear segment overlap IoU |
| doors | 0.45 | bbox IoU or center distance |
| windows | 0.45 | bbox IoU or center distance |
| labels | 0.70 | text match + spatial overlap |
| dimensions | 0.75 | value tolerance + proximity |

Override thresholds: `--threshold walls=0.60`

## Usage

Score one case (prediction must exist):

```bash
python validation/run_score.py --case demo_minimal
```

Run pipeline then score:

```bash
python validation/run_score.py --case my_plan --run-pipeline
```

Score all cases that have `prediction.json`:

```bash
python validation/run_score.py --all
```

Ad-hoc comparison:

```bash
python validation/run_score.py \
  --ground-truth validation/cases/demo_minimal/ground_truth.json \
  --prediction validation/cases/demo_minimal/prediction.json
```

Reports are written to `cases/<case_id>/report.json` (or `validation/reports/` with `--all`).

## Success targets

From `docs/project_context.md`:

- Room accuracy >95%
- Wall accuracy >95%
- Window accuracy >90%
- Door accuracy >90%

Use **recall** as the primary “accuracy” metric for missed-object detection, and **F1** for overall category quality.

## Adding a new case

1. Copy `cases/_template/` to `cases/<your_plan_id>/`.
2. Add `plan.pdf` (or set `pdf_path`/`image` in `manifest.json`).
3. Annotate `ground_truth.json` from the PDF at the same DPI/scale as the pipeline
   (see `cases/_template/README.md` for the annotation guide).
4. Generate `prediction.json` via `--run-pipeline` or copy from `Arqen/out.json`.
5. Run `python validation/run_score.py --case <your_plan_id>`.

Cases without `ground_truth.json` (e.g. the `capture_*` replays) are still
tracked via structural metrics and the baseline snapshot.

## Baseline and regression gate

```bash
python validation/capture_baseline.py      # run all cases, write baselines/baseline.json
python validation/compare_to_baseline.py   # re-run and diff vs baseline (CI gate)
python validation/compare_to_baseline.py --update   # accept current behavior as new baseline
```

`capture_baseline.py` records, per case: structural metrics (wall/room counts,
total area, closure rates, interior coverage) plus per-category P/R/F1 where
ground truth exists. `compare_to_baseline.py` re-runs every baselined case and
fails (exit 1) on any regression beyond the tolerances in
`arqen_validation/compare.py` (counts +/-max(2, 10%), area +/-5%, closure drop
> 0.05, F1/recall drop > 0.02).

**Workflow for every pipeline code change:**

1. Explain why the change is needed and the expected impact.
2. `python -m pytest -m unit` and `python -m pytest -m integration` must pass.
3. `python validation/compare_to_baseline.py` must pass (or justify and
   `--update` the baseline with the improvement documented).

## Test suite

```bash
python -m pytest -m unit          # fast per-stage unit tests (tests/unit/)
python -m pytest -m integration   # end-to-end runs on representative plans
```

Synthetic cases (`cases/synth_*`) are generated with exact ground truth by
`validation/generate_synth_cases.py` — regenerate rather than hand-edit.
