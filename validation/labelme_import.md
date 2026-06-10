# Importing LabelMe annotations

Your 20 annotated drawings can be imported as scored validation cases.

## Quick start

```bash
python validation/import_labelme_cases.py ^
  --labelme-dir "C:\Users\jakep\Downloads\arqen-labs-jun\arqen-labs-jun\floor-plan-annotated" ^
  --images-root "C:\Users\jakep\Downloads\arqen-labs-jun\arqen-labs-jun\floor-plan-cropped" ^
  --pilot FP_86_2 ^
  --run-score
```

Import all 20:

```bash
python validation/import_labelme_cases.py --labelme-dir ... --images-root ... --all --run-score
```

Each case lands in `validation/cases/labelme_fp_XX/` with `image.png`, `manifest.json`, `ground_truth.json`, and `labelme_conversion.json` (skipped labels, wall approximation notes).

## Label mapping

| LabelMe label | Arqen category |
|---------------|----------------|
| Room | `rooms` |
| Wall | `walls` (polygon band → approximate centerline) |
| Door | `doors` |
| Window | `windows` |
| Text | `labels` (uses `description` field when set) |
| Toilet, Shower, … | skipped (fixtures — not in schema yet) |

## Scale

Cropped PNGs have no embedded scale. The importer auto-infers `scale`/`dpi` via
multi-hypothesis calibration (`infer_crop_calibration` in
`validation/arqen_validation/labelme.py`). Re-run without re-importing:

```bash
python validation/import_labelme_cases.py --recalibrate --cases-root validation/cases
```

Edit `scale`/`dpi` in `manifest.json` per case when you know the true drawing scale.

## Scoring and triage

**Wall span-coverage recall** (`report.json` → `categories.walls.coverage.recall`)
is the fair wall metric on these crops — strict 1:1 wall recall is misleading when
only a few walls are annotated (see `sparse_gt` bucket in triage).

```bash
python validation/score_labelme_cases.py
python validation/triage_labelme_cases.py
```

Annotate **all interior partitions** as Wall shapes (not just rooms) for meaningful strict wall scores.

## After import

```bash
python validation/run_score.py --case labelme_fp_86_2
python validation/capture_baseline.py --case labelme_fp_86_2
```

Read `report.json` → `missing_objects` and `false_positives`, and
`validation/reports/labelme_triage.md` for per-case failure buckets.
