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

Cropped PNGs have no embedded scale. The importer defaults to `1in=1ft` in `manifest.json`. **Room/wall matching uses pixel geometry**, so scores are meaningful even before scale is corrected. Edit `scale` per case when you know the drawing scale.

## After import

```bash
python validation/run_score.py --case labelme_fp_86_2
python validation/capture_baseline.py --case labelme_fp_86_2
```

Read `report.json` → `missing_objects` and `false_positives` to prioritize pipeline fixes (#6 footprint, #7 doors, interior walls).
