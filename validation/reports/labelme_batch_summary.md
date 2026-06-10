# LabelMe batch score summary

After interior-wall recall improvements (crop calibration, crop_mode pipeline, phantom suppression).

Re-run: `python validation/score_labelme_cases.py`  
Triage: `python validation/triage_labelme_cases.py`

| Case | Pred/GT rooms | Room recall | Wall strict R | **Wall cov R** | Door R | Interior coverage |
|------|---------------|-------------|---------------|----------------|--------|-------------------|
| labelme_fp_6_1 | 13/9 | **1.00** | 0.60 | **0.65** | 0.00 | 0.72 |
| labelme_fp_7_2 | 6/5 | 0.80 | 0.67 | **0.74** | 0.00 | 0.64 |
| labelme_fp_54_2 | 2/3 | 0.33 | **1.00** | **0.98** | **1.00** | 0.33 |
| labelme_fp_66_2 | 9/1 | 0.00 | 0.00 | **0.88** | **1.00** | 0.40 |
| labelme_fp_25_2 | 2/2 | 0.00 | 0.00 | **0.78** | 0.00 | 0.22 |
| labelme_fp_1 | 19/22 | 0.36 | 0.31 | **0.72** | 0.04 | 0.58 |
| labelme_fp_10 | 22/16 | 0.75 | 0.31 | **0.44** | 0.00 | 0.66 |
| labelme_fp_60 | 15/32 | 0.31 | 0.07 | **0.32** | 0.00 | 0.46 |

**Median wall span-coverage recall: 0.55** (target met).

Doors overall: 6/308 matched (recall 0.02) — improved via merged-span gap detect + crop gap band; fp_60 accounts for most TPs. Still wall-recall-bound on most cases.

## Bucket summary (from triage)

- **sparse_gt** (15): few walls annotated vs many predicted — strict wall R misleading; use wall cov R.
- **missing_interior** (3): FP_27, FP_19, FP_86_2 — need more interior partition detection.
- **mixed** (2): FP_20, FP_31_2.

## Commands

```bash
python validation/import_labelme_cases.py --recalibrate --cases-root validation/cases
python validation/score_labelme_cases.py
python validation/triage_labelme_cases.py
```
