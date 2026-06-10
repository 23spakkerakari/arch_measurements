# LabelMe batch score summary

Generated after importing 20 annotations from
`Downloads/arqen-labs-jun/arqen-labs-jun/floor-plan-annotated`.

| Case | Pred/GT rooms | Room recall | Wall recall | Interior coverage |
|------|---------------|-------------|-------------|-------------------|
| labelme_fp_6_1 | 16/9 | **1.00** | 0.40 | 0.72 |
| labelme_fp_7_2 | 5/5 | **1.00** | 0.67 | 0.65 |
| labelme_fp_87 | 2/2 | **1.00** | 0.50 | 0.63 |
| labelme_fp_36_2 | 3/1 | **1.00** | 0.00 | 0.73 |
| labelme_fp_10 | 24/16 | 0.75 | 0.23 | 0.69 |
| labelme_fp_60 | 20/32 | 0.50 | 0.21 | 0.61 |
| labelme_fp_48_1 | 3/2 | 0.50 | 0.33 | 0.58 |
| labelme_fp_1 | 13/22 | 0.23 | 0.31 | 0.51 |
| labelme_fp_54_2 | 5/3 | 0.33 | **1.00** | 0.51 |

Worst room recall: FP_19 (0.01), FP_20 (0.00), FP_27 (0.00) — often where few
walls were annotated in LabelMe (pipeline can't close room boundaries) or
auto px/ft calibration is off.

## Next fixes (from FN lists in per-case `report.json`)

1. **Wall GT quality** — LabelMe walls are thick polygons; we approximate
   centerlines. Re-annotate critical walls as lines for better wall scores.
2. **Per-case scale** — edit `manifest.json` `scale`/`dpi` when true drawing
   scale is known; auto-infer uses a 50 ft span guess.
3. **Pipeline** — cases with high pred room count but low recall (FP_35_1:
   8 pred / 2 GT) need phantom-room work; cases with 0 pred rooms need footprint
   / interior-wall fixes (#6, #7).

Re-run: `python validation/score_labelme_cases.py`
