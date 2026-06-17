# TRDI Overall Floor Plan @ 1/8" (144 DPI)

Regression case for window over-detection on the **Overall Floor Plan** sheet
(`3.-2024.02.29-Architectural-Construction-Drawing.pdf`, page 0).

## Config (matches web app)

- Scale: `1/8in=1ft`
- DPI: 144 (px/ft ≈ 18)
- `crop_mode`: true

## Ground truth

Eight exterior office windows (four on the north wall, four on the west wall).
Boxes were placed on window symbols at full-sheet pixel coordinates (4896×3168).

Note: on this PDF the plan geometry sits on the right portion of the page
(x ≈ 2600+), not the upper-left margin.

## Baseline (Phase 0)

Run:

```bash
python validation/window_metrics.py --cases trdi_overall_1_8 --run-pipeline
```

**Recorded baseline (2026-06-16, before Phase 2):** P=0.32, R=1.00, F1=0.48, TP/FP/FN = 8/17/0

**After Phase 2 envelope gate:** P=0.36, R=1.00, F1=0.53, TP/FP/FN = 8/14/0

**After over-split / long-wall controls:** P=0.39, R=0.88, F1=0.54, TP/FP/FN = 7/11/1

Remaining FPs are mostly on the warehouse south and lobby north exterior (outside the
8 annotated office windows). The single FN is a west-wall opening split across a
structural gap — one fragment center falls outside the 40 px matching tolerance.

## Related case

`trdi_overall` uses a downscaled production input (effective dpi 71) with ROI;
this case uses the native 144 DPI full sheet.
