# arch_drawing1_2024 — window takeoff regression

Source: `3.-2024.02.29-Architectural-Construction-Drawing1.pdf` (office layout, 3/8" = 1 ft).

Known issues from initial takeoff (screenshot):

- **False negatives:** missed window symbols on top wall (left) and left wall (above win1)
- **False positives:** win4, win7 on solid wall sections
- **Oversized:** win1–win3, win9, win12 merging multiple adjacent window symbols

## Commands

```bash
python validation/run_score.py --case arch_drawing1_2024 --predict
python Arqen/debug_windows.py --image validation/cases/arch_drawing1_2024/raster.png --scale "3/8in=1ft" --dpi 150 --out debug_runs/arch_drawing1_2024
python validation/run_score.py --case arch_drawing1_2024
```

Annotate `windows[]` in `ground_truth.json` before scoring.
