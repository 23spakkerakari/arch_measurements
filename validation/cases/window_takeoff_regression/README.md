# Window takeoff regression case

Use this folder template when capturing a real plan with known window errors
(missed detections, misaligned overlays, clustered false positives).

## Setup

1. Copy this folder to `validation/cases/<case_id>/`.
2. Add `plan.pdf` or `plan.png` and fill in `manifest.json` (scale, DPI, ROI).
3. Run the pipeline and save output:
   ```bash
   python validation/run_score.py --case <case_id> --predict
   ```
4. Annotate `ground_truth.json` with `windows[]` entries (`bbox_px` or `center_px`).
   Focus on the failing windows first (false negatives and misaligned boxes).
5. Score:
   ```bash
   python validation/run_score.py --case <case_id>
   ```

## Debug a single plan

```bash
python Arqen/debug_windows.py \
  --image path/to/plan.png \
  --scale "3/8in=1ft" \
  --dpi 150 \
  --out debug_runs/windows
```

Outputs:

- `window_debug_overlay.png` — green = accepted candidates, red = rejected, orange = final windows
- `window_candidates.json` — per-candidate `reject_reason` tags (`no_sill`, `ink_not_open`, `dimension_line`, etc.)

## Recommended GT windows

For the misalignment / missed-window workflow, annotate at minimum:

- One **false negative** (clear window symbol, no detection)
- One **misaligned** detection (overlay shifted vs symbol)
- One **cluster** region if multiple IDs appear on a single opening
