# Template case

1. Copy this folder to `validation/cases/<your_plan_id>/`.
2. Replace `plan.pdf` with the original floor plan PDF (or set `"image"` in
   `manifest.json` to a PNG/JPG raster — paths are relative to the case folder).
3. Fill in `manifest.json` (`scale`, `dpi`, optional `roi`, `doorway_close_ft`).
4. Annotate all objects in `ground_truth.json` (guide below).
5. Run scoring: `python validation/run_score.py --case <your_plan_id> --run-pipeline`
6. Re-capture the baseline so the new case is gated:
   `python validation/capture_baseline.py`

## Annotation guide

Annotate in **pixel coordinates of the rasterized page at the manifest DPI**
(the same raster the pipeline sees). Tip: rasterize once and annotate over it:

```bash
python -c "import fitz; d=fitz.open('plan.pdf'); p=d.load_page(0); p.get_pixmap(dpi=300).save('raster.png')"
```

| Category | Geometry to record | Conventions |
|----------|--------------------|-------------|
| `rooms` | `bbox_px` (or `polygon_px` for non-rectangular) | Interior region of the room, inside wall faces |
| `walls` | `px_coords [x1,y1,x2,y2]` | Wall **centerline** (midway between the two drawn strokes); add `facing`, `is_exterior`, `length_raw` (ft) |
| `doors` | `bbox_px` or `center_px` | Box over the door opening incl. swing arc origin; add `host_wall_id` |
| `windows` | `bbox_px` or `center_px` | Box over the window symbol within the wall band |
| `labels` | `text` (+ `bbox_px`) | Room name text as printed; link `room_id` |
| `dimensions` | `value_raw` (decimal ft) + `text` | One entry per dimension string; `center_px` at the text |

Rules of thumb:

- Every GT object needs a unique `id` within its category.
- Walls shorter than ~2 ft and purely decorative linework can be omitted.
- Validate against the schema: `validation/schema/ground_truth.schema.json`
  (the unit suite runs this automatically for committed cases).
- A case **without** `ground_truth.json` is still useful: it is tracked via
  structural metrics and the baseline snapshot (counts, area, closure).

## Synthetic cases

`synth_*` cases are generated — do not hand-edit them. Regenerate with:

```bash
python validation/generate_synth_cases.py
```
