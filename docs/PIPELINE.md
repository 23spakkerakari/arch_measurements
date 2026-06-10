# Arqen Pipeline Map

End-to-end map of how architectural floor plans become structured geometry and downstream engineering inputs.

**Authoritative implementation:** `Arqen/preprocess.py` (`analyze_page()`).  
**Do not use:** `Arqen/segmentation_pipeline/` — stale fork.

For deployment topology, failure modes, and roadmap, see [`Arqen/ARCHITECTURE.md`](../Arqen/ARCHITECTURE.md).

---

## Pipeline Overview

```
PDF
 ↓
Image Processing
 ↓
Line Detection
 ↓
Symbol Detection
 ↓
Text OCR
 ↓
Room Segmentation
 ↓
Geometry Construction
 ↓
Structured JSON
 ↓
HVAC Load Inputs
```

Two analysis tracks share the CV core:

| Track | Entry | Notes |
|-------|-------|-------|
| **Production CV** | `preprocess.py` CLI, `cv_service.py` `/cv-analyze` | Deterministic OpenCV |
| **ArchTakeoff web** | `claude_demo/arch-takeoff/` | CV-first; Claude for scale, labels, fallback |

---

## Stage 1 — PDF

**Purpose:** Accept a floor-plan document and produce a raster image the CV pipeline can process.

### Entry points

| Caller | File | How PDF arrives |
|--------|------|-----------------|
| CLI | `Arqen/preprocess.py` → `main()` | Path argument; PyMuPDF rasterizes server-side |
| HTTP API | `Arqen/cv_service.py` → `POST /cv-analyze` | Client sends **pre-rasterized** base64 image (not raw PDF) |
| Web app | `claude_demo/arch-takeoff/js/upload.js` | PDF.js renders first page in browser |
| Validation | `validation/arqen_validation/runner.py` | Manifest `pdf` or `image` field |
| Viewer | `Arqen/viewer.py` | PDF overlay for debugging |

### Key functions

| Function | File | Output |
|----------|------|--------|
| `pdf_to_images(path, dpi)` | `Arqen/preprocess.py` | `list[np.ndarray]` RGB, one array per page |
| `renderPdfFirstPage(dataUrl)` | `claude_demo/arch-takeoff/js/upload.js` | Canvas PNG at **144 DPI** (`scale=2` × 72) |
| `_rasterize_page(pdf_path, page, dpi)` | `Arqen/viewer.py` | PNG bytes for HTML viewer |

### Mechanism

- **Python:** PyMuPDF (`fitz`); zoom = `dpi / 72`; RGBA stripped to RGB.
- **Web:** PDF never reaches Python — browser rasterizes, then POSTs base64 to CV service.

### Configuration

| Parameter | CLI default | HTTP / web default |
|-----------|-------------|-------------------|
| `dpi` | 300 | 150 (HTTP), 144 (browser) |
| `page` | 1 (1-indexed) | First page only in web |

### Risks

- **DPI mismatch:** Same scale string at 300 DPI (CLI) vs 144 DPI (web) produces different `px_per_unit` and systematic length errors. Always pass the DPI used for rasterization.

---

## Stage 2 — Image Processing

**Purpose:** Convert raw RGB pixels into calibrated, cleaned binary masks that isolate building ink from sheet noise.

**Orchestrator step:** `analyze_page()` stages 0–2 in `Arqen/preprocess.py`.

### Sub-stages

```
RGB image
  → [optional] ROI crop (_crop_to_roi)
  → Scale calibration (parse_scale → px_per_unit)
  → preprocess() → (binary_footprint_mask, wall_pair_mask)
```

### 2a — ROI crop (optional)

| Function | Input | Output |
|----------|-------|--------|
| `_crop_to_roi(image, roi)` | RGB + `{x0_pct, y0_pct, x1_pct, y1_pct}` | Cropped image, `(offset_x, offset_y)`, full W/H |

When ROI is set: `apply_margins=False`, `use_exclusion=False` — hard-coded sheet-fraction exclusion zones are disabled.

### 2b — Scale calibration

| Function | File | Input | Output |
|----------|------|-------|--------|
| `parse_scale(scale_str, dpi, output_unit)` | `Arqen/scale_parse.py` | e.g. `"1/4in=1ft"`, `"1:100"` | `{px_per_unit, unit_label}` |

Supported formats: equality (`=`), ratio (`:`), arch feet-inches (`1'-0"`), ft, in, mm. Meters on the real-world side are not yet parsed.

### 2c — `preprocess(image, px_per_unit, apply_margins)`

| Step | Function | Description |
|------|----------|-------------|
| Wall ink extraction | `_extract_wall_lines()` ×2 | Full-res + margin-filtered masks |
| Double-stroke filter | `_find_wall_pairs()` | Keep H/V parallel stroke pairs only |
| Grid removal | `_strip_spanning_grid_lines()` | Drop long border/grid runs |
| Downscale | ÷4 (`DOWNSCALE=4`) | Speed for morphology |
| Gap bridging | Dilate + directional CLOSE | Bridge door/window gaps ≤12 real units |
| Upscale | Back to full resolution | `binary` for footprint flood-fill |

**Helpers:**

- `_blank_sheet_margins()` / `_build_exclusion_mask()` — title block zones (top 12%, bottom 18%, etc.)
- `wall_pair_gap_range(px_per_unit)` — adaptive min/max gap between wall stroke pairs

**Outputs:**

| Mask | Role |
|------|------|
| `binary` | Footprint flood-fill input |
| `wall_pair_mask` | Double-stroke wall ink for snap, Hough, room cut layer |

### Debug artifacts

`debug_pipeline.py` writes `01_mask_full` through `04_binary_footprint_input` for this stage.

---

## Stage 3 — Line Detection

**Purpose:** Find exterior and interior wall centerlines as orthogonal segments with real-world lengths.

### Exterior walls (polygon path)

| Step | Function | File | Output |
|------|----------|------|--------|
| Flood fill | `flood_fill_interior(binary)` | `preprocess.py` | Solid building mask |
| Component pick | `find_footprint(binary, use_exclusion)` | `preprocess.py` | Single largest component |
| Contour | `find_footprint_contour(mask)` | `preprocess.py` | External contour |
| Simplify | `simplify_polygon(contour, epsilon_factor)` | `preprocess.py` | Polygon vertices |
| Segment | `extract_wall_segments(polygon, min_length_px)` | `extract_wall_segments_class.py` | `(x1,y1,x2,y2)` tuples |
| Filter | `_filter_wall_segments(...)` | `preprocess.py` | ROI / span filtered |
| Snap | `snap_segments_to_walls(segments, wall_pair_mask)` | `preprocess.py` | Segments aligned to ink |

**Segment pipeline** (`extract_wall_segments_class.py`):

```
polygon_to_segments()
  → filter_short_segments()
  → filter_non_orthogonal_segments()   # ±10° tolerance
  → merge_collinear_segments()
```

### Interior walls (Hough supplement)

| Function | File | Role |
|----------|------|------|
| `detect_hough_segments(wall_mask, bbox)` | `room_wall_split.py` | HoughLinesP on cropped mask |
| `find_interior_segments()` | `room_wall_split.py` | T-junction to exterior boundary |
| `_hough_supplement()` | `preprocess.py` | Filter, pair-check, dedup candidates |
| `segment_traces_exterior()` | `room_wall_split.py` | Reject lines tracing exterior |
| `merge_and_deduplicate_segments()` | `preprocess.py` | Coaxial merge |

### Post-processing

| Function | Passes |
|----------|--------|
| `cleanup_wall_list()` | `drop_duplicate_exterior_strokes`, `drop_dimension_like_walls`, `drop_spanning_coaxial_walls`, `consolidate_coaxial_wall_duplicates`, `drop_redundant_exterior_spans` (×2 each) |

### Facing assignment

`assign_segment_facings()` — exterior: outward normal probe; interior: adjacency graph + bbox heuristics. Image-up = **North**.

### Constraints

- Orthogonal walls only (angled edges dropped)
- Single building per sheet (largest footprint component)
- Walls assumed drawn as **double parallel strokes**

### Debug artifacts

`05_footprint_component` through `11_final_walls` in `debug_pipeline.py`.

---

## Stage 4 — Symbol Detection

**Purpose:** Identify plan symbols — doors, windows, fixtures — as discrete geometry objects.

### Current status: **not implemented in CV path**

| Symbol | CV behavior | Where it appears |
|--------|-------------|------------------|
| **Doors** | Morphological bridging only | `doorway_close_ft` in `build_room_label_map()` seals gaps in `cut_layer` |
| **Windows** | Footprint close bridges gaps ≤12 ft | `preprocess()` adaptive `close_k_size` |
| **Doors/windows as JSON** | Not emitted | — |

### Indirect handling

Openings are treated as **gaps to bridge**, not objects to detect:

1. `preprocess()` morphological CLOSE bridges doorway/window breaks in the footprint mask.
2. `build_room_label_map()` uses `doorway_close_ft` (default 2.5 ft) to seal door openings in the room cut layer.

### Where symbols exist elsewhere

| Layer | Evidence |
|-------|----------|
| **LLM fallback** | `analysis.js` → Claude vision prompt can request window counts per wall |
| **Validation ground truth** | `validation/cases/*/ground_truth.json` — `doors[]`, `windows[]` for scoring |
| **Synthetic tests** | `validation/arqen_validation/synth.py` — draws door gaps and window openings |

### Dimension-string rejection

`_find_wall_pairs()` and `drop_dimension_like_walls()` explicitly filter single-stroke annotation ink (how dimension strings are typically drawn).

---

## Stage 5 — Text OCR

**Purpose:** Read plan text — scale notation, room names, dimension callouts.

### Current status: **no OCR library in CV path**

No Tesseract, EasyOCR, or similar. Dimension strings are **filtered out**, not read.

| Capability | Implementation | Deterministic? |
|------------|----------------|----------------|
| Scale string parsing | `parse_scale()` — user-supplied, not read from image | Yes |
| Scale auto-detect | `detectScaleQuick()` in `analysis.js` — Claude vision | No |
| Room name OCR | `assignRoomLabels()` in `analysis.js` — Claude maps R1/R2… to plan text | No |
| Dimension callouts | `drop_dimension_like_walls()` removes parallel offset strokes | N/A (rejection, not extraction) |
| Full vision takeoff | Claude fallback when CV fails | No |

### Validation schema (target, not CV output)

`validation/schema/ground_truth.schema.json` defines six categories:

```
rooms, walls, doors, windows, labels, dimensions
```

`dimensions[]` entries carry `value_raw` / `text` — used for ground truth and LLM predictions only.

### Web overlay aid

`coords.js` → `cvResultToAnalysis()` builds client-side `dimension_lines[]` from wall segments (visual overlay, not OCR).

---

## Stage 6 — Room Segmentation

**Purpose:** Partition the building interior into labeled room cells and split exterior walls by adjacent room.

**Primary file:** `Arqen/room_wall_split.py`  
**Entry:** `split_exterior_walls_by_room()` called from `analyze_page()`

### Flow

```
exterior segments + wall_pair_mask + contour
  → detect_hough_segments() + find_interior_segments()   # interior partitions
  → build_room_label_map()                                 # connected components
  → walk_wall_and_split_by_room()                          # per-room exterior runs
  → runs_to_sub_segments()                                   # w3.s1, room_id IDs
```

### `build_room_label_map()` internals

1. `interior_mask` — filled footprint, eroded by wall thickness
2. `cut_layer` — wall ink + interior partition lines (+ endpoint extension)
3. Morphological CLOSE with `close_kernel_px = doorway_close_ft × px_per_unit`
4. `room_mask = interior_mask AND NOT cut_layer`
5. `connectedComponentsWithStats` — filter by `min_room_area_px` (default **25 ft²**)

### Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `doorway_close_ft` | 2.5 | Door gap sealing in cut layer |
| `min_room_ft2` | 25.0 | Minimum room area |
| `min_segment_ft` | 4.0 | Minimum exterior sub-segment |
| `near_tol` | 15 px | Interior/exterior junction tolerance |
| `room_debug_dir` | None | Writes `interior_mask.png`, `cut_layer.png`, `room_mask.png` |

### Output per room

```json
{
  "id": "R1",
  "area_px": 12345,
  "centroid_px": [cx, cy],
  "bbox_px": [x0, y0, x1, y1],
  "area": "400.0 ft²",
  "area_raw": 400.0
}
```

Room **labels** (e.g. "Kitchen") are added later by Claude in the web path — not by CV.

### Debug artifacts

`12_room_mask`, `13_cut_layer`, `14_exterior_splits` in `debug_pipeline.py`.

---

## Stage 7 — Geometry Construction

**Purpose:** Assemble measured, facing-assigned geometry primitives into a coherent building model.

Geometry is built incrementally inside `analyze_page()` — there is no explicit wall-graph or junction model in output.

### Primitives constructed

| Primitive | Construction | Consumer fields |
|-----------|--------------|-----------------|
| Footprint polygon | Contour → `approxPolyDP` | `footprint_polygon_px`, `polygon_vertices` |
| Footprint AABB | Min/max of polygon | `footprint_bbox_px` |
| Exterior segments | Polygon edges → snap → room split | `walls[]` with `is_exterior: true`, `room_id`, `parent_wall_id` |
| Interior segments | Hough supplement → measure | `walls[]` with `is_exterior: false` |
| Room cells | Connected components | `rooms[]` |
| Measurements | `pixel_length / px_per_unit` | `length`, `length_raw` |
| Total area | `contourArea / px_per_unit²` | `total_area` |
| Facings | Normal probes + adjacency | `facing`: North / South / East / West |

### Key functions

| Function | Role |
|----------|------|
| `measure_walls(segments, px_per_unit, ...)` | Pixel coords → real lengths + facings |
| `assign_segment_facings(...)` | Cardinal direction per segment |
| `cleanup_wall_list(...)` | Six dedup passes before emit |
| `_shift_px_coords(...)` | Remap to full image if ROI was used |

### Exterior sub-segment IDs

Room-aware exterior walls use compound IDs:

```
w3.s1  →  parent_wall_id: "w3", segment_index: 1, room_id: "R14"
```

### Web coordinate transform

`coords.js` → `cvResultToAnalysis()` adds percentage-based overlay coords:

- `x1_pct…y2_pct`, `centroid_pct`, `footprint_polygon_pct`

### Validation geometry metrics

`validation/arqen_validation/`:

- `wall_network_closure()` — endpoint connectivity
- `interior_coverage()` — room/wall spatial consistency

---

## Stage 8 — Structured JSON

**Purpose:** Emit a versioned, machine-readable building model for clients, validation, and downstream tools.

### Production output (`analyze_page()`)

```json
{
  "detected_scale": "1in=16ft",
  "total_area": "8702.1 ft²",
  "units": "imperial",
  "polygon_vertices": 80,
  "footprint_polygon_px": [[x, y], ...],
  "footprint_bbox_px": [x0, y0, x1, y1],
  "image_size_px": [width, height],
  "px_per_ft": 9.38,
  "rooms": [
    {
      "id": "R1",
      "area_px": 12345,
      "centroid_px": [cx, cy],
      "bbox_px": [x0, y0, x1, y1],
      "area": "400.0 ft²",
      "area_raw": 400.0
    }
  ],
  "walls": [
    {
      "id": "w3.s1",
      "name": "North Wall 3 part 1 → R14",
      "facing": "North",
      "length": "34.88 ft",
      "length_raw": 34.88,
      "angle_deg": 0.0,
      "px_coords": [x1, y1, x2, y2],
      "is_exterior": true,
      "room_id": "R14",
      "parent_wall_id": "w3",
      "segment_index": 1,
      "segment_count": 2
    }
  ],
  "mask_cache_path": "/tmp/arqen_mask_*.png",
  "mask_roi_offset": [ox, oy]
}
```

### HTTP additions (`cv_service.py`)

- `mask_base64` — inline PNG data URL of `wall_pair_mask` (temp file deleted after encode)

### Error payload

```json
{"error": "No building footprint found"}
```

### Web-enriched fields (post `cvResultToAnalysis` + Claude)

| Field | Source |
|-------|--------|
| `rooms[].label` | `assignRoomLabels()` |
| `walls[].room` | Derived from room label |
| `scale_confidence` | Claude scale detect |
| `dimension_lines[]` | Client overlay from wall segments |
| `footprint_polygon_pct` | Percentage coords for canvas |

### Emit paths

| Path | How |
|------|-----|
| CLI | `preprocess.py --output out.json` |
| HTTP | `POST /cv-analyze` response body |
| Web | `analysis.js` stores result in app state → `export.js` CSV/PNG |
| Validation | `validation/arqen_validation/runner.py` writes `prediction.json` |

### Sample file

`Arqen/out.json` — note older samples may predate room-split IDs (`w3.s1`).

---

## Stage 9 — HVAC Load Inputs

**Purpose:** Feed conditioned-space geometry into load calculation tools (Manual J, ASHRAE, etc.).

### Current status: **no HVAC module in codebase**

No Manual J, BTU, CFM, ASHRAE, U-value, or climate-zone logic exists. This stage is a **planned downstream consumer** of Stage 8 JSON.

### Available today (from CV + web enrichment)

| JSON field | HVAC relevance |
|------------|----------------|
| `rooms[].area_raw` | Conditioned floor area per zone |
| `rooms[].label` | Space type classification (Claude, not CV) |
| `total_area` | Building gross area |
| `walls[]` with `facing`, `length_raw`, `is_exterior` | Envelope perimeter by orientation |
| `footprint_polygon_px` | Building shape / exposure geometry |
| `px_per_ft` + `detected_scale` | Measurement calibration |

### Mapping example (external tool)

```
Per room:
  zone_name     ← rooms[].label  (or rooms[].id if unlabeled)
  floor_area    ← rooms[].area_raw

Per orientation (aggregate exterior walls):
  north_wall_ft ← sum(walls[].length_raw where facing=North and is_exterior)
  south_wall_ft ← ...
  east_wall_ft  ← ...
  west_wall_ft  ← ...

Building:
  total_area    ← total_area
  footprint     ← footprint_polygon_px
```

### Not available (requires future work)

| Input | Gap |
|-------|-----|
| Glazing area per wall | Windows not detected in CV |
| Door counts / sizes | Doors not modeled as geometry |
| Ceiling height | Not extracted |
| U-values, SHGC, infiltration | Not in pipeline |
| Occupancy / internal loads | Not in pipeline |
| Climate zone | External input |
| Duct routing | Not in pipeline |

### Recommended path to HVAC integration

1. **Phase 1 (now):** Export `rooms[]` + exterior `walls[]` by facing to CSV/JSON adapter.
2. **Phase 2:** Add door/window objects (Stage 4) with widths and host-wall association.
3. **Phase 3:** Room label → space-type lookup table (residential ASHRAE defaults).
4. **Phase 4:** Dedicated HVAC export schema or Manual J API integration.

---

## Full Stage Dependency Chain

```
PDF / PNG / base64
  → [optional] _crop_to_roi
  → parse_scale → px_per_unit
  → preprocess → (binary, wall_pair_mask)
  → find_footprint → contour → simplify_polygon
  → extract_wall_segments → _filter_wall_segments → snap_segments_to_walls
  → split_exterior_walls_by_room → (rooms, exterior_sub_walls)
  → _hough_supplement → measure_walls (interior)
  → cleanup_wall_list + length filters
  → [optional] coordinate remap if ROI
  → JSON emit + mask cache
  → [HTTP] mask_base64
  → [Web] cvResultToAnalysis → assignRoomLabels → export
  → [Future] HVAC load calculator
```

---

## Key File Index

| File | Pipeline stage(s) |
|------|-------------------|
| `Arqen/preprocess.py` | All CV stages; `analyze_page()` orchestrator |
| `Arqen/scale_parse.py` | Image Processing (calibration) |
| `Arqen/extract_wall_segments_class.py` | Line Detection (exterior) |
| `Arqen/room_wall_split.py` | Line Detection (interior), Room Segmentation |
| `Arqen/cv_service.py` | HTTP wrapper, JSON emit |
| `Arqen/debug_pipeline.py` | Debug PNGs for every stage |
| `Arqen/viewer.py` | PDF rasterization + HTML overlay |
| `claude_demo/arch-takeoff/js/upload.js` | PDF (browser rasterization) |
| `claude_demo/arch-takeoff/js/analysis.js` | Text OCR (LLM), web orchestration |
| `claude_demo/arch-takeoff/js/coords.js` | Geometry Construction (web coords) |
| `claude_demo/arch-takeoff/js/export.js` | Structured JSON export |
| `validation/arqen_validation/runner.py` | Batch pipeline runner |
| `validation/schema/ground_truth.schema.json` | Target JSON schema (6 categories) |

---

## Configuration Quick Reference

| Parameter | Where set | Default |
|-----------|-----------|---------|
| `scale` / `scale_str` | CLI, HTTP, manifest | **required** |
| `dpi` | CLI 300, HTTP 150, web 144 | varies |
| `roi` | Web UI, HTTP | optional |
| `doorway_close_ft` | HTTP, `analyze_page()` | 2.5 |
| `room_debug_dir` | `validate_room_split.py` | None |
| `ARQEN_DEBUG_DUMP=1` | env (`cv_service.py`) | off |
| `SERVICE_SECRET` | env | optional HTTP auth |

---

## Maturity by Stage

| Stage | Maturity | Notes |
|-------|----------|-------|
| PDF | Production | PyMuPDF + browser PDF.js |
| Image Processing | Advanced prototype | Scale-adaptive, heavily tuned |
| Line Detection | Advanced prototype | Orthogonal, double-stroke only |
| Symbol Detection | Not started (CV) | Morphology bridging only |
| Text OCR | LLM-only | No deterministic OCR |
| Room Segmentation | Recently integrated | Manually validated |
| Geometry Construction | Advanced prototype | No junction graph |
| Structured JSON | Production | CV schema stable; web adds fields |
| HVAC Load Inputs | Not started | Consumer mapping documented above |

---

## Related Documents

- [`Arqen/ARCHITECTURE.md`](../Arqen/ARCHITECTURE.md) — deployment diagram, failure points, roadmap
- [`docs/project_context.md`](project_context.md) — stack, goals, success metrics
- [`validation/README.md`](../validation/README.md) — accuracy scoring framework
