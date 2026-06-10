# Arqen Labs — Project Context

Living reference for stack, goals, known issues, and success metrics. See also:

- `docs/PIPELINE.md` — end-to-end stage map (PDF → HVAC load inputs)
- `Arqen/ARCHITECTURE.md` — pipeline details; §8 covers dependencies, env vars, DPI conventions
- `validation/README.md` — accuracy scoring framework (rooms, walls, doors, windows, labels, dimensions, boundary closure)
- `docs/baseline_metrics.md` — captured baseline numbers for the current implementation
- `docs/improvement_proposals.md` — ranked improvement backlog
- `docs/floor_plan_extraction_analysis.md` — detected vs required objects, HVAC gaps, accuracy bottlenecks, prioritized roadmap

## Current Stack

- PDF input only (primary); PNG/JPG supported in web UI
- OpenCV for geometry extraction (`preprocess.py`, `cv_service.py`)
- Anthropic Claude for OCR, scale detection, room labels, and vision fallback
- JSON structured output (`out.json` schema)

## Goal

Convert floor plans into engineering-grade building geometry.

## Known Issues

- Missed rooms (threshold filters, doorway bridging, footprint splits at openings)
- OCR inconsistencies (room labels and scale via LLM only; non-deterministic)
- Wall segmentation failures (wall-pair filter, polygon simplification, Hough supplement, dedup)
- Window identification accuracy (not detected in CV path; LLM-only today)
- Door detection not modeled as geometry (`doorway_close_ft` morphology only)

## Success Metrics

- Room accuracy >95%
- Wall accuracy >95%
- Window accuracy >90%
- Door accuracy >90%
