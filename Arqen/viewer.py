#!/usr/bin/env python3
"""
viewer.py — Generate a self-contained interactive HTML wall viewer.

The output is a single .html file (no server needed) that shows:
  • Left panel  — architectural plan image
  • Right panel — wall list with direction icons
  • Hover a wall → highlighted on the plan with a color-coded glow

Usage:
  python viewer.py --json result.json                      # uses visualization path from JSON
  python viewer.py --json result.json --pdf plan.pdf       # rasterizes PDF for clean image
  python viewer.py --json result.json --pdf plan.pdf --open
"""

import argparse
import base64
import json
import os
import sys
import webbrowser
from pathlib import Path

import fitz


def _rasterize_page(pdf_path: str, page: int = 1, dpi: int = 150) -> bytes:
    doc = fitz.open(pdf_path)
    idx = page - 1
    if idx < 0 or idx >= len(doc):
        print(f"Error: page {page} out of range (PDF has {len(doc)} pages)", file=sys.stderr)
        sys.exit(1)
    pix = doc[idx].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    doc.close()
    return pix.tobytes("png")


def _file_to_b64(path: str) -> tuple[str, str]:
    suffix = Path(path).suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    with open(path, "rb") as f:
        return mime, base64.b64encode(f.read()).decode()


# ── HTML template ────────────────────────────────────────────────────────────

def _generate_html(data: dict, img_b64: str, img_mime: str) -> str:
    walls       = data["walls"]
    img_w, img_h = data["image_size_px"]
    total_area  = data.get("total_area", "")
    scale_str   = data.get("detected_scale", "")
    walls_json  = json.dumps(walls)
    n_walls     = len(walls)

    meta_parts = []
    if scale_str:
        meta_parts.append(f"Scale: {scale_str}")
    if total_area:
        meta_parts.append(f"Total area: {total_area}")
    meta_html = "  &nbsp;·&nbsp;  ".join(meta_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Arqen — Wall Viewer</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: #0e0e0e;
  color: #d8d8d8;
  display: flex;
  height: 100vh;
  overflow: hidden;
}}

/* ── Left: plan ─────────────────────────────────── */
#plan-panel {{
  flex: 1;
  min-width: 0;
  position: relative;
  background: #090909;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}}

#plan-container {{
  position: relative;
  line-height: 0;
  max-width: 100%;
  max-height: 100vh;
}}

#plan-img {{
  display: block;
  max-width: 100%;
  max-height: 100vh;
  object-fit: contain;
  user-select: none;
}}

#overlay-canvas {{
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}}

/* ── Right: wall list ───────────────────────────── */
#wall-panel {{
  width: 296px;
  min-width: 296px;
  background: #141414;
  border-left: 1px solid #1f1f1f;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

#panel-header {{
  padding: 16px 18px 12px;
  border-bottom: 1px solid #1f1f1f;
  flex-shrink: 0;
}}

#panel-title {{
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #555;
  display: flex;
  align-items: center;
  gap: 8px;
}}

#panel-title span.count {{
  background: #222;
  color: #888;
  font-size: 10px;
  font-weight: 500;
  padding: 1px 6px;
  border-radius: 10px;
  letter-spacing: 0;
}}

#panel-meta {{
  margin-top: 6px;
  font-size: 11px;
  color: #3d3d3d;
  line-height: 1.6;
}}

#wall-list {{
  overflow-y: auto;
  flex: 1;
  padding: 4px 0 8px;
}}

#wall-list::-webkit-scrollbar {{ width: 3px; }}
#wall-list::-webkit-scrollbar-track {{ background: transparent; }}
#wall-list::-webkit-scrollbar-thumb {{ background: #2a2a2a; border-radius: 2px; }}

/* ── Wall item ──────────────────────────────────── */
.wall-item {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 18px;
  cursor: default;
  border-left: 2px solid transparent;
  transition: background 60ms, border-color 60ms;
}}

.wall-item:hover,
.wall-item.active {{
  background: #1c1c1c;
}}

.wall-item.active  {{ border-left-color: var(--accent); }}
.wall-item:hover   {{ border-left-color: #333; }}

.wall-icon {{
  flex-shrink: 0;
  width: 30px;
  height: 30px;
  border-radius: 7px;
  background: #1e1e1e;
  border: 1px solid #272727;
  display: flex;
  align-items: center;
  justify-content: center;
}}

.wall-icon svg {{
  width: 14px;
  height: 14px;
}}

.wall-info {{
  flex: 1;
  min-width: 0;
}}

.wall-name {{
  font-size: 12px;
  font-weight: 500;
  color: #c8c8c8;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.wall-item.active .wall-name {{
  color: #e8e8e8;
}}

.wall-length {{
  font-size: 11px;
  color: #555;
  margin-top: 1px;
  font-variant-numeric: tabular-nums;
}}

.wall-id {{
  font-size: 10px;
  color: #333;
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
  letter-spacing: 0.03em;
}}

/* ── Tooltip label that follows the highlighted wall ── */
#wall-label {{
  position: fixed;
  background: #111;
  border: 1px solid #333;
  color: #ccc;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 4px;
  pointer-events: none;
  white-space: nowrap;
  display: none;
  z-index: 10;
}}
</style>
</head>
<body>

<div id="plan-panel">
  <div id="plan-container">
    <img id="plan-img"
         src="data:{img_mime};base64,{img_b64}"
         alt="Architectural Plan">
    <canvas id="overlay-canvas"></canvas>
  </div>
</div>

<div id="wall-panel">
  <div id="panel-header">
    <div id="panel-title">
      Walls <span class="count">{n_walls}</span>
    </div>
    <div id="panel-meta">{meta_html}</div>
  </div>
  <div id="wall-list"></div>
</div>

<div id="wall-label"></div>

<script>
const WALLS       = {walls_json};
const IMG_COORD_W = {img_w};
const IMG_COORD_H = {img_h};

const img       = document.getElementById('plan-img');
const canvas    = document.getElementById('overlay-canvas');
const ctx       = canvas.getContext('2d');
const wallList  = document.getElementById('wall-list');
const label     = document.getElementById('wall-label');

// ── Color palette (per facing direction) ──────────────────────────────────
const COLORS = {{
  North: '#4a9eff',
  East:  '#52c98e',
  South: '#ff6b6b',
  West:  '#f0b429',
}};

// Arrow rotation: arrow points in the facing direction
const FACING_DEG = {{ North: 0, East: 90, South: 180, West: 270 }};

function arrowSVG(facing) {{
  const deg = FACING_DEG[facing] ?? 0;
  const c = COLORS[facing] ?? '#888';
  return `<svg viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg"
    style="transform:rotate(${{deg}}deg);transition:transform 0.15s">
    <path d="M7 11 L7 3 M4 6 L7 3 L10 6"
      stroke="${{c}}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
}}

// ── Build wall list ────────────────────────────────────────────────────────
WALLS.forEach((wall, i) => {{
  const item = document.createElement('div');
  item.className = 'wall-item';
  item.style.setProperty('--accent', COLORS[wall.facing] ?? '#4a9eff');
  item.innerHTML = `
    <div class="wall-icon">${{arrowSVG(wall.facing)}}</div>
    <div class="wall-info">
      <div class="wall-name">${{wall.name}}</div>
      <div class="wall-length">${{wall.length}}</div>
    </div>
    <div class="wall-id">${{wall.id}}</div>`;

  item.addEventListener('mouseenter', (e) => highlight(i, item));
  item.addEventListener('mouseleave', ()  => clear(item));
  wallList.appendChild(item);
}});

// ── Canvas sync ────────────────────────────────────────────────────────────
function syncCanvas() {{
  // canvas pixel buffer = image rendered size (devicePixelRatio-aware)
  const dpr  = window.devicePixelRatio || 1;
  const rect = img.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width  = rect.width  + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}

function scale() {{
  const rect = img.getBoundingClientRect();
  return {{ sx: rect.width / IMG_COORD_W, sy: rect.height / IMG_COORD_H }};
}}

// ── Highlight one wall ─────────────────────────────────────────────────────
function highlight(idx, item) {{
  syncCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const wall = WALLS[idx];
  const [x1, y1, x2, y2] = wall.px_coords;
  const {{ sx, sy }} = scale();
  const c = COLORS[wall.facing] ?? '#fff';

  const px1 = x1 * sx, py1 = y1 * sy;
  const px2 = x2 * sx, py2 = y2 * sy;

  // Dim overlay
  ctx.fillStyle = 'rgba(0,0,0,0.40)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Glow halo
  ctx.beginPath();
  ctx.moveTo(px1, py1);
  ctx.lineTo(px2, py2);
  ctx.strokeStyle = c + '40';
  ctx.lineWidth   = 14;
  ctx.lineCap     = 'round';
  ctx.stroke();

  // Main line
  ctx.strokeStyle = c;
  ctx.lineWidth   = 3;
  ctx.stroke();

  // Endpoint dots
  [[ px1, py1 ], [ px2, py2 ]].forEach(([x, y]) => {{
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = c;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }});

  // Active state on list item
  document.querySelectorAll('.wall-item').forEach(el => el.classList.remove('active'));
  item.classList.add('active');
}}

function clear(item) {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  item.classList.remove('active');
}}

// ── Init ───────────────────────────────────────────────────────────────────
if (img.complete) syncCanvas();
img.addEventListener('load', syncCanvas);
window.addEventListener('resize', () => {{
  syncCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.querySelectorAll('.wall-item').forEach(el => el.classList.remove('active'));
}});
</script>

</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML wall viewer from preprocess.py output"
    )
    parser.add_argument("--json",   required=True,
                        help="Path to preprocess.py JSON output file")
    parser.add_argument("--pdf",    default=None,
                        help="PDF path — rasterizes a clean (unannotated) image (optional)")
    parser.add_argument("--page",   type=int, default=1,
                        help="PDF page to rasterize (default: 1)")
    parser.add_argument("--dpi",    type=int, default=150,
                        help="DPI for PDF rasterization (default: 150)")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: <json>.html)")
    parser.add_argument("--open",   action="store_true",
                        help="Auto-open in default browser after generating")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "walls" not in data:
        print("Error: JSON does not contain wall data. Run preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    # ── Get image ─────────────────────────────────────────────────────────────
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: {pdf_path} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Rasterizing page {args.page} at {args.dpi} DPI …", file=sys.stderr)
        png_bytes = _rasterize_page(str(pdf_path), page=args.page, dpi=args.dpi)
        img_mime  = "image/png"
        img_b64   = base64.b64encode(png_bytes).decode()
        print(f"Image encoded ({len(png_bytes) // 1024} KB)", file=sys.stderr)

    elif "visualization" in data and Path(data["visualization"]).exists():
        vis_path = data["visualization"]
        print(f"Using visualization image: {vis_path}", file=sys.stderr)
        img_mime, img_b64 = _file_to_b64(vis_path)
        print(f"Image encoded ({len(img_b64) * 3 // 4 // 1024} KB)", file=sys.stderr)

    else:
        print(
            "Error: no image found.\n"
            "  • Run preprocess.py with --visualize to save an annotated image, or\n"
            "  • Pass --pdf <plan.pdf> to rasterize a clean image.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Generate HTML ─────────────────────────────────────────────────────────
    html = _generate_html(data, img_b64, img_mime)

    out_path = Path(args.output) if args.output else json_path.with_suffix(".html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Viewer saved → {out_path}", file=sys.stderr)

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
