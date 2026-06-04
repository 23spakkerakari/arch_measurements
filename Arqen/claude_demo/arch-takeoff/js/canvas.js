/**
 * canvas.js
 * Draws dimension lines and labels onto the result canvas.
 */

/**
 * Fetch the wall_pair_mask PNG from the server and store it in appState.maskImage.
 * Called lazily from drawCanvas whenever the mask_cache_path changes.
 * On successful load, triggers a fresh drawCanvas to composite the mask.
 */
function loadMaskImageIfNeeded(data) {
  const maskPath = data && data.mask_cache_path;
  if (!maskPath) {
    appState.maskImage = null;
    appState._loadedMaskPath = null;
    return;
  }
  if (appState._loadedMaskPath === maskPath) return;  // already loaded / loading
  appState._loadedMaskPath = maskPath;
  appState.maskImage = null;  // clear stale image while new one loads
  const img = new Image();
  img.onload = () => {
    appState.maskImage = img;
    drawCanvas();
  };
  img.onerror = () => {
    console.warn('[mask] failed to load mask image from', maskPath);
    appState.maskImage = null;
  };
  img.src = `/api/mask-image?path=${encodeURIComponent(maskPath)}`;
}

function drawCanvas() {
  const data = appState.analysisResult;
  if (!data) return;

  // Kick off (or serve from cache) the wall_pair_mask fetch.
  loadMaskImageIfNeeded(data);

  const img    = document.getElementById('result-img');
  const canvas = document.getElementById('overlay-canvas');

  const render = () => {
    const container = img.parentElement;
    const imgRect   = img.getBoundingClientRect();
    const boxRect   = container.getBoundingClientRect();

    // Pin overlay to the displayed image, not the full scrollable container.
    canvas.style.left   = `${imgRect.left - boxRect.left + container.scrollLeft}px`;
    canvas.style.top    = `${imgRect.top - boxRect.top + container.scrollTop}px`;
    canvas.style.width  = `${imgRect.width}px`;
    canvas.style.height = `${imgRect.height}px`;

    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(imgRect.width  * dpr);
    canvas.height = Math.round(imgRect.height * dpr);

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, imgRect.width, imgRect.height);

    const W = imgRect.width;
    const H = imgRect.height;

    // ── CV wall-pair mask overlay ────────────────────────
    // Renders the raw OpenCV wall_pair_mask as a semi-transparent green tint.
    // White pixels in the mask = pixels OpenCV identified as double-line wall pairs.
    // Gaps in the tint explain exactly why a wall was missed.
    if (appState.layers.mask && appState.maskImage) {
      const maskImg = appState.maskImage;
      const [imgW, imgH] = data.image_size_px || [maskImg.naturalWidth, maskImg.naturalHeight];
      const [ox, oy] = data.mask_roi_offset || [0, 0];
      const dx = (ox / imgW) * W;
      const dy = (oy / imgH) * H;
      const dw = (maskImg.naturalWidth / imgW) * W;
      const dh = (maskImg.naturalHeight / imgH) * H;

      // Build a green-tinted version of the mask on an offscreen canvas:
      // fill with solid green, then clip to the mask's white pixels.
      const off = document.createElement('canvas');
      off.width  = Math.max(1, Math.round(dw));
      off.height = Math.max(1, Math.round(dh));
      const octx = off.getContext('2d');
      octx.fillStyle = '#00e878';
      octx.fillRect(0, 0, off.width, off.height);
      octx.globalCompositeOperation = 'destination-in';
      octx.drawImage(maskImg, 0, 0, off.width, off.height);

      ctx.save();
      ctx.globalAlpha = 0.40;
      ctx.drawImage(off, dx, dy, dw, dh);
      ctx.restore();
    }

    const poly = data.footprint_polygon_pct;
    const fp = data.footprint_bbox_cv || data.footprint_bbox;
    if (appState.layers.dims) {
      ctx.strokeStyle = data._userRoi ? 'rgba(0, 200, 120, 0.9)' : 'rgba(255, 60, 60, 0.75)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      if (poly && poly.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(poly[0][0] * W, poly[0][1] * H);
        for (let i = 1; i < poly.length; i++) {
          ctx.lineTo(poly[i][0] * W, poly[i][1] * H);
        }
        ctx.closePath();
        ctx.stroke();
      } else if (fp && fp.x0_pct != null) {
        ctx.strokeRect(
          fp.x0_pct * W, fp.y0_pct * H,
          (fp.x1_pct - fp.x0_pct) * W, (fp.y1_pct - fp.y0_pct) * H
        );
      }
      ctx.setLineDash([]);
    }

    const dimLines = data.dimension_lines || [];
    const walls    = data.walls || [];
    const hi = appState.highlightedWall;

    // Resolve the highlighted wall object from the walls array.
    const hiWall = (hi !== null && hi !== undefined) ? walls[hi] : null;

    // Look up the matching dim line by wall ID so the index into the filtered
    // dimension_lines subset never diverges from the wall list index.
    const hiDimLine = hiWall && hiWall.id
      ? dimLines.find(dl => dl.wallId === hiWall.id) || null
      : (hi !== null && hi !== undefined ? dimLines[hi] : null);

    const hiWallHasCoords = hiWall &&
      hiWall.x1_pct != null && hiWall.y1_pct != null &&
      hiWall.x2_pct != null && hiWall.y2_pct != null;
    // Synthesise a dim-line-shaped object from wall coords when needed
    const hiTarget  = hiDimLine || (hiWallHasCoords ? {
      x1_pct: hiWall.x1_pct, y1_pct: hiWall.y1_pct,
      x2_pct: hiWall.x2_pct, y2_pct: hiWall.y2_pct,
      label:  hiWall.length || null,
    } : null);

    if (appState.layers.dims) {
      if (hiTarget) {
        // Dim overlay
        ctx.fillStyle = 'rgba(0,0,0,0.45)';
        ctx.fillRect(0, 0, W, H);

        // Draw all lines at reduced opacity
        ctx.save();
        ctx.globalAlpha = 0.25;
        drawDimLines(ctx, dimLines, W, H);
        ctx.restore();

        // Draw the selected line brightly with glow
        drawHighlightedDimLine(ctx, hiTarget, hi, W, H);
      } else {
        drawDimLines(ctx, dimLines, W, H);
      }
    }

    // ── DRAW WALL rubber-band preview ────────────────────
    // Show a dashed line from the first click to the current cursor position,
    // plus a dot at the first anchor point.
    if (appState.drawWallMode && appState.drawWallFirstPoint) {
      const p1 = appState.drawWallFirstPoint;
      const cur = appState._drawCursor || p1;
      const ax = p1.x * W,  ay = p1.y * H;
      const bx = cur.x * W, by = cur.y * H;

      ctx.save();
      ctx.strokeStyle = 'rgba(0, 230, 120, 0.9)';
      ctx.lineWidth   = 1.8;
      ctx.setLineDash([8, 5]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      ctx.setLineDash([]);

      // Anchor dot at first click
      ctx.fillStyle = '#00e878';
      ctx.beginPath();
      ctx.arc(ax, ay, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.7)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Instruction label above anchor
      ctx.font         = 'bold 10px "Space Mono", monospace';
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle    = 'rgba(8,12,18,0.85)';
      ctx.fillRect(ax + 8, ay - 20, 114, 16);
      ctx.fillStyle = '#00e878';
      ctx.fillText('Click 2nd point to finish', ax + 10, ay - 6);
      ctx.restore();
    }
  };

  if (img.complete && img.naturalWidth) {
    render();
  } else {
    img.onload = render;
  }

  window.removeEventListener('resize', render);
  window.addEventListener('resize', render);
}

// ── Highlighted single dim line ──────────────────────────
function drawHighlightedDimLine(ctx, dl, idx, W, H) {
  const x1 = dl.x1_pct * W;
  const y1 = dl.y1_pct * H;
  const x2 = dl.x2_pct * W;
  const y2 = dl.y2_pct * H;
  const color = WALL_STROKES[idx % WALL_STROKES.length];

  // Glow halo
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color + '55';
  ctx.lineWidth   = 12;
  ctx.lineCap     = 'round';
  ctx.setLineDash([]);
  ctx.stroke();

  // Main line
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2.5;
  ctx.stroke();

  drawArrowheadColored(ctx, x1, y1, x2, y2, color);
  drawArrowheadColored(ctx, x2, y2, x1, y1, color);
  drawWitnessLinesColored(ctx, x1, y1, x2, y2, color);

  // Endpoint dots
  [[x1, y1], [x2, y2]].forEach(([px, py]) => {
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  if (appState.layers.labels && dl.label) {
    drawDimLabelColored(ctx, dl.label, x1, y1, x2, y2, color);
  }
}

function drawArrowheadColored(ctx, tipX, tipY, fromX, fromY, color) {
  const angle  = Math.atan2(tipY - fromY, tipX - fromX);
  const spread = 0.4;
  const len    = 10;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - Math.cos(angle - spread) * len, tipY - Math.sin(angle - spread) * len);
  ctx.lineTo(tipX - Math.cos(angle + spread) * len, tipY - Math.sin(angle + spread) * len);
  ctx.closePath();
  ctx.fill();
}

function drawWitnessLinesColored(ctx, x1, y1, x2, y2, color) {
  const dx    = x2 - x1;
  const dy    = y2 - y1;
  const len   = Math.sqrt(dx * dx + dy * dy) || 1;
  const perpX = (-dy / len) * 8;
  const perpY = ( dx / len) * 8;

  ctx.strokeStyle = color + '66';
  ctx.lineWidth   = 0.8;
  ctx.setLineDash([4, 3]);

  [[x1, y1], [x2, y2]].forEach(([px, py]) => {
    ctx.beginPath();
    ctx.moveTo(px - perpX, py - perpY);
    ctx.lineTo(px + perpX, py + perpY);
    ctx.stroke();
  });

  ctx.setLineDash([]);
}

function drawDimLabelColored(ctx, label, x1, y1, x2, y2, color) {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;

  const perpX = (-dy / len) * 14;
  const perpY = ( dx / len) * 14;

  ctx.font         = 'bold 10px "Space Mono", monospace';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';

  const lx = mx + perpX;
  const ly = my + perpY;
  const m  = ctx.measureText(label);

  ctx.fillStyle   = 'rgba(8,12,18,0.9)';
  ctx.strokeStyle = color + '88';
  ctx.lineWidth   = 1;
  ctx.fillRect(lx - m.width / 2 - 4, ly - 8, m.width + 8, 16);
  ctx.strokeRect(lx - m.width / 2 - 4, ly - 8, m.width + 8, 16);

  ctx.fillStyle = color;
  ctx.fillText(label, lx, ly);
}

// ── Dimension lines ─────────────────────────────────────
function drawDimLines(ctx, dimLines, W, H) {
  dimLines.forEach(dl => {
    const x1 = dl.x1_pct * W;
    const y1 = dl.y1_pct * H;
    const x2 = dl.x2_pct * W;
    const y2 = dl.y2_pct * H;

    ctx.strokeStyle = '#f0a500';
    ctx.lineWidth   = 1.2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    drawArrowhead(ctx, x1, y1, x2, y2);
    drawArrowhead(ctx, x2, y2, x1, y1);
    drawWitnessLines(ctx, x1, y1, x2, y2);

    if (appState.layers.labels && dl.label) {
      drawDimLabel(ctx, dl.label, x1, y1, x2, y2);
    }
  });
}

function drawArrowhead(ctx, tipX, tipY, fromX, fromY) {
  const angle  = Math.atan2(tipY - fromY, tipX - fromX);
  const spread = 0.4;
  const len    = 10;
  ctx.fillStyle = '#f0a500';
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - Math.cos(angle - spread) * len, tipY - Math.sin(angle - spread) * len);
  ctx.lineTo(tipX - Math.cos(angle + spread) * len, tipY - Math.sin(angle + spread) * len);
  ctx.closePath();
  ctx.fill();
}

function drawWitnessLines(ctx, x1, y1, x2, y2) {
  const dx    = x2 - x1;
  const dy    = y2 - y1;
  const len   = Math.sqrt(dx * dx + dy * dy) || 1;
  const perpX = (-dy / len) * 8;
  const perpY = ( dx / len) * 8;

  ctx.strokeStyle = 'rgba(240,165,0,0.4)';
  ctx.lineWidth   = 0.8;
  ctx.setLineDash([4, 3]);

  [[x1, y1], [x2, y2]].forEach(([px, py]) => {
    ctx.beginPath();
    ctx.moveTo(px - perpX, py - perpY);
    ctx.lineTo(px + perpX, py + perpY);
    ctx.stroke();
  });

  ctx.setLineDash([]);
}

/**
 * Toggle the DRAW WALL two-click mode.
 * First click sets the start point; second click creates the wall.
 */
function toggleDrawWallMode(btn) {
  appState.drawWallMode = !appState.drawWallMode;
  appState.drawWallFirstPoint = null;
  appState._drawCursor = null;
  btn.classList.toggle('active', appState.drawWallMode);
  const canvas = document.getElementById('overlay-canvas');
  canvas.style.cursor = appState.drawWallMode ? 'crosshair' : '';
  canvas.style.pointerEvents = appState.drawWallMode ? 'auto' : '';
  if (!appState.drawWallMode) drawCanvas();  // clear rubber-band preview
}

function _initAddWallClickHandler() {
  const canvas = document.getElementById('overlay-canvas');
  if (canvas._addWallHandlerAttached) return;
  canvas._addWallHandlerAttached = true;

  // ── Mousemove: update rubber-band cursor for draw mode ──
  canvas.addEventListener('mousemove', (e) => {
    if (!appState.drawWallMode || !appState.drawWallFirstPoint) return;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W === 0 || H === 0) return;
    appState._drawCursor = {
      x: (e.clientX - rect.left) / W,
      y: (e.clientY - rect.top)  / H,
    };
    drawCanvas();
  });

  canvas.addEventListener('click', (e) => {
    if (!appState.drawWallMode) return;

    const rect = canvas.getBoundingClientRect();
    const xCanvas = e.clientX - rect.left;
    const yCanvas = e.clientY - rect.top;
    const W = rect.width;
    const H = rect.height;

    if (W === 0 || H === 0) return;

    const xPct = xCanvas / W;
    const yPct = yCanvas / H;

    if (!appState.drawWallFirstPoint) {
      appState.drawWallFirstPoint = { x: xPct, y: yPct };
      appState._drawCursor = { x: xPct, y: yPct };
      drawCanvas();
    } else {
      const p1 = appState.drawWallFirstPoint;
      appState.drawWallFirstPoint = null;
      appState._drawCursor = null;
      createManualWall(p1.x, p1.y, xPct, yPct);
    }
  });
}

// Initialise the handler as soon as the DOM is ready.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _initAddWallClickHandler);
} else {
  _initAddWallClickHandler();
}

function drawDimLabel(ctx, label, x1, y1, x2, y2) {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;

  const perpX = (-dy / len) * 14;
  const perpY = ( dx / len) * 14;

  ctx.font         = 'bold 10px "Space Mono", monospace';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';

  const lx = mx + perpX;
  const ly = my + perpY;
  const m  = ctx.measureText(label);

  ctx.fillStyle   = 'rgba(8,12,18,0.85)';
  ctx.strokeStyle = 'rgba(240,165,0,0.5)';
  ctx.lineWidth   = 0.7;
  ctx.fillRect(lx - m.width / 2 - 4, ly - 8, m.width + 8, 16);
  ctx.strokeRect(lx - m.width / 2 - 4, ly - 8, m.width + 8, 16);

  ctx.fillStyle = '#f0a500';
  ctx.fillText(label, lx, ly);
}
