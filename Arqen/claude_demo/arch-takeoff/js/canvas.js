/**
 * canvas.js
 * Draws dimension lines and labels onto the result canvas.
 */

/**
 * Load the wall_pair_mask into appState.maskImage for the debug overlay.
 *
 * Two sources are supported:
 *   mask_base64     — inline data-URL returned by the Render production service
 *                     (no extra HTTP request needed)
 *   mask_cache_path — temp-file path served by the local Express proxy
 *                     (local development only)
 *
 * Called lazily from drawCanvas whenever the analysis result changes.
 * On successful load, triggers a fresh drawCanvas to composite the mask.
 */
function loadMaskImageIfNeeded(data) {
  // Prefer the inline data-URL (production) over the file-path endpoint (local dev).
  const maskSrc = (data && data.mask_base64)
    || (data && data.mask_cache_path
        ? `/api/mask-image?path=${encodeURIComponent(data.mask_cache_path)}`
        : null);

  if (!maskSrc) {
    appState.maskImage = null;
    appState._loadedMaskPath = null;
    return;
  }
  if (appState._loadedMaskPath === maskSrc) return;  // already loaded / loading
  appState._loadedMaskPath = maskSrc;
  appState.maskImage = null;  // clear stale image while new one loads
  const img = new Image();
  img.onload = () => {
    appState.maskImage = img;
    drawCanvas();
  };
  img.onerror = () => {
    console.warn('[mask] failed to load mask image');
    appState.maskImage = null;
  };
  img.src = maskSrc;
}

function drawCanvas() {
  const data = appState.analysisResult;
  if (!data) return;

  // Keep pointer-events in sync with whether draggable endpoints are visible.
  _syncCanvasPointerEvents();

  // Kick off (or serve from cache) the wall_pair_mask fetch.
  loadMaskImageIfNeeded(data);

  const img    = document.getElementById('result-img');
  const canvas = document.getElementById('overlay-canvas');

  const render = () => {
    // Layout size inside plan-stage (pre-transform); stage CSS transform handles zoom/pan.
    const stage = document.getElementById('result-plan-stage');
    const W = stage ? stage.offsetWidth : img.offsetWidth;
    const H = stage ? stage.offsetHeight : img.offsetHeight;
    if (!W || !H) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    // Bitmap is HiDPI; CSS size must match the image/stage or overlays drift on retina displays.
    canvas.style.width  = `${W}px`;
    canvas.style.height = `${H}px`;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

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
      // User hint (green) — drawn before analysis.
      const hint = data._userRoiHint || appState.buildingRoi;
      if (hint && hint.x0_pct != null) {
        ctx.strokeStyle = 'rgba(0, 200, 120, 0.95)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([8, 5]);
        ctx.strokeRect(
          hint.x0_pct * W, hint.y0_pct * H,
          (hint.x1_pct - hint.x0_pct) * W, (hint.y1_pct - hint.y0_pct) * H
        );
        ctx.setLineDash([]);
      }

      // Auto-expanded analysis envelope (subtle cyan).
      const expanded = data.analysis_roi_pct;
      if (expanded && expanded.x0_pct != null) {
        ctx.strokeStyle = 'rgba(80, 200, 255, 0.65)';
        ctx.lineWidth = 1.25;
        ctx.setLineDash([4, 6]);
        ctx.strokeRect(
          expanded.x0_pct * W, expanded.y0_pct * H,
          (expanded.x1_pct - expanded.x0_pct) * W,
          (expanded.y1_pct - expanded.y0_pct) * H
        );
        ctx.setLineDash([]);
      }

      // Detected footprint polygon (red) when no user ROI workflow.
      if (!data._userRoi) {
        ctx.strokeStyle = 'rgba(255, 60, 60, 0.75)';
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
    }

    const dimLines = data.dimension_lines || [];
    const walls    = data.walls || [];

    // ── All detected walls (visible even without a room) ─
    walls.forEach((wall, i) => {
      if (wall.x1_pct == null) return;
      if (findRoomForWall(wall.id)) return;
      const x1 = wall.x1_pct * W, y1 = wall.y1_pct * H;
      const x2 = wall.x2_pct * W, y2 = wall.y2_pct * H;
      const isInterior = wall.is_exterior === false;
      ctx.save();
      ctx.strokeStyle = isInterior ? '#4a9eff' : '#ff8c42';
      ctx.lineWidth   = isInterior ? 4 : 5;
      ctx.lineCap     = 'round';
      ctx.globalAlpha = isInterior ? 0.42 : 0.55;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.restore();
    });

    // ── Door and window openings (CV-detected) ─────────────
    const doors = data.doors || [];
    const windows = data.windows || [];
    doors.forEach(d => {
      if (d.x0_pct == null) return;
      const x0 = d.x0_pct * W, y0 = d.y0_pct * H;
      const x1 = d.x1_pct * W, y1 = d.y1_pct * H;
      ctx.save();
      ctx.fillStyle = 'rgba(255, 140, 66, 0.25)';
      ctx.strokeStyle = '#ff8c42';
      ctx.lineWidth = 2;
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      if (d.id && d.center_pct) {
        ctx.fillStyle = '#ff8c42';
        ctx.font = 'bold 11px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(d.id, d.center_pct[0] * W, d.center_pct[1] * H);
      }
      ctx.restore();
    });
    windows.forEach(w => {
      if (w.x0_pct == null) return;
      const x0 = w.x0_pct * W, y0 = w.y0_pct * H;
      const x1 = w.x1_pct * W, y1 = w.y1_pct * H;
      ctx.save();
      ctx.fillStyle = 'rgba(74, 158, 255, 0.25)';
      ctx.strokeStyle = '#4a9eff';
      ctx.lineWidth = 2;
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      if (w.id && w.center_pct) {
        ctx.fillStyle = '#4a9eff';
        ctx.font = 'bold 11px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(w.id, w.center_pct[0] * W, w.center_pct[1] * H);
      }
      ctx.restore();
    });

    // ── Room wall highlights ─────────────────────────────
    // Draw colored strokes on exterior sub-segments (and other assigned walls).
    appState.rooms.forEach(room => {
      const isActive = room.id === appState.activeRoomId;
      room.wallIds.forEach(wallId => {
        const wall = walls.find(w => w.id === wallId);
        if (!wall || wall.x1_pct == null) return;
        const x1 = wall.x1_pct * W, y1 = wall.y1_pct * H;
        const x2 = wall.x2_pct * W, y2 = wall.y2_pct * H;
        const isExterior = wall.is_exterior !== false;
        ctx.save();
        ctx.strokeStyle = room.color;
        ctx.lineWidth   = isActive ? 10 : (isExterior ? 7 : 5);
        ctx.lineCap     = 'round';
        ctx.globalAlpha = isActive ? 0.45 : (isExterior ? 0.32 : 0.18);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.restore();
      });
    });

    // ── Canvas-focused wall highlight ────────────────────
    if (appState.canvasFocusedWallId) {
      const focusedWall = walls.find(w => w.id === appState.canvasFocusedWallId);
      if (focusedWall && focusedWall.x1_pct != null) {
        // Dim the rest of the plan so the selected wall stands out
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.35)';
        ctx.fillRect(0, 0, W, H);
        ctx.restore();

        const x1 = focusedWall.x1_pct * W, y1 = focusedWall.y1_pct * H;
        const x2 = focusedWall.x2_pct * W, y2 = focusedWall.y2_pct * H;
        const FOCUS_COLOR = '#ffb432';

        ctx.save();
        // Glow halo
        ctx.strokeStyle = FOCUS_COLOR + '66';
        ctx.lineWidth = 14;
        ctx.lineCap = 'round';
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        // Main stroke
        ctx.strokeStyle = FOCUS_COLOR;
        ctx.lineWidth = 4;
        ctx.globalAlpha = 1;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.restore();

        // Endpoint handles when in resize mode
        if (appState.resizeWallId === focusedWall.id) {
          drawWallEndpointHandles(ctx, x1, y1, x2, y2, FOCUS_COLOR, focusedWall.id);
        }
      }
    }

    // ── Lasso-selected wall highlights ───────────────────
    if (appState.selectedWalls && appState.selectedWalls.size > 0) {
      appState.selectedWalls.forEach(wallId => {
        const wall = walls.find(w => w.id === wallId);
        if (!wall || wall.x1_pct == null) return;
        const x1 = wall.x1_pct * W, y1 = wall.y1_pct * H;
        const x2 = wall.x2_pct * W, y2 = wall.y2_pct * H;
        ctx.save();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 6;
        ctx.globalAlpha = 0.45;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = 1;
        ctx.setLineDash([8, 5]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      });
    }

    // ── Lasso selection rectangle ────────────────────────
    if (appState.lassoState?.active) {
      const { x1, y1, x2, y2 } = appState.lassoState;
      const rx = Math.min(x1, x2) * W;
      const ry = Math.min(y1, y2) * H;
      const rw = Math.abs(x2 - x1) * W;
      const rh = Math.abs(y2 - y1) * H;
      ctx.save();
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.fillStyle = 'rgba(0,212,255,0.06)';
      ctx.fillRect(rx, ry, rw, rh);
      ctx.setLineDash([]);
      ctx.restore();
    }

    // Draw each pinned wall's measurement independently (supports multi-select).
    if (appState.layers.dims && appState.visibleWalls.size > 0) {
      appState.visibleWalls.forEach(wallId => {
        const idx  = walls.findIndex(w => w.id === wallId);
        if (idx < 0) return;
        const wall = walls[idx];

        const dimLine = dimLines.find(dl => dl.wallId === wallId) || null;
        const hasCoords = wall.x1_pct != null && wall.y1_pct != null &&
                          wall.x2_pct != null && wall.y2_pct != null;
        const target = dimLine || (hasCoords ? {
          x1_pct: wall.x1_pct, y1_pct: wall.y1_pct,
          x2_pct: wall.x2_pct, y2_pct: wall.y2_pct,
          label:  wall.length || null,
        } : null);

            if (target) drawHighlightedDimLine(ctx, target, idx, W, H, wallId);
      });
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
function drawHighlightedDimLine(ctx, dl, idx, W, H, wallId) {
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

  drawWallEndpointHandles(ctx, x1, y1, x2, y2, color, wallId);

  if (appState.layers.labels && dl.label) {
    drawDimLabelColored(ctx, dl.label, x1, y1, x2, y2, color);
  }
}

/** Draw draggable endpoint handles for a wall segment. */
function drawWallEndpointHandles(ctx, x1, y1, x2, y2, color, wallId) {
  [[x1, y1], [x2, y2]].forEach(([px, py], epIdx) => {
    const isHovered  = appState.hoveredEndpoint
      && appState.hoveredEndpoint.wallId === wallId
      && appState.hoveredEndpoint.endpointIdx === epIdx;
    const isDragging = appState.dragState
      && appState.dragState.wallId === wallId
      && appState.dragState.endpointIdx === epIdx;
    const active = isHovered || isDragging;
    const r = active ? 7 : 5;

    if (active) {
      ctx.beginPath();
      ctx.arc(px, py, r + 4, 0, Math.PI * 2);
      ctx.strokeStyle = color + '66';
      ctx.lineWidth   = 2;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fillStyle = active ? '#ffffff' : color;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.strokeStyle = active ? color : 'rgba(0,0,0,0.6)';
    ctx.lineWidth = active ? 2.5 : 1.5;
    ctx.stroke();
  });
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

// ── Endpoint drag helper ─────────────────────────────────
/**
 * Project a cursor position (in pct-space) onto the axis of an existing wall,
 * anchored at the OPPOSITE endpoint from the one being dragged.
 *
 * Axis-lock ensures the wall angle never changes — the dragged endpoint can
 * only slide along the line's current direction vector.
 *
 * @param {object} wall        - Wall object with x1_pct … y2_pct
 * @param {number} endpointIdx - 0 = dragging start (P1), 1 = dragging end (P2)
 * @param {number} cursorXPct  - Mouse X as fraction of canvas width
 * @param {number} cursorYPct  - Mouse Y as fraction of canvas height
 * @returns {{ newXPct: number, newYPct: number }}
 */
function _projectEndpoint(wall, endpointIdx, cursorXPct, cursorYPct) {
  // Anchor is the opposite endpoint (stays fixed).
  const anchorX = endpointIdx === 0 ? wall.x2_pct : wall.x1_pct;
  const anchorY = endpointIdx === 0 ? wall.y2_pct : wall.y1_pct;

  // Current dragged endpoint (used to build the unit axis vector).
  const dragX = endpointIdx === 0 ? wall.x1_pct : wall.x2_pct;
  const dragY = endpointIdx === 0 ? wall.y1_pct : wall.y2_pct;

  // Unit vector pointing from anchor toward the dragged endpoint.
  const rawDx = dragX - anchorX;
  const rawDy = dragY - anchorY;
  const rawLen = Math.sqrt(rawDx * rawDx + rawDy * rawDy) || 1e-9;
  const unitX = rawDx / rawLen;
  const unitY = rawDy / rawLen;

  // Project cursor onto axis through anchor.
  const t = (cursorXPct - anchorX) * unitX + (cursorYPct - anchorY) * unitY;

  return {
    newXPct: anchorX + t * unitX,
    newYPct: anchorY + t * unitY,
  };
}

/**
 * Return the ID of the wall whose line segment is closest to (canvasX, canvasY),
 * within HIT_RADIUS pixels, or null if none found.
 * Used for room wall-assignment clicks.
 */
function _findWallLineHit(canvasX, canvasY, W, H) {
  const HIT_RADIUS = 10;
  const result = appState.analysisResult;
  if (!result) return null;

  const walls = result.walls || [];
  let bestWallId = null;
  let bestDist   = HIT_RADIUS;

  walls.forEach(wall => {
    if (wall.x1_pct == null || wall.y1_pct == null) return;
    const x1 = wall.x1_pct * W, y1 = wall.y1_pct * H;
    const x2 = wall.x2_pct * W, y2 = wall.y2_pct * H;
    const dx = x2 - x1, dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    const t = lenSq > 0
      ? Math.max(0, Math.min(1, ((canvasX - x1) * dx + (canvasY - y1) * dy) / lenSq))
      : 0;
    const px   = x1 + t * dx, py = y1 + t * dy;
    const dist = Math.sqrt((canvasX - px) ** 2 + (canvasY - py) ** 2);
    if (dist < bestDist) {
      bestDist   = dist;
      bestWallId = wall.id;
    }
  });

  return bestWallId;
}

/**
 * Return the first { wallId, endpointIdx } within HIT_RADIUS pixels of
 * (canvasX, canvasY), or null.
 */
function _findEndpointHit(canvasX, canvasY, W, H) {
  const HIT_RADIUS = 12;
  const result = appState.analysisResult;
  if (!result) return null;

  const walls = result.walls || [];
  let best = null;
  let bestDist = HIT_RADIUS;

  appState.visibleWalls.forEach(wallId => {
    _checkWallEndpoints(wallId, walls, canvasX, canvasY, W, H, (hit, dist) => {
      if (dist < bestDist) {
        bestDist = dist;
        best = hit;
      }
    });
  });

  if (appState.resizeWallId) {
    _checkWallEndpoints(appState.resizeWallId, walls, canvasX, canvasY, W, H, (hit, dist) => {
      if (dist < bestDist) {
        bestDist = dist;
        best = hit;
      }
    });
  }

  return best;
}

function _checkWallEndpoints(wallId, walls, canvasX, canvasY, W, H, onHit) {
  const wall = walls.find(w => w.id === wallId);
  if (!wall) return;

  const pts = [
    { x: wall.x1_pct * W, y: wall.y1_pct * H, idx: 0 },
    { x: wall.x2_pct * W, y: wall.y2_pct * H, idx: 1 },
  ];
  pts.forEach(({ x, y, idx }) => {
    const d = Math.sqrt((canvasX - x) ** 2 + (canvasY - y) ** 2);
    onHit({ wallId, endpointIdx: idx }, d);
  });
}

/**
 * Sync the overlay-canvas pointer-events so mouse interactions (hover + drag)
 * are captured whenever needed:
 *   - draw-wall mode is active, OR
 *   - at least one wall is pinned/visible (endpoints become draggable).
 * Otherwise the canvas is transparent to mouse events so the plan image beneath
 * remains fully clickable.
 */
function _syncCanvasPointerEvents() {
  const canvas = document.getElementById('overlay-canvas');
  if (!canvas) return;
  // Enable pointer events whenever there are walls (canvas click highlights them),
  // or when a special mode is active.
  const hasWalls = (appState.analysisResult?.walls || []).length > 0;
  const needEvents = hasWalls || appState.drawWallMode || appState.visibleWalls.size > 0
    || appState.resizeWallId != null;
  canvas.style.pointerEvents = needEvents ? 'auto' : 'none';
}

/**
 * Toggle a wall's highlight from a canvas click, syncing the sidebar item.
 * Equivalent to clicking the wall row in the sidebar list.
 */
function _toggleWallHighlightFromCanvas(wallId) {
  const result = appState.analysisResult;
  if (!result) return;
  const wallIdx = (result.walls || []).findIndex(w => w.id === wallId);
  if (wallIdx < 0) return;
  const listEl = document.getElementById('wall-list');
  const items  = listEl ? listEl.querySelectorAll('.wall-item') : [];
  const itemEl = items[wallIdx];
  if (itemEl) {
    toggleWallVisibility(wallId, itemEl);
    // Scroll the sidebar item into view so the user can see it
    itemEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  } else {
    // Sidebar not rendered yet — just update state + canvas
    if (appState.visibleWalls.has(wallId)) {
      appState.visibleWalls.delete(wallId);
    } else {
      appState.visibleWalls.add(wallId);
    }
    drawCanvas();
  }
}

/**
 * Toggle the DRAW WALL two-click mode.
 * First click sets the start point; second click creates the wall.
 */
function toggleDrawWallMode(btn) {
  appState.drawWallMode = !appState.drawWallMode;
  appState.drawWallFirstPoint = null;
  appState._drawCursor = null;
  if (appState.drawWallMode) clearCanvasWallFocus();
  btn.classList.toggle('active', appState.drawWallMode);
  const canvas = document.getElementById('overlay-canvas');
  canvas.style.cursor = appState.drawWallMode ? 'crosshair' : '';
  _syncCanvasPointerEvents();
  if (!appState.drawWallMode) drawCanvas();  // clear rubber-band preview
}

/**
 * Return IDs of walls whose midpoints fall within the given pct-space rectangle.
 * Used by the lasso selection to find all walls in the dragged region.
 */
function _getWallsInRect(x1Pct, y1Pct, x2Pct, y2Pct) {
  const result = appState.analysisResult;
  if (!result) return [];
  return (result.walls || [])
    .filter(w => {
      if (w.x1_pct == null) return false;
      const mx = (w.x1_pct + w.x2_pct) / 2;
      const my = (w.y1_pct + w.y2_pct) / 2;
      return mx >= x1Pct && mx <= x2Pct && my >= y1Pct && my <= y2Pct;
    })
    .map(w => w.id);
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
    // Ignore clicks that were actually the end of an endpoint drag or lasso
    if (canvas._dragConsumedClick) { canvas._dragConsumedClick = false; return; }

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W === 0 || H === 0) return;

    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (!appState.drawWallMode) {
      const wallId = _findWallLineHit(cx, cy, W, H);
      if (wallId) {
        e.stopPropagation();
        openWallActionPopover(wallId, e.clientX, e.clientY);
      } else {
        clearCanvasWallFocus();
      }
      return;
    }

    // Draw-wall two-click mode
    const xPct = cx / W;
    const yPct = cy / H;

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

  // ── Endpoint drag / lasso: mousedown ─────────────────
  canvas.addEventListener('mousedown', (e) => {
    if (appState.drawWallMode) return;
    if (e.button !== 0) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W === 0 || H === 0) return;

    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    // Shift+drag starts a lasso selection
    if (e.shiftKey) {
      e.preventDefault();
      clearCanvasWallFocus();
      appState.lassoState = { x1: cx / W, y1: cy / H, x2: cx / W, y2: cy / H, active: true };
      canvas._dragConsumedClick = true;
      return;
    }

    const hit = _findEndpointHit(cx, cy, W, H);
    if (!hit) return;

    e.preventDefault();
    e.stopPropagation();
    canvas._dragConsumedClick = true;

    const result = appState.analysisResult;
    const wall   = (result.walls || []).find(w => w.id === hit.wallId);
    if (!wall) return;

    // Build the unit axis vector once and store it in dragState so it never
    // shifts during the drag (prevents axis-drift from floating-point).
    const rawDx = wall.x2_pct - wall.x1_pct;
    const rawDy = wall.y2_pct - wall.y1_pct;
    const rawLen = Math.sqrt(rawDx * rawDx + rawDy * rawDy) || 1e-9;

    appState.dragState = {
      wallId:       hit.wallId,
      endpointIdx:  hit.endpointIdx,
      anchorXPct:   hit.endpointIdx === 0 ? wall.x2_pct : wall.x1_pct,
      anchorYPct:   hit.endpointIdx === 0 ? wall.y2_pct : wall.y1_pct,
      // Unit vector pointing FROM anchor TOWARD dragged endpoint
      unitX: (hit.endpointIdx === 0 ? -rawDx : rawDx) / rawLen,
      unitY: (hit.endpointIdx === 0 ? -rawDy : rawDy) / rawLen,
    };
    appState.hoveredEndpoint = null;
    canvas.style.cursor = 'grabbing';
  });

  // ── Endpoint drag / lasso: mousemove ─────────────────
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W === 0 || H === 0) return;

    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    // Active lasso drag — update the rect and redraw
    if (appState.lassoState?.active) {
      appState.lassoState.x2 = cx / W;
      appState.lassoState.y2 = cy / H;
      canvas.style.cursor = 'crosshair';
      drawCanvas();
      return;
    }

    if (appState.dragState) {
      // Active endpoint drag — project cursor onto axis and update wall coords
      const ds = appState.dragState;
      const result = appState.analysisResult;
      const wall   = (result.walls || []).find(w => w.id === ds.wallId);
      if (!wall) return;

      const cxPct = cx / W;
      const cyPct = cy / H;
      const t = (cxPct - ds.anchorXPct) * ds.unitX + (cyPct - ds.anchorYPct) * ds.unitY;
      // Prevent collapsing to zero length (minimum 5 px in pct-space ≈ 0.005)
      const MIN_T = 5 / Math.min(W, H);
      const clampedT = Math.max(MIN_T, t);
      const newXPct = ds.anchorXPct + clampedT * ds.unitX;
      const newYPct = ds.anchorYPct + clampedT * ds.unitY;

      if (ds.endpointIdx === 0) {
        wall.x1_pct = newXPct; wall.y1_pct = newYPct;
      } else {
        wall.x2_pct = newXPct; wall.y2_pct = newYPct;
      }

      // Keep dimension_lines in sync
      const dl = (result.dimension_lines || []).find(d => d.wallId === ds.wallId);
      if (dl) {
        dl.x1_pct = wall.x1_pct; dl.y1_pct = wall.y1_pct;
        dl.x2_pct = wall.x2_pct; dl.y2_pct = wall.y2_pct;
      }

      drawCanvas();
      return;
    }

    if (appState.drawWallMode) return;

    // Endpoint hover — grab cursor takes priority
    const epHit = _findEndpointHit(cx, cy, W, H);
    const prev  = appState.hoveredEndpoint;
    const epChanged = (epHit?.wallId !== prev?.wallId) || (epHit?.endpointIdx !== prev?.endpointIdx);
    if (epChanged) {
      appState.hoveredEndpoint = epHit || null;
      drawCanvas();
    }

    // Cursor: grab on endpoint, pointer on wall line, default otherwise
    if (epHit) {
      canvas.style.cursor = 'grab';
    } else {
      const wallHit = _findWallLineHit(cx, cy, W, H);
      canvas.style.cursor = wallHit ? 'pointer' : (e.shiftKey ? 'crosshair' : '');
    }
  });

  // ── Endpoint drag / lasso: mouseup ───────────────────
  canvas.addEventListener('mouseup', (e) => {
    if (e.button !== 0) return;

    // Complete a lasso selection
    if (appState.lassoState?.active) {
      const rect = canvas.getBoundingClientRect();
      const W = rect.width, H = rect.height;
      const lasso = appState.lassoState;
      lasso.active = false;
      appState.lassoState = null;

      const minX = Math.min(lasso.x1, lasso.x2);
      const maxX = Math.max(lasso.x1, lasso.x2);
      const minY = Math.min(lasso.y1, lasso.y2);
      const maxY = Math.max(lasso.y1, lasso.y2);

      // Only act if the user dragged a meaningful distance (>5px)
      if ((maxX - minX) * W > 5 || (maxY - minY) * H > 5) {
        const wallIds = _getWallsInRect(minX, minY, maxX, maxY);
        if (wallIds.length > 0) {
          appState.selectedWalls = new Set(wallIds);
          appState.canvasFocusedWallId = null;
          appState.resizeWallId = null;
          closeWallActionPopover();
          _showLassoBar();
        }
      }

      canvas.style.cursor = '';
      drawCanvas();
      return;
    }

    if (!appState.dragState) return;

    const { wallId } = appState.dragState;
    appState.dragState = null;
    canvas.style.cursor = '';

    // Recompute length and update sidebar label
    recalculateWallLength(wallId);
    drawCanvas();
  });

  // Also release drag/lasso if mouse leaves the canvas
  canvas.addEventListener('mouseleave', () => {
    if (appState.lassoState?.active) {
      appState.lassoState = null;
      canvas.style.cursor = '';
      drawCanvas();
    }
    if (appState.dragState) {
      const { wallId } = appState.dragState;
      appState.dragState = null;
      canvas.style.cursor = '';
      recalculateWallLength(wallId);
      drawCanvas();
    }
    if (appState.hoveredEndpoint) {
      appState.hoveredEndpoint = null;
      drawCanvas();
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
