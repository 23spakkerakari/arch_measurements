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

    // ── Room wall highlights ─────────────────────────────
    // Draw subtle colored strokes along walls that belong to any room,
    // with the active room drawn more prominently.
    appState.rooms.forEach(room => {
      const isActive = room.id === appState.activeRoomId;
      room.wallIds.forEach(wallId => {
        const wall = walls.find(w => w.id === wallId);
        if (!wall || wall.x1_pct == null) return;
        const x1 = wall.x1_pct * W, y1 = wall.y1_pct * H;
        const x2 = wall.x2_pct * W, y2 = wall.y2_pct * H;
        ctx.save();
        ctx.strokeStyle = room.color;
        ctx.lineWidth   = isActive ? 10 : 5;
        ctx.lineCap     = 'round';
        ctx.globalAlpha = isActive ? 0.38 : 0.18;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.restore();
      });
    });

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

  // Endpoint dots — enlarged + ring when hovered or actively dragged
  [[x1, y1], [x2, y2]].forEach(([px, py], epIdx) => {
    const isHovered  = appState.hoveredEndpoint
      && appState.hoveredEndpoint.wallId === wallId
      && appState.hoveredEndpoint.endpointIdx === epIdx;
    const isDragging = appState.dragState
      && appState.dragState.wallId === wallId
      && appState.dragState.endpointIdx === epIdx;
    const active = isHovered || isDragging;
    const r = active ? 7 : 4;

    // Outer glow ring when active
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
    const wall = walls.find(w => w.id === wallId);
    if (!wall) return;

    const pts = [
      { x: wall.x1_pct * W, y: wall.y1_pct * H, idx: 0 },
      { x: wall.x2_pct * W, y: wall.y2_pct * H, idx: 1 },
    ];
    pts.forEach(({ x, y, idx }) => {
      const d = Math.sqrt((canvasX - x) ** 2 + (canvasY - y) ** 2);
      if (d < bestDist) {
        bestDist = d;
        best = { wallId, endpointIdx: idx };
      }
    });
  });

  return best;
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
  const needEvents = hasWalls || appState.drawWallMode || appState.visibleWalls.size > 0 || !!appState.activeRoomId;
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
  btn.classList.toggle('active', appState.drawWallMode);
  const canvas = document.getElementById('overlay-canvas');
  canvas.style.cursor = appState.drawWallMode ? 'crosshair' : '';
  _syncCanvasPointerEvents();
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
    // Ignore clicks that were actually the end of an endpoint drag
    if (canvas._dragConsumedClick) { canvas._dragConsumedClick = false; return; }

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W === 0 || H === 0) return;

    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (!appState.drawWallMode) {
      const wallId = _findWallLineHit(cx, cy, W, H);
      if (wallId) {
        if (appState.activeRoomId) {
          // Room-assignment mode
          toggleWallRoomAssignment(wallId);
        } else {
          // Highlight mode — show/hide the wall measurement on the plan
          _toggleWallHighlightFromCanvas(wallId);
        }
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

  // ── Endpoint drag: mousedown ──────────────────────────
  canvas.addEventListener('mousedown', (e) => {
    if (appState.drawWallMode) return;
    if (e.button !== 0) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W === 0 || H === 0) return;

    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

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

  // ── Endpoint drag: mousemove ──────────────────────────
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W === 0 || H === 0) return;

    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (appState.dragState) {
      // Active drag — project cursor onto axis and update wall coords
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

    // Wall-line hover — pointer cursor when not near an endpoint
    if (!epHit && !appState.activeRoomId) {
      const wallHit = _findWallLineHit(cx, cy, W, H);
      canvas.style.cursor = wallHit ? 'pointer' : '';
    } else if (epHit) {
      canvas.style.cursor = 'grab';
    } else if (appState.activeRoomId) {
      const wallHit = _findWallLineHit(cx, cy, W, H);
      canvas.style.cursor = wallHit ? 'crosshair' : '';
    }
  });

  // ── Endpoint drag: mouseup ────────────────────────────
  canvas.addEventListener('mouseup', (e) => {
    if (!appState.dragState) return;
    if (e.button !== 0) return;

    const { wallId } = appState.dragState;
    appState.dragState = null;
    canvas.style.cursor = '';

    // Recompute length and update sidebar label
    recalculateWallLength(wallId);
    drawCanvas();
  });

  // Also release drag if mouse leaves the canvas
  canvas.addEventListener('mouseleave', () => {
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
