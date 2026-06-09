/**
 * analysis.js
 * Sends the plan (image or PDF) to the Claude vision API and renders takeoff results.
 *
 * When scale mode is "manual", the prompt asks Claude to return wall endpoints
 * as percentage coordinates. Wall lengths are then computed client-side using
 * the user's DPI and scale, avoiding LLM arithmetic errors.
 */

let logTimer = null;

async function detectScaleQuick(imageDataUrl) {
  const base64 = imageDataUrl.split(',')[1];
  const mimeType = imageDataUrl.split(';')[0].replace('data:', '');
  const headers = { 'Content-Type': 'application/json' };
  if (CONFIG.ANTHROPIC_API_KEY) {
    headers['x-api-key'] = CONFIG.ANTHROPIC_API_KEY;
    headers['anthropic-version'] = '2023-06-01';
  }
  const response = await fetch(CONFIG.API_ENDPOINT, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: CONFIG.MODEL,
      max_tokens: 256,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: mimeType, data: base64 } },
          { type: 'text', text: 'Return ONLY JSON: {"detected_scale":"1:50"} with the drawing scale. No markdown.' },
        ],
      }],
    }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error?.message || `Scale detect failed ${response.status}`);
  }
  const data = await response.json();
  const rawText = data.content.map(c => c.text || '').join('').trim();
  return parseAiJson(rawText).detected_scale;
}

/**
 * After OpenCV detects walls, make a focused Claude call to assign each wall
 * to a room label visible in the floor plan.
 *
 * Sends the image + a compact list of wall midpoints; Claude returns a JSON
 * map of { wall_id → room_name }. Falls back gracefully if Claude is
 * unreachable or the response is unparseable.
 *
 * @param {string} imageDataUrl  - base64 data URL of the plan image
 * @param {Array}  walls         - wall objects from CV (must have .id, .*_pct)
 * @returns {Array} walls with .room field populated where detected
 */
/**
 * Map geometric room cells (R1, R2, …) to visible plan text labels via Claude.
 * One label per room centroid — cheaper and more stable than per-wall labeling.
 *
 * @param {string} imageDataUrl
 * @param {Array}  rooms  - from CV rooms[] with centroid_pct
 * @returns {Array} rooms with .label populated where detected
 */
async function assignRoomLabels(imageDataUrl, rooms) {
  if (!rooms.length) return rooms;

  const roomList = rooms
    .filter(r => r.centroid_pct)
    .map(r => {
      const [mx, my] = r.centroid_pct;
      return `  {"id":"${r.id}","centroid":[${mx.toFixed(3)},${my.toFixed(3)}]}`;
    })
    .join(',\n');

  const prompt = `You are reading an architectural floor plan image.
Below is a JSON array of detected room cells. Each entry has an "id" (R1, R2, …) and "centroid" as [x,y] fractions of the image (0,0=top-left, 1,1=bottom-right).

Room cells:
[
${roomList}
]

For each room cell, identify the room name or number printed on the plan nearest that centroid (e.g. "MANAGER'S OFFICE", "302", "LOBBY"). Use null if no label is clearly associated.

Return ONLY valid JSON — no markdown, no explanation:
{"room_labels":[{"id":"<room_id>","label":"<text or null>"}]}`;

  const base64   = imageDataUrl.split(',')[1];
  const mimeType = imageDataUrl.split(';')[0].replace('data:', '');

  const headers = { 'Content-Type': 'application/json' };
  if (CONFIG.ANTHROPIC_API_KEY) {
    headers['x-api-key']          = CONFIG.ANTHROPIC_API_KEY;
    headers['anthropic-version']  = '2023-06-01';
  }

  const response = await fetch(CONFIG.API_ENDPOINT, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model:      CONFIG.MODEL,
      max_tokens: 1024,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: mimeType, data: base64 } },
          { type: 'text',  text: prompt },
        ],
      }],
    }),
  });

  if (!response.ok) throw new Error(`Room label API error ${response.status}`);

  const data    = await response.json();
  const rawText = data.content.map(c => c.text || '').join('').trim();
  const parsed  = parseAiJson(rawText);

  if (!parsed.room_labels || !Array.isArray(parsed.room_labels)) return rooms;

  const labelMap = new Map(parsed.room_labels.map(e => [e.id, e.label || null]));
  return rooms.map(r => ({
    ...r,
    label: labelMap.has(r.id) ? labelMap.get(r.id) : (r.label || null),
  }));
}

/** Apply room labels from geometric cells onto wall sub-segments. */
function applyRoomLabelsToWalls(walls, rooms) {
  if (!rooms.length) return walls;
  const byId = new Map(rooms.map(r => [r.id, r.label || null]));
  return walls.map(w => {
    const label = w.room_id ? byId.get(w.room_id) : null;
    return { ...w, room: label || w.room || null };
  });
}

async function assignRoomsToCvWalls(imageDataUrl, walls) {
  if (!walls.length) return walls;

  // Build a compact midpoint list for the prompt
  const wallList = walls
    .filter(w => w.x1_pct != null)
    .map(w => {
      const mx = ((w.x1_pct + w.x2_pct) / 2).toFixed(3);
      const my = ((w.y1_pct + w.y2_pct) / 2).toFixed(3);
      return `  {"id":"${w.id}","mid":[${mx},${my}]}`;
    })
    .join(',\n');

  const prompt = `You are reading an architectural floor plan image.
Below is a JSON array of detected wall segments. Each entry has an "id" and "mid" (midpoint as [x,y] fractions of the image: 0,0=top-left, 1,1=bottom-right).

Wall segments:
[
${wallList}
]

For each wall, identify which labeled room, unit, or space it belongs to based on the visible room numbers, unit numbers, or room names in the drawing (e.g. "302", "Bedroom", "Living Room", "Corridor"). A wall belongs to a room if it forms part of that room's boundary. Use null for walls with no clearly associated label (e.g. exterior walls between units).

Return ONLY valid JSON — no markdown, no explanation:
{"wall_rooms":[{"id":"<wall_id>","room":"<label or null>"}]}`;

  const base64   = imageDataUrl.split(',')[1];
  const mimeType = imageDataUrl.split(';')[0].replace('data:', '');

  const headers = { 'Content-Type': 'application/json' };
  if (CONFIG.ANTHROPIC_API_KEY) {
    headers['x-api-key']          = CONFIG.ANTHROPIC_API_KEY;
    headers['anthropic-version']  = '2023-06-01';
  }

  const response = await fetch(CONFIG.API_ENDPOINT, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model:      CONFIG.MODEL,
      max_tokens: 1024,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: mimeType, data: base64 } },
          { type: 'text',  text: prompt },
        ],
      }],
    }),
  });

  if (!response.ok) throw new Error(`Room label API error ${response.status}`);

  const data    = await response.json();
  const rawText = data.content.map(c => c.text || '').join('').trim();
  const parsed  = parseAiJson(rawText);

  if (!parsed.wall_rooms || !Array.isArray(parsed.wall_rooms)) return walls;

  // Build a lookup map and apply to walls
  const roomMap = new Map(parsed.wall_rooms.map(e => [e.id, e.room || null]));
  return walls.map(w => ({
    ...w,
    room: roomMap.has(w.id) ? roomMap.get(w.id) : (w.room || null),
  }));
}

async function startAnalysis() {
  if (!appState.buildingRoi) {
    const el = document.getElementById('roi-status');
    if (el) {
      el.textContent = 'Please draw a box around the floor plan first.';
      el.style.color = '#ff6496';
      setTimeout(() => { el.style.color = ''; updateRoiStatus(); }, 2500);
    }
    return;
  }
  goToStep(3);
  document.getElementById('log-lines').innerHTML = '';
  let logIdx = 0;
  logTimer = setInterval(() => {
    if (logIdx < LOG_MSGS.length) {
      const msg = LOG_MSGS[logIdx++];
      addLog(msg);
      document.getElementById('analyzing-text').textContent = msg.toUpperCase().replace('…', '');
    }
  }, 800);

  const scaleMode   = appState.scaleMode;
  const manualScale = document.getElementById('manual-scale').value;
  const units  = appState.units;
  const detail = 'full';

  const isManual = scaleMode === 'manual';
  if (isManual) {
    const inputDpi = parseInt(document.getElementById('image-dpi').value, 10);
    if (inputDpi) appState.imageDpi = inputDpi;
  }
  const sourceUrl = appState.imageDataUrl || appState.fileDataUrl;
  if (!sourceUrl) {
    clearInterval(logTimer);
    throw new Error('No plan image available. Re-upload the file.');
  }

  try {
    addLog('Running CV wall detection…');
    let scaleStr = isManual ? manualScale : null;
    if (!scaleStr) {
      addLog('Detecting drawing scale…');
      scaleStr = await detectScaleQuick(sourceUrl);
    }
    const dpi = appState.imageDpi || parseInt(document.getElementById('image-dpi').value, 10) || 150;

    let parsed = await runCvAnalyze(sourceUrl, scaleStr, dpi, appState.buildingRoi);
    parsed = applyUserRoiToResult(parsed);
    clearInterval(logTimer);

    // Map geometric room cells to plan text labels, then tag walls
    try {
      addLog('Detecting room labels…');
      if (parsed.rooms && parsed.rooms.length) {
        parsed.rooms = await assignRoomLabels(sourceUrl, parsed.rooms);
        parsed.walls = applyRoomLabelsToWalls(parsed.walls || [], parsed.rooms);
      } else {
        parsed.walls = await assignRoomsToCvWalls(sourceUrl, parsed.walls || []);
      }
    } catch (roomErr) {
      addLog('Room detection skipped: ' + roomErr.message);
    }

    appState.analysisResult = parsed;
    buildRoomsFromWalls(parsed.walls || [], parsed.rooms || []);
    addLog('CV analysis complete!');
    setTimeout(() => renderResults(parsed), 400);
    return;
  } catch (cvErr) {
    addLog('CV failed: ' + cvErr.message);
    addLog('Falling back to vision model…');
  }

  const isManualFallback = scaleMode === 'manual';

  const scaleInstruction = isManualFallback
    ? `The drawing scale is EXACTLY ${manualScale}. Do NOT compute wall lengths yourself — return pixel-percentage coordinates instead (see output schema). Set "detected_scale" to "${manualScale}" and "scale_confidence" to "high".`
    : `Auto-detect the scale from any scale bar, north arrow annotation, dimension strings,
       or labeled measurements visible in the drawing.`;

  const detailInstr = detail === 'full'
    ? 'Identify every wall (exterior and interior). For each wall provide its name and facing direction (North/South/East/West).'
    : detail === 'detailed'
    ? 'Identify all walls including interior partitions. For each wall provide its name and facing direction.'
    : 'Identify the main exterior walls. For each wall provide its name and facing direction.';

  const prompt = buildPrompt(scaleInstruction, detailInstr, units, isManual);

  try {
    const base64   = sourceUrl.split(',')[1];
    const mimeType = sourceUrl.split(';')[0].replace('data:', '');

    const fileBlock = {
      type: 'image',
      source: { type: 'base64', media_type: mimeType, data: base64 },
    };

    const headers = { 'Content-Type': 'application/json' };
    if (CONFIG.ANTHROPIC_API_KEY) {
      headers['x-api-key'] = CONFIG.ANTHROPIC_API_KEY;
      headers['anthropic-version'] = '2023-06-01';
    }

    const response = await fetch(CONFIG.API_ENDPOINT, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: CONFIG.MODEL,
        max_tokens: CONFIG.MAX_TOKENS,
        messages: [{
          role: 'user',
          content: [fileBlock, { type: 'text', text: prompt }],
        }],
      }),
    });

    clearInterval(logTimer);

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error?.message || `API error ${response.status}`);
    }

    const data       = await response.json();
    const rawText    = data.content.map(c => c.text || '').join('').trim();
    const stopReason = data.stop_reason;

    let parsed;
    try {
      parsed = parseAiJson(rawText, stopReason);
    } catch (e) {
      throw new Error('Could not parse AI response as JSON: ' + e.message);
    }

    parsed = await normalizeAnalysisCoords(parsed, appState.imageDataUrl);

    if (isManualFallback) {
      parsed = computeWallLengths(parsed, manualScale, appState.imageDpi, units);
    }


    appState.analysisResult = parsed;
    buildRoomsFromWalls(parsed.walls || [], parsed.rooms || []);
    addLog('Analysis complete!');
    setTimeout(() => renderResults(parsed), 400);

  } catch (err) {
    clearInterval(logTimer);
    addLog('ERROR: ' + err.message);
    goToStep(4);
    const container = document.getElementById('error-container');
    container.innerHTML = `<div class="error-banner">⚠ Analysis failed: ${err.message}. Please try again with a clearer plan.</div>`;
    container.classList.remove('hidden');
  }
}

// ── JSON parser (handles markdown fences, trailing commas, truncation) ──
function parseAiJson(rawText, stopReason) {
  let text = rawText
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();

  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) throw new Error('No JSON object found in response');

  const cleaned = jsonMatch[0].replace(/,\s*([\]}])/g, '$1');

  try {
    return JSON.parse(cleaned);
  } catch (firstErr) {
    const repaired = repairTruncatedJson(cleaned);
    if (repaired) {
      try {
        return JSON.parse(repaired);
      } catch { /* fall through */ }
    }

    const hint = stopReason === 'max_tokens'
      ? ' Response was cut off — try Standard detail level or a single-page plan.'
      : '';
    throw new Error(firstErr.message + hint);
  }
}

function repairTruncatedJson(s) {
  let attempt = s;
  for (let tries = 0; tries < 100; tries++) {
    const lastClose = attempt.lastIndexOf('}');
    if (lastClose < 0) return null;
    attempt = attempt.slice(0, lastClose + 1).replace(/,\s*$/, '');
    const suffix = closeOpenBrackets(attempt);
    try {
      JSON.parse(attempt + suffix);
      return attempt + suffix;
    } catch {
      attempt = attempt.slice(0, lastClose);
    }
  }
  return null;
}

function closeOpenBrackets(s) {
  const stack = [];
  let inString = false;
  let escape = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (escape) { escape = false; continue; }
    if (ch === '\\' && inString) { escape = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (ch === '{' || ch === '[') stack.push(ch);
    else if (ch === '}' || ch === ']') stack.pop();
  }
  let suffix = '';
  while (stack.length) {
    suffix += stack.pop() === '{' ? '}' : ']';
  }
  return suffix;
}

// ── Scale parser ────────────────────────────────────────
function parseScale(scaleStr, dpi) {
  const s = scaleStr.trim().toLowerCase()
    .replace(/\s+/g, '')
    .replace(/"/g, 'in')
    .replace(/'/g, 'ft');

  if (s.includes('=')) {
    const [left, right] = s.split('=');
    const paperInches = parseFraction(left.replace(/in(ch(es)?)?/g, ''));
    const realUnits   = parseFloat(right.replace(/ft|foot|feet|m|meter|metre/g, ''));
    return (paperInches * dpi) / realUnits;
  }

  if (s.includes(':')) {
    const [a, b] = s.split(':');
    const ratio = parseFloat(b) / parseFloat(a);
    const pxPerMm = dpi / 25.4;
    return (pxPerMm * 1000) / ratio;
  }

  return null;
}

function parseFraction(s) {
  if (s.includes('/')) {
    const [n, d] = s.split('/');
    return parseFloat(n) / parseFloat(d);
  }
  return parseFloat(s);
}

// ── Compute wall lengths from pixel coordinates ─────────
function computeWallLengths(data, scaleStr, dpi, units) {
  const pxPerUnit = parseScale(scaleStr, dpi);
  if (!pxPerUnit) return data;

  const img = document.getElementById('preview-img');
  const imgW = img.naturalWidth  || img.width  || 1;
  const imgH = img.naturalHeight || img.height || 1;

  const isMetric  = scaleStr.includes(':');
  const unitLabel = isMetric ? 'm' : 'ft';
  const areaLabel = isMetric ? 'm²' : 'ft²';

  const walls = (data.walls || []).map(wall => {
    if (wall.x1_pct != null && wall.y1_pct != null &&
        wall.x2_pct != null && wall.y2_pct != null) {
      const x1 = wall.x1_pct * imgW;
      const y1 = wall.y1_pct * imgH;
      const x2 = wall.x2_pct * imgW;
      const y2 = wall.y2_pct * imgH;
      const pxLen = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
      const realLen = pxLen / pxPerUnit;
      return { ...wall, length: `${realLen.toFixed(1)} ${unitLabel}`, length_raw: realLen };
    }
    return wall;
  });

  let totalArea = data.total_area;
  if (data.area_x1_pct != null && data.area_y1_pct != null &&
      data.area_x2_pct != null && data.area_y2_pct != null) {
    const w = Math.abs(data.area_x2_pct - data.area_x1_pct) * imgW;
    const h = Math.abs(data.area_y2_pct - data.area_y1_pct) * imgH;
    const realW = w / pxPerUnit;
    const realH = h / pxPerUnit;
    totalArea = `${(realW * realH).toFixed(1)} ${areaLabel}`;
  }

  return { ...data, walls, total_area: totalArea };
}

// ── Image crop helper ───────────────────────────────────
function cropImageToRegion(dataUrl, region) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const { x1_pct, y1_pct, x2_pct, y2_pct } = region;
      const sx = Math.round(x1_pct * img.naturalWidth);
      const sy = Math.round(y1_pct * img.naturalHeight);
      const sw = Math.round((x2_pct - x1_pct) * img.naturalWidth);
      const sh = Math.round((y2_pct - y1_pct) * img.naturalHeight);
      const c  = document.createElement('canvas');
      c.width  = sw;
      c.height = sh;
      c.getContext('2d').drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
      resolve(c.toDataURL('image/png'));
    };
    img.src = dataUrl;
  });
}

// ── Coordinate transform: crop-relative → full-image ────
function transformCoordsToFullImage(data, region) {
  const { x1_pct: rx1, y1_pct: ry1, x2_pct: rx2, y2_pct: ry2 } = region;
  const rw = rx2 - rx1, rh = ry2 - ry1;

  function tx(x) { return rx1 + x * rw; }
  function ty(y) { return ry1 + y * rh; }

  const walls = (data.walls || []).map(w => ({
    ...w,
    ...(w.x1_pct != null ? { x1_pct: tx(w.x1_pct), y1_pct: ty(w.y1_pct), x2_pct: tx(w.x2_pct), y2_pct: ty(w.y2_pct) } : {}),
  }));
  const dimension_lines = (data.dimension_lines || []).map(dl => ({
    ...dl,
    ...(dl.x1_pct != null ? { x1_pct: tx(dl.x1_pct), y1_pct: ty(dl.y1_pct), x2_pct: tx(dl.x2_pct), y2_pct: ty(dl.y2_pct) } : {}),
  }));
  return { ...data, walls, dimension_lines };
}

// ── Prompt builder ──────────────────────────────────────
function buildPrompt(scaleInstruction, detailInstr, units, isManual) {
  const unitLabel = units === 'metric'
    ? 'metres (m) and square metres (m²)'
    : 'feet (ft) and square feet (ft²)';

  const wallSchema = isManual
    ? `    {
      "name": "North Wall of Room 445",
      "facing": "North",
      "room": "445",
      "x1_pct": 0.12,
      "y1_pct": 0.05,
      "x2_pct": 0.88,
      "y2_pct": 0.05
    }`
    : `    {
      "name": "North Wall of Room 445",
      "facing": "North",
      "room": "445",
      "length": "5.2 m",
      "x1_pct": 0.12,
      "y1_pct": 0.05,
      "x2_pct": 0.88,
      "y2_pct": 0.05
    }`;

  const imgSize = appState.planImageSize;
  const sizeNote = imgSize
    ? `The provided image is exactly ${imgSize.w}×${imgSize.h} pixels.\n`
    : '';

  const coordNote = `
COORDINATE SYSTEM (critical — overlays depend on this):
${sizeNote}Return x1_pct, y1_pct, x2_pct, y2_pct as fractions of the FULL PROVIDED IMAGE
(top-left = 0,0; bottom-right = 1,1). x increases left-to-right, y increases top-to-bottom.
Place each endpoint on the wall centerline in the drawing — not on margins, legends,
title blocks, or dimension text.
Optionally include "footprint_bbox" (building only, image fractions) for reference.
${isManual ? 'Do NOT compute wall lengths — return coordinates only.\n' : ''}`;

  return `You are an expert architectural drawing analyst and quantity surveyor.
Analyze this architectural floor plan.

SCALE INSTRUCTIONS: ${scaleInstruction}

MEASUREMENT UNITS: Return all measurements in ${unitLabel}.

DETECTION SCOPE: ${detailInstr}
${coordNote}
CRITICAL: Return ONLY a valid JSON object — no markdown, no explanation, no code
blocks, nothing else. Pure JSON only. Do NOT use trailing commas.

Return exactly this structure:
{
  "detected_scale": "e.g. 1:100 or 1/8 inch = 1 foot",
  "scale_confidence": "high|medium|low",
  "total_area": "e.g. 142.5 m² or 1534 ft²",
  "units": "${units}",
  "walls": [
${wallSchema}
  ]
}

For the "room" field: extract the room/unit number from the drawing (e.g. "302", "Bedroom", "Living Room"). Use null if the wall is not clearly associated with a labeled room.
`;
}

// ── Manual two-point wall creation ──────────────────────
/**
 * Create a wall object from two percentage-coordinate endpoints and add it to
 * the current analysis result. Length is computed client-side from the scale.
 * Called by the DRAW WALL two-click mode in canvas.js.
 */
function createManualWall(x1Pct, y1Pct, x2Pct, y2Pct) {
  const result = appState.analysisResult;
  if (!result) return;

  const [imgW, imgH] = result.image_size_px;
  const scaleStr = result.detected_scale
    || document.getElementById('manual-scale')?.value
    || '1/4"=1ft';
  const dpi = appState.imageDpi
    || parseInt(document.getElementById('image-dpi')?.value, 10)
    || 150;

  const pxPerUnit = parseScale(scaleStr, dpi);
  const isMetric  = scaleStr.includes(':');
  const unitLabel = isMetric ? 'm' : 'ft';

  const dx    = (x2Pct - x1Pct) * imgW;
  const dy    = (y2Pct - y1Pct) * imgH;
  const pxLen = Math.sqrt(dx * dx + dy * dy);
  const realLen = pxPerUnit ? pxLen / pxPerUnit : 0;

  // Angle clockwise from North (image-up), matching preprocess.py's wall_angle_deg.
  const angleDeg = (Math.atan2(dx, -dy) * 180 / Math.PI + 360) % 360;
  const normalAngle = (angleDeg + 90) % 360;
  let facing;
  if (normalAngle >= 315 || normalAngle < 45)  facing = 'North';
  else if (normalAngle < 135)                   facing = 'East';
  else if (normalAngle < 225)                   facing = 'South';
  else                                           facing = 'West';

  const wallId = `w-draw-${Date.now()}`;
  const newWall = {
    id: wallId,
    name: `${facing} Wall (drawn)`,
    facing,
    length:     `${realLen.toFixed(2)} ${unitLabel}`,
    length_raw: realLen,
    x1_pct: x1Pct, y1_pct: y1Pct,
    x2_pct: x2Pct, y2_pct: y2Pct,
    px_coords: [
      Math.round(x1Pct * imgW), Math.round(y1Pct * imgH),
      Math.round(x2Pct * imgW), Math.round(y2Pct * imgH),
    ],
    _userAdded: true,
  };

  result.walls.push(newWall);

  const minSegPx = Math.max(70, Math.min(imgW, imgH) * 0.03);
  const minLenFt = 4;
  const inExclusion = (mx, my) => (my < 0.02 || my > 0.98 || mx < 0.02 || mx > 0.98);
  result.dimension_lines = result.walls.filter(w => {
    const ddx = (w.x2_pct - w.x1_pct) * imgW;
    const ddy = (w.y2_pct - w.y1_pct) * imgH;
    const mx  = (w.x1_pct + w.x2_pct) / 2;
    const my  = (w.y1_pct + w.y2_pct) / 2;
    if (inExclusion(mx, my)) return false;
    if ((w.length_raw || 0) < minLenFt) return false;
    return Math.hypot(ddx, ddy) >= minSegPx;
  }).map(w => ({
    x1_pct: w.x1_pct, y1_pct: w.y1_pct,
    x2_pct: w.x2_pct, y2_pct: w.y2_pct,
    label:  w.length,
    wallId: w.id,
  }));

  renderResults(result);
}

// ── Delete a wall ───────────────────────────────────────
function deleteWall(wallId) {
  const result = appState.analysisResult;
  if (!result) return;

  result.walls = result.walls.filter(w => w.id !== wallId);

  // Clean up room assignments for this wall
  appState.rooms.forEach(r => {
    r.wallIds = r.wallIds.filter(id => id !== wallId);
  });

  const [imgW, imgH] = result.image_size_px || [1, 1];
  const minSegPx = Math.max(70, Math.min(imgW, imgH) * 0.03);
  const minLenFt = 4;
  const inExclusion = (mx, my) => (my < 0.02 || my > 0.98 || mx < 0.02 || mx > 0.98);
  result.dimension_lines = result.walls.filter(w => {
    const dx = (w.x2_pct - w.x1_pct) * imgW;
    const dy = (w.y2_pct - w.y1_pct) * imgH;
    const mx = (w.x1_pct + w.x2_pct) / 2;
    const my = (w.y1_pct + w.y2_pct) / 2;
    if (inExclusion(mx, my)) return false;
    if ((w.length_raw || 0) < minLenFt) return false;
    return Math.hypot(dx, dy) >= minSegPx;
  }).map(w => ({
    x1_pct: w.x1_pct, y1_pct: w.y1_pct,
    x2_pct: w.x2_pct, y2_pct: w.y2_pct,
    label:  w.length,
    wallId: w.id,
  }));

  appState.visibleWalls.delete(wallId);
  renderResults(result);
}

// ── Recalculate wall length after endpoint drag ──────────
/**
 * Recompute `length` / `length_raw` for a wall whose coordinates have just
 * been updated by dragging an endpoint, then patch the sidebar label in-place
 * without triggering a full re-render.
 */
function recalculateWallLength(wallId) {
  const result = appState.analysisResult;
  if (!result) return;

  const wall = (result.walls || []).find(w => w.id === wallId);
  if (!wall) return;

  const [imgW, imgH] = result.image_size_px || [1, 1];
  const scaleStr = result.detected_scale
    || document.getElementById('manual-scale')?.value
    || '1/4"=1ft';
  const dpi = appState.imageDpi
    || parseInt(document.getElementById('image-dpi')?.value, 10)
    || 150;

  const pxPerUnit = parseScale(scaleStr, dpi);
  const isMetric  = scaleStr.includes(':');
  const unitLabel = isMetric ? 'm' : 'ft';

  const dx     = (wall.x2_pct - wall.x1_pct) * imgW;
  const dy     = (wall.y2_pct - wall.y1_pct) * imgH;
  const pxLen  = Math.sqrt(dx * dx + dy * dy);
  const realLen = pxPerUnit ? pxLen / pxPerUnit : 0;

  wall.length     = `${realLen.toFixed(2)} ${unitLabel}`;
  wall.length_raw = realLen;

  // Sync the dimension_lines label
  const dl = (result.dimension_lines || []).find(d => d.wallId === wallId);
  if (dl) dl.label = wall.length;

  // Patch the sidebar <div class="wall-dims"> in-place (lookup by data attribute)
  const listEl = document.getElementById('wall-list');
  if (listEl) {
    const item = listEl.querySelector(`[data-wall-id="${wallId}"]`);
    if (item) {
      const dimsEl = item.querySelector('.wall-dims');
      if (dimsEl) dimsEl.textContent = `${wall.facing || '—'} · ${wall.length}`;
    }
  }
}

// ── Room management ──────────────────────────────────────

function findRoomForWall(wallId) {
  return appState.rooms.find(r => r.wallIds.includes(wallId)) || null;
}

/**
 * Directly assign a wall to a room (or unassign if roomId is null).
 * If the wall is already in the target room, clicking again unassigns it (toggle).
 * Called by the room picker popover.
 */
function assignWallToRoom(wallId, roomId) {
  if (!roomId) {
    appState.rooms.forEach(r => {
      const i = r.wallIds.indexOf(wallId);
      if (i >= 0) r.wallIds.splice(i, 1);
    });
  } else {
    const targetRoom = appState.rooms.find(r => r.id === roomId);
    if (!targetRoom) return;
    const alreadyIn = targetRoom.wallIds.includes(wallId);
    appState.rooms.forEach(r => {
      const i = r.wallIds.indexOf(wallId);
      if (i >= 0) r.wallIds.splice(i, 1);
    });
    if (!alreadyIn) targetRoom.wallIds.push(wallId);
  }
  renderRoomsPanel();
  _renderWallList(appState.analysisResult?.walls || []);
  drawCanvas();
}

/**
 * Assign all lasso-selected walls to a room, then clear the selection.
 */
function _assignSelectedWallsToRoom(roomId) {
  const room = appState.rooms.find(r => r.id === roomId);
  if (!room) return;
  appState.selectedWalls.forEach(wallId => {
    appState.rooms.forEach(r => {
      const i = r.wallIds.indexOf(wallId);
      if (i >= 0) r.wallIds.splice(i, 1);
    });
    if (!room.wallIds.includes(wallId)) room.wallIds.push(wallId);
  });
  clearLassoSelection();
  renderRoomsPanel();
  _renderWallList(appState.analysisResult?.walls || []);
  drawCanvas();
}

/**
 * Show the floating lasso-assign bar with the current selection count and room chips.
 */
function _showLassoBar() {
  const bar = document.getElementById('lasso-assign-bar');
  if (!bar) return;
  const count = appState.selectedWalls.size;
  let html = `
    <span class="lab-count">${count} wall${count !== 1 ? 's' : ''} selected</span>
    <span class="lab-label">Assign to:</span>
    <div class="lab-rooms">`;
  appState.rooms.forEach(room => {
    html += `<button class="lab-room-chip"
      style="color:${room.color};border-color:${room.color};background:color-mix(in srgb,${room.color} 12%,transparent)"
      onclick="_assignSelectedWallsToRoom('${room.id}')">${room.name}</button>`;
  });
  if (appState.rooms.length === 0) {
    html += `<span style="font-family:var(--mono);font-size:10px;color:rgba(255,255,255,0.4)">No rooms yet — add one first</span>`;
  }
  html += `</div><button class="lab-clear" title="Clear selection" onclick="clearLassoSelection()">✕</button>`;
  bar.innerHTML = html;
  bar.classList.remove('hidden');
}

/**
 * Clear the lasso wall selection and hide the assign bar.
 */
function clearLassoSelection() {
  appState.selectedWalls = new Set();
  const bar = document.getElementById('lasso-assign-bar');
  if (bar) bar.classList.add('hidden');
  drawCanvas();
}

/**
 * Build room-list HTML for a wall (shared by sidebar picker and canvas action popover).
 */
function _buildRoomPickerListHtml(wallId, onAssignCallback) {
  const currentRoom = findRoomForWall(wallId);
  let listHtml = '';

  if (appState.rooms.length === 0) {
    listHtml += `<div class="rpp-option rpp-add" onclick="addRoom();${onAssignCallback}();">
      <span class="rpp-dot"></span><span>Add a room…</span>
    </div>`;
  } else {
    appState.rooms.forEach(room => {
      const isCurrent = currentRoom && currentRoom.id === room.id;
      listHtml += `<div class="rpp-option${isCurrent ? ' rpp-selected' : ''}"
        style="color:${room.color}"
        onclick="assignWallToRoom('${wallId}','${room.id}');${onAssignCallback}();">
        <span class="rpp-dot"></span>
        <span style="color:var(--text)">${room.name}</span>
        ${isCurrent ? '<span class="rpp-check">✓</span>' : ''}
      </div>`;
    });
    if (currentRoom) {
      listHtml += `<div class="rpp-divider"></div>
      <div class="rpp-option rpp-none"
        onclick="assignWallToRoom('${wallId}',null);${onAssignCallback}();">
        <span class="rpp-dot"></span><span>No room</span>
      </div>`;
    }
    listHtml += `<div class="rpp-divider"></div>
    <div class="rpp-option rpp-add"
      onclick="addRoom();${onAssignCallback}();">
      <span class="rpp-dot"></span><span>Add room…</span>
    </div>`;
  }

  return listHtml;
}

function _positionPopover(popover, pageX, pageY) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const pW = 220;
  const pH = 280;
  let left = pageX + 10;
  let top  = pageY - 6;
  if (left + pW > vw - 8) left = Math.max(8, pageX - pW - 10);
  if (top  + pH > vh - 8) top  = Math.max(8, vh - pH - 8);
  popover.style.left = `${left}px`;
  popover.style.top  = `${top}px`;
}

/**
 * Clear canvas wall focus, resize mode, and close the action popover.
 */
function clearCanvasWallFocus() {
  appState.canvasFocusedWallId = null;
  appState.resizeWallId = null;
  closeWallActionPopover();
  drawCanvas();
}

function closeWallActionPopover() {
  const popover = document.getElementById('room-picker-popover');
  if (popover) popover.classList.add('hidden');
}

/**
 * Open the canvas wall action popover (View A: assign / delete / resize).
 */
function openWallActionPopover(wallId, pageX, pageY) {
  const popover = document.getElementById('room-picker-popover');
  if (!popover) return;

  appState.canvasFocusedWallId = wallId;
  appState.resizeWallId = null;

  const wall = (appState.analysisResult?.walls || []).find(w => w.id === wallId);
  const wallLabel = wall ? (wall.name || wall.id) : wallId;
  const wallDims  = wall ? `${wall.facing || '—'} · ${wall.length || '—'}` : '';

  popover.innerHTML = `
    <div class="wap-header">
      <div class="wap-wall-name">${wallLabel}</div>
      <div class="wap-wall-dims">${wallDims}</div>
    </div>
    <div class="wap-actions">
      <button type="button" class="wap-action" onclick="showRoomPickerInPopover('${wallId}')">
        <span class="wap-icon">⊕</span> Add to room
      </button>
      <button type="button" class="wap-action wap-action-danger" onclick="deleteWallFromCanvas('${wallId}')">
        <span class="wap-icon">✕</span> Delete wall
      </button>
      <button type="button" class="wap-action" onclick="enterWallResizeMode('${wallId}')">
        <span class="wap-icon">↔</span> Resize wall
      </button>
    </div>`;

  popover.classList.remove('hidden');
  appState._popoverPos = { x: pageX, y: pageY };
  _positionPopover(popover, pageX, pageY);
  drawCanvas();
}

/**
 * Swap the action popover to View B: room list.
 */
function showRoomPickerInPopover(wallId) {
  const popover = document.getElementById('room-picker-popover');
  if (!popover) return;

  const pos = appState._popoverPos || { x: popover.offsetLeft, y: popover.offsetTop };
  const listHtml = _buildRoomPickerListHtml(wallId, 'clearCanvasWallFocus');
  popover.innerHTML = `
    <div class="rpp-title">
      <button class="wap-back" type="button"
        onclick="openWallActionPopover('${wallId}', ${pos.x}, ${pos.y})">←</button>
      Assign to room
    </div>
    <div class="rpp-list">${listHtml}</div>`;
}

/**
 * Enter resize mode: show endpoint handles and enable drag.
 */
function enterWallResizeMode(wallId) {
  appState.canvasFocusedWallId = wallId;
  appState.resizeWallId = wallId;
  closeWallActionPopover();
  drawCanvas();
}

/**
 * Delete a wall from the canvas action popover.
 */
function deleteWallFromCanvas(wallId) {
  deleteWall(wallId);
  clearCanvasWallFocus();
}

/**
 * Open the room picker popover anchored near (pageX, pageY).
 * Used by sidebar wall row clicks only.
 */
function renderRoomPickerPopover(wallId, pageX, pageY) {
  const popover = document.getElementById('room-picker-popover');
  if (!popover) return;

  const listHtml = _buildRoomPickerListHtml(wallId, 'closeRoomPickerPopover');

  popover.innerHTML = `<div class="rpp-title">Assign to room</div><div class="rpp-list">${listHtml}</div>`;
  popover.classList.remove('hidden');
  appState._popoverPos = { x: pageX, y: pageY };
  _positionPopover(popover, pageX, pageY);
}

function closeRoomPickerPopover() {
  closeWallActionPopover();
}

function _initPopoverClickGuard() {
  const popover = document.getElementById('room-picker-popover');
  if (popover && !popover._clickGuardAttached) {
    popover._clickGuardAttached = true;
    popover.addEventListener('click', (e) => e.stopPropagation());
  }
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _initPopoverClickGuard);
} else {
  _initPopoverClickGuard();
}

document.addEventListener('click', (e) => {
  const popover = document.getElementById('room-picker-popover');
  const canvas  = document.getElementById('overlay-canvas');
  const popoverOpen = popover && !popover.classList.contains('hidden');

  if (popoverOpen && !popover.contains(e.target)) {
    closeWallActionPopover();
  }

  // Clear canvas focus when clicking outside canvas and popover (unless resize mode)
  if (!appState.canvasFocusedWallId && !appState.resizeWallId) return;
  if (popoverOpen && popover.contains(e.target)) return;
  if (canvas && canvas.contains(e.target)) return;
  if (appState.resizeWallId) {
    // Keep focus highlight during resize; only clear resize on outside click
    appState.resizeWallId = null;
    drawCanvas();
    return;
  }
  clearCanvasWallFocus();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (appState.resizeWallId || appState.canvasFocusedWallId) {
      clearCanvasWallFocus();
    } else {
      closeWallActionPopover();
      closeRoomPickerPopover();
    }
  }
});

/**
 * Extract unique room names from freshly-analysed walls and build/merge
 * appState.rooms. Only called once per analysis pass, not on re-renders.
 */
function buildRoomsFromWalls(walls, cvRooms) {
  const nameToRoom = new Map();

  // Prefer geometric CV room cells when labels are available
  if (cvRooms && cvRooms.length) {
    const labelByCvId = new Map(
      cvRooms.map(r => [r.id, (r.label && String(r.label).trim()) || r.id])
    );
    walls.forEach(wall => {
      if (!wall.room_id) return;
      const name = labelByCvId.get(wall.room_id) || wall.room_id;
      if (!nameToRoom.has(name)) {
        nameToRoom.set(name, {
          id: wall.room_id,
          name,
          wallIds: [],
          color: WALL_STROKES[nameToRoom.size % WALL_STROKES.length],
          cvRoomId: wall.room_id,
        });
      }
      nameToRoom.get(name).wallIds.push(wall.id);
    });
  } else {
    walls.forEach(wall => {
      if (!wall.room) return;
      const name = String(wall.room).trim();
      if (!name) return;
      if (!nameToRoom.has(name)) {
        const existing = appState.rooms.find(r => r.name === name);
        if (existing) {
          existing.wallIds = [];
          nameToRoom.set(name, existing);
        } else {
          nameToRoom.set(name, {
            id: `room-${Date.now()}-${nameToRoom.size}`,
            name,
            wallIds: [],
            color: WALL_STROKES[nameToRoom.size % WALL_STROKES.length],
          });
        }
      }
      nameToRoom.get(name).wallIds.push(wall.id);
    });
  }

  if (nameToRoom.size > 0) {
    const builtRooms = Array.from(nameToRoom.values());
    const builtNames = new Set(builtRooms.map(r => r.name));
    const userRooms  = appState.rooms.filter(r => !builtNames.has(r.name));
    appState.rooms   = [...builtRooms, ...userRooms];
  }
}

function renderRoomsPanel() {
  const panel = document.getElementById('rooms-panel');
  if (!panel) return;

  const rooms = appState.rooms;
  panel.classList.toggle('hidden', rooms.length === 0);
  if (rooms.length === 0) return;

  const activeId = appState.activeRoomId;
  const result   = appState.analysisResult;
  const scaleStr = result?.detected_scale || '1/4"=1ft';
  const unitLabel = scaleStr.includes(':') ? 'm' : 'ft';

  const unassignedCount = (result?.walls || []).filter(w => {
    const assigned = new Set(rooms.flatMap(r => r.wallIds));
    return !assigned.has(w.id);
  }).length;

  let html = `<div class="card-title">ROOMS</div>`;
  html += `<div class="rooms-chips">`;
  if (unassignedCount > 0) {
    const isActive = appState.activeRoomId === '__unassigned__';
    html += `<button class="room-chip room-chip-unassigned${isActive ? ' active' : ''}" onclick="setActiveRoom('__unassigned__')">UNASSIGNED (${unassignedCount})</button>`;
  }
  rooms.forEach(room => {
    const isActive   = room.id === activeId;
    const colorStyle = isActive ? `style="--room-color:${room.color}"` : '';
    html += `<button class="room-chip${isActive ? ' active' : ''}" ${colorStyle} onclick="setActiveRoom('${room.id}')">${room.name}</button>`;
  });
  html += `<button class="room-chip room-chip-add" onclick="addRoom()">+ ADD</button>`;
  html += `</div>`;

  if (activeId) {
    const room = rooms.find(r => r.id === activeId);
    if (room) {
      const wallCount = room.wallIds.length;
      const totalLen  = room.wallIds.reduce((sum, wid) => {
        const w = (result?.walls || []).find(w => w.id === wid);
        return sum + (w?.length_raw || 0);
      }, 0);
      html += `
        <div class="room-detail">
          <span class="room-detail-stats">${wallCount} wall${wallCount !== 1 ? 's' : ''} &middot; ${totalLen.toFixed(1)} ${unitLabel} perimeter</span>
          <div class="room-detail-actions">
            <button class="btn btn-ghost btn-sm" onclick="renameRoom('${room.id}')">Rename</button>
            <button class="btn btn-ghost btn-sm room-delete-btn" onclick="deleteRoom('${room.id}')">Delete</button>
          </div>
        </div>`;
    }
  }

  panel.innerHTML = html;
}

function setActiveRoom(roomId) {
  appState.activeRoomId = appState.activeRoomId === roomId ? null : roomId;
  renderRoomsPanel();
  _renderWallList(appState.analysisResult?.walls || []);
  drawCanvas();
}

function _wallsForActiveFilter(walls) {
  if (appState.activeRoomId !== '__unassigned__') return walls;
  const assigned = new Set(appState.rooms.flatMap(r => r.wallIds));
  return walls.filter(w => !assigned.has(w.id));
}

function addRoom() {
  const name = prompt('Enter room name or number (e.g. 302):');
  if (!name || !name.trim()) return;
  const trimmed = name.trim();
  if (appState.rooms.some(r => r.name === trimmed)) {
    alert(`Room "${trimmed}" already exists.`);
    return;
  }
  const newRoom = {
    id: `room-${Date.now()}`,
    name: trimmed,
    wallIds: [],
    color: WALL_STROKES[appState.rooms.length % WALL_STROKES.length],
  };
  appState.rooms.push(newRoom);
  appState.activeRoomId = newRoom.id;
  renderRoomsPanel();
  _renderWallList(appState.analysisResult?.walls || []);
  drawCanvas();
}

function renameRoom(roomId) {
  const room = appState.rooms.find(r => r.id === roomId);
  if (!room) return;
  const newName = prompt('New room name:', room.name);
  if (!newName || !newName.trim()) return;
  room.name = newName.trim();
  renderRoomsPanel();
  _renderWallList(appState.analysisResult?.walls || []);
}

function deleteRoom(roomId) {
  appState.rooms = appState.rooms.filter(r => r.id !== roomId);
  if (appState.activeRoomId === roomId) appState.activeRoomId = null;
  renderRoomsPanel();
  _renderWallList(appState.analysisResult?.walls || []);
  drawCanvas();
}

function toggleWallRoomAssignment(wallId) {
  const activeId = appState.activeRoomId;
  if (!activeId) return;
  const room = appState.rooms.find(r => r.id === activeId);
  if (!room) return;

  const idx = room.wallIds.indexOf(wallId);
  if (idx >= 0) {
    room.wallIds.splice(idx, 1);
  } else {
    // Remove from any other room first (wall belongs to one room only)
    appState.rooms.forEach(r => {
      if (r.id !== activeId) {
        const i = r.wallIds.indexOf(wallId);
        if (i >= 0) r.wallIds.splice(i, 1);
      }
    });
    room.wallIds.push(wallId);
  }
  renderRoomsPanel();
  _renderWallList(appState.analysisResult?.walls || []);
  drawCanvas();
}

/**
 * Re-render the wall list grouped by room.
 * Called whenever room assignments change (replaces the old badge-only update).
 */
function _renderWallList(allWalls) {
  const listEl = document.getElementById('wall-list');
  if (!listEl) return;
  listEl.innerHTML = '';

  const walls = _wallsForActiveFilter(allWalls);
  const wallIndexMap = new Map(allWalls.map((w, i) => [w.id, i]));
  const assignedIds  = new Set(appState.rooms.flatMap(r => r.wallIds));

  // Update the card-title badge with unassigned count
  const cardTitle = listEl.closest('.card')?.querySelector('.card-title');
  if (cardTitle) {
    const unassignedCount = allWalls.filter(w => !assignedIds.has(w.id)).length;
    const filterNote = appState.activeRoomId === '__unassigned__'
      ? ' <span class="unassigned-badge">filtered</span>'
      : '';
    if (unassignedCount > 0) {
      cardTitle.innerHTML = `WALL MEASUREMENTS <span class="unassigned-badge">${walls.length} shown · ${unassignedCount} unassigned</span>${filterNote}`;
    } else {
      cardTitle.innerHTML = `WALL MEASUREMENTS <span class="unassigned-badge">${walls.length} walls</span>${filterNote}`;
    }
  }

  // Optional flat list when UNASSIGNED chip is active; otherwise show every wall grouped.
  const groups = [];
  if (appState.activeRoomId === '__unassigned__') {
    walls.forEach((wall, i) => _renderWallItem(listEl, wall, i));
    return;
  }

  const unassigned = walls.filter(w => !assignedIds.has(w.id));
  if (unassigned.length > 0) groups.push({ type: 'unassigned', walls: unassigned });
  appState.rooms.forEach(room => {
    const roomWalls = room.wallIds.map(id => walls.find(w => w.id === id)).filter(Boolean);
    if (roomWalls.length > 0) groups.push({ type: 'room', room, walls: roomWalls });
  });

  if (groups.length === 0) {
    walls.forEach((wall, i) => _renderWallItem(listEl, wall, i));
    return;
  }

  groups.forEach(group => {
    const groupEl  = document.createElement('div');
    groupEl.className = 'room-group';

    // Group header
    const headerEl = document.createElement('div');
    const isActiveGroup = group.type === 'room' && group.room.id === appState.activeRoomId;
    if (group.type === 'room') {
      headerEl.className = `room-group-header${isActiveGroup ? ' active-group' : ''}`;
      headerEl.style.setProperty('--group-color', group.room.color);
      headerEl.innerHTML = `
        <span class="room-group-name">${group.room.name}</span>
        <span class="room-group-count">${group.walls.length} wall${group.walls.length !== 1 ? 's' : ''}</span>
      `;
    } else {
      headerEl.className = 'room-group-header room-group-unassigned';
      headerEl.innerHTML = `
        <span class="room-group-name">UNASSIGNED</span>
        <span class="room-group-count">${group.walls.length} wall${group.walls.length !== 1 ? 's' : ''}</span>
      `;
    }
    groupEl.appendChild(headerEl);

    group.walls.forEach(wall => {
      const i = wallIndexMap.get(wall.id) ?? 0;
      _renderWallItem(groupEl, wall, i);
    });

    listEl.appendChild(groupEl);
  });
}

function _renderWallItem(containerEl, wall, colorIdx) {
  const assignedRoom = findRoomForWall(wall.id);
  const isInterior = wall.is_exterior === false;
  const dotColor = assignedRoom
    ? assignedRoom.color
    : (isInterior ? '#4a9eff' : '#ff8c42');
  const roomBadge = assignedRoom
    ? `<span class="wall-room-badge" style="color:${assignedRoom.color};border-color:${assignedRoom.color};background:color-mix(in srgb,${assignedRoom.color} 10%,transparent)">${assignedRoom.name}</span>`
    : `<span class="wall-room-badge wall-room-badge-unassigned">No room</span>`;
  const typeBadge = isInterior
    ? '<span class="wall-type-badge">Interior</span>'
    : (wall.is_exterior ? '<span class="wall-type-badge">Exterior</span>' : '');

  const item = document.createElement('div');
  item.className = 'wall-item' + (assignedRoom ? '' : ' wall-item-unassigned');
  item.dataset.wallId = wall.id;
  item.innerHTML = `
    <div class="wall-dot" style="background:${dotColor};opacity:0.9"></div>
    <div class="wall-info">
      <div class="wall-name">${wall.name || wall.id || 'Wall'}${typeBadge}${roomBadge}</div>
      <div class="wall-dims">${wall.facing || '—'} · ${wall.length || '—'}</div>
    </div>
    <div class="wall-notes">${wall.notes || ''}</div>
    <span class="wall-show-icon" title="Show measurement on plan">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="8" cy="8" rx="7" ry="4.5" stroke="currentColor" stroke-width="1.4"/>
        <circle cx="8" cy="8" r="2.2" fill="currentColor"/>
      </svg>
    </span>
    <button class="wall-delete" title="Remove this wall">×</button>
  `;

  item.querySelector('.wall-delete').addEventListener('click', (e) => {
    e.stopPropagation();
    deleteWall(wall.id);
  });

  // Eye icon: show/hide dimension overlay on canvas (independent of room assignment)
  item.querySelector('.wall-show-icon').addEventListener('click', (e) => {
    e.stopPropagation();
    toggleWallVisibility(wall.id, item);
  });

  if (appState.visibleWalls.has(wall.id)) item.classList.add('highlighted');

  if (assignedRoom) {
    item.classList.add('room-assigned');
    item.style.setProperty('--room-color', assignedRoom.color);
  }

  // Main click: open room picker popover
  item.addEventListener('click', (e) => {
    e.stopPropagation();
    renderRoomPickerPopover(wall.id, e.clientX, e.clientY);
  });

  containerEl.appendChild(item);
}

// ── Render results ──────────────────────────────────────
function renderResults(data) {
  goToStep(4);
  document.getElementById('error-container').classList.add('hidden');

  const walls = data.walls || [];

  document.getElementById('stat-walls').textContent      = walls.length;
  document.getElementById('stat-area').textContent       = data.total_area || '—';
  document.getElementById('stat-scale').textContent      = data.detected_scale || '—';
  document.getElementById('stat-confidence').textContent = (data.scale_confidence || '—').toUpperCase();
  document.getElementById('scale-badge').textContent     = `SCALE: ${data.detected_scale || 'UNKNOWN'}`;

  _renderWallList(walls);

  if (data.notes) {
    document.getElementById('notes-text').textContent = data.notes;
    document.getElementById('notes-container').classList.remove('hidden');
  }

  renderRoomsPanel();
  if (typeof resetViewport === 'function') resetViewport();
  else drawCanvas();
}

