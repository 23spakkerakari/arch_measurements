/**
 * analysis.js
 * Sends the plan (image or PDF) to the Claude vision API and renders takeoff results.
 *
 * When scale mode is "manual", the prompt asks Claude to return wall endpoints
 * as percentage coordinates. Wall lengths are then computed client-side using
 * the user's DPI and scale, avoiding LLM arithmetic errors.
 */

let logTimer = null;

async function startAnalysis() {
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
  const units       = appState.units;
  const planType    = document.getElementById('plan-type').value;
  const detail      = document.getElementById('detail-level').value;

  if (scaleMode === 'manual') {
    appState.imageDpi = parseInt(document.getElementById('image-dpi').value, 10) || 300;
  }

  const isManual = scaleMode === 'manual';

  const scaleInstruction = isManual
    ? `The drawing scale is EXACTLY ${manualScale}. Do NOT compute wall lengths yourself — return pixel-percentage coordinates instead (see output schema). Set "detected_scale" to "${manualScale}" and "scale_confidence" to "high".`
    : `Auto-detect the scale from any scale bar, north arrow annotation, dimension strings,
       or labeled measurements visible in the drawing.`;

  const detailInstr = detail === 'full'
    ? 'Identify every wall (exterior and interior). For each wall provide its name and facing direction (North/South/East/West).'
    : detail === 'detailed'
    ? 'Identify all walls including interior partitions. For each wall provide its name and facing direction.'
    : 'Identify the main exterior walls. For each wall provide its name and facing direction.';

  const prompt = buildPrompt(scaleInstruction, detailInstr, units, planType, isManual);

  try {
    const sourceUrl = appState.fileDataUrl || appState.imageDataUrl;
    const base64    = sourceUrl.split(',')[1];
    const mimeType  = sourceUrl.split(';')[0].replace('data:', '');
    const isPdf     = appState.fileType === 'pdf';

    const fileBlock = isPdf
      ? { type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: base64 } }
      : { type: 'image',    source: { type: 'base64', media_type: mimeType, data: base64 } };

    const headers = { 'Content-Type': 'application/json' };
    if (CONFIG.ANTHROPIC_API_KEY) {
      headers['x-api-key'] = CONFIG.ANTHROPIC_API_KEY;
      headers['anthropic-version'] = '2023-06-01';
      if (isPdf) headers['anthropic-beta'] = 'pdfs-2024-09-25';
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

    const data    = await response.json();
    const rawText = data.content.map(c => c.text || '').join('').trim();

    let parsed;
    try {
      const jsonMatch = rawText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('No JSON object found in response');
      const cleaned = jsonMatch[0]
        .replace(/,\s*([\]}])/g, '$1');
      parsed = JSON.parse(cleaned);
    } catch (e) {
      throw new Error('Could not parse AI response as JSON: ' + e.message);
    }

    if (isManual) {
      parsed = computeWallLengths(parsed, manualScale, appState.imageDpi, units);
    }

    appState.analysisResult = parsed;
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

// ── Prompt builder ──────────────────────────────────────
function buildPrompt(scaleInstruction, detailInstr, units, planType, isManual) {
  const unitLabel = units === 'metric'
    ? 'metres (m) and square metres (m²)'
    : 'feet (ft) and square feet (ft²)';

  const wallSchema = isManual
    ? `    {
      "name": "North Wall of Room 445",
      "facing": "North",
      "x1_pct": 0.12,
      "y1_pct": 0.05,
      "x2_pct": 0.88,
      "y2_pct": 0.05
    }`
    : `    {
      "name": "North Wall of Room 445",
      "facing": "North",
      "length": "5.2 m"
    }`;

  const coordNote = isManual
    ? `\nIMPORTANT: For each wall, return its start and end points as fractional coordinates
(0.0–1.0) relative to the image dimensions. x increases left-to-right, y increases
top-to-bottom. Do NOT compute wall lengths — just return the coordinates.
Also return "area_x1_pct", "area_y1_pct", "area_x2_pct", "area_y2_pct" as the
bounding box of the overall building footprint.\n`
    : '';

  return `You are an expert architectural drawing analyst and quantity surveyor.
Analyze this architectural ${planType === 'auto' ? 'drawing' : planType}.

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
  "units": "${units}",${isManual ? `
  "area_x1_pct": 0.05,
  "area_y1_pct": 0.03,
  "area_x2_pct": 0.95,
  "area_y2_pct": 0.97,` : ''}
  "walls": [
${wallSchema}
  ]
}
`;
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

  const listEl = document.getElementById('wall-list');
  listEl.innerHTML = '';
  walls.forEach((wall, i) => {
    const color = WALL_STROKES[i % WALL_STROKES.length];
    const item  = document.createElement('div');
    item.className = 'wall-item';
    item.innerHTML = `
      <div class="wall-dot" style="background:${color};opacity:0.9"></div>
      <div class="wall-info">
        <div class="wall-name">${wall.name || wall.id || 'Wall ' + (i + 1)}</div>
        <div class="wall-dims">${wall.facing || '—'} · ${wall.length || '—'}</div>
      </div>
      <div class="wall-notes">${wall.notes || ''}</div>
    `;
    item.addEventListener('click', () => highlightWall(i, item));
    listEl.appendChild(item);
  });

  if (data.notes) {
    document.getElementById('notes-text').textContent = data.notes;
    document.getElementById('notes-container').classList.remove('hidden');
  }

  drawCanvas();
}
