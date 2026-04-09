/**
 * analysis.js
 * Sends the plan (image or PDF) to the Claude vision API and renders takeoff results.
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

  const scaleInstruction = scaleMode === 'manual'
    ? `The drawing scale is EXACTLY ${manualScale} — this is a hard constraint provided by the user. You MUST use this scale for every measurement. Do NOT override it, even if the drawing contains a different scale annotation. Set "detected_scale" to "${manualScale}" and "scale_confidence" to "high".`
    : `Auto-detect the scale from any scale bar, north arrow annotation, dimension strings,
       or labeled measurements visible in the drawing.`;

  const detailInstr = detail === 'full'
    ? 'Identify every wall (exterior and interior). For each wall provide its name, facing direction (North/South/East/West), length measurement, and any relevant notes.'
    : detail === 'detailed'
    ? 'Identify all walls including interior partitions. For each wall provide its name, facing direction, and length measurement.'
    : 'Identify the main exterior walls. For each wall provide its name, facing direction, and length measurement.';

  const prompt = buildPrompt(scaleInstruction, detailInstr, units, planType);

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
        .replace(/,\s*([\]}])/g, '$1');   // strip trailing commas
      parsed = JSON.parse(cleaned);
    } catch (e) {
      throw new Error('Could not parse AI response as JSON: ' + e.message);
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

// ── Prompt builder ──────────────────────────────────────
function buildPrompt(scaleInstruction, detailInstr, units, planType) {
  const unitLabel = units === 'metric'
    ? 'metres (m) and square metres (m²)'
    : 'feet (ft) and square feet (ft²)';

  return `You are an expert architectural drawing analyst and quantity surveyor.
Analyze this architectural ${planType === 'auto' ? 'drawing' : planType}.

SCALE INSTRUCTIONS: ${scaleInstruction}

MEASUREMENT UNITS: Return all measurements in ${unitLabel}.

DETECTION SCOPE: ${detailInstr}

CRITICAL: Return ONLY a valid JSON object — no markdown, no explanation, no code
blocks, nothing else. Pure JSON only. Do NOT use trailing commas.

Return exactly this structure:
{
  "detected_scale": "e.g. 1:100 or 1/8 inch = 1 foot",
  "scale_confidence": "high|medium|low",
  "total_area": "e.g. 142.5 m² or 1534 ft²",
  "units": "${units}",
  "walls": [
    {
      "name": "North Wall of Room 445",
      "facing": "North",
      "length": "5.2 m"
    }
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
