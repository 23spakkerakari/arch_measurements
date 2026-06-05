/**
 * ui.js
 * Step navigation, logging, and shared UI helpers.
 */

const VIEWS = ['upload', 'config', 'analyzing', 'results'];

function goToStep(n) {
  VIEWS.forEach((view, i) => {
    document.getElementById(`view-${view}`).classList.add('hidden');
    const stepEl = document.getElementById(`step-${i + 1}`);
    stepEl.classList.remove('active', 'done');
    if (i + 1 < n) stepEl.classList.add('done');
    if (i + 1 === n) stepEl.classList.add('active');
  });
  document.getElementById(`view-${VIEWS[n - 1]}`).classList.remove('hidden');
  appState.currentStep = n;

  // Reset draw-region state when returning to upload step
  if (n === 1) {
    appState.analysisRegion = null;
    _rgnActive = false;
    _rgnStart  = null;
    const rgnBtn = document.getElementById('btn-draw-region');
    if (rgnBtn) rgnBtn.classList.remove('active');
    const canvas = document.getElementById('region-canvas');
    if (canvas) {
      canvas.style.pointerEvents = 'none';
      canvas.style.cursor = 'default';
    }
  }
}

// ── Analysis log ───────────────────────────────────────
const LOG_MSGS = [
  'Loading vision model…',
  'Preprocessing plan…',
  'Detecting scale reference…',
  'Extracting wall geometry…',
  'Identifying wall directions…',
  'Computing wall measurements…',
  'Calculating areas…',
  'Calibrating to detected scale…',
  'Generating takeoff data…',
  'Finalizing results…',
];

function addLog(msg) {
  const el = document.getElementById('log-lines');
  const now = new Date();
  const ts = [now.getHours(), now.getMinutes(), now.getSeconds()]
    .map(n => n.toString().padStart(2, '0'))
    .join(':');
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-ts">${ts}</span><span class="log-msg">${msg}</span>`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

// ── Config controls ────────────────────────────────────
function setScaleMode(m) {
  appState.scaleMode = m;
  document.getElementById('radio-auto').classList.toggle('selected', m === 'auto');
  document.getElementById('radio-manual').classList.toggle('selected', m === 'manual');
  const isManual = m === 'manual';
  document.getElementById('manual-scale-field').style.display = isManual ? 'block' : 'none';
  document.getElementById('dpi-field').style.display = isManual ? 'block' : 'none';
}


function toggleLayer(layer, btn) {
  appState.layers[layer] = !appState.layers[layer];
  btn.classList.toggle('active', appState.layers[layer]);
  drawCanvas();
}

function highlightWall(idx, itemEl) {
  document.querySelectorAll('.wall-item').forEach(el => el.classList.remove('highlighted'));
  appState.highlightedWall = appState.highlightedWall === idx ? null : idx;
  if (appState.highlightedWall !== null) itemEl.classList.add('highlighted');
  drawCanvas();
}

// ── Draw Region mode (configure step) ──────────────────
// Lets the user draw a rectangular crop region on the preview image before
// running analysis.  The region is stored as appState.analysisRegion and the
// image is cropped to it before being sent to Claude.

let _rgnStart  = null;
let _rgnActive = false;

function toggleDrawRegion(btn) {
  _rgnActive = !_rgnActive;
  btn.classList.toggle('active', _rgnActive);
  const canvas = document.getElementById('region-canvas');

  if (_rgnActive) {
    // Size canvas to match the preview image NOW (not deferred to mousedown)
    const img = document.getElementById('preview-img');
    if (img.offsetWidth > 0) {
      canvas.width  = img.offsetWidth;
      canvas.height = img.offsetHeight;
    }
    canvas.style.cursor        = 'crosshair';
    canvas.style.pointerEvents = 'auto';
    canvas.addEventListener('mousedown', _onRgnStart);
    canvas.addEventListener('mousemove', _onRgnMove);
    canvas.addEventListener('mouseup',   _onRgnEnd);
  } else {
    canvas.style.cursor        = 'default';
    canvas.style.pointerEvents = 'none';
    canvas.removeEventListener('mousedown', _onRgnStart);
    canvas.removeEventListener('mousemove', _onRgnMove);
    canvas.removeEventListener('mouseup',   _onRgnEnd);
    _rgnStart = null;
  }
}

function clearRegion() {
  appState.analysisRegion = null;
  _rgnStart = null;
  const canvas = document.getElementById('region-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function _rgnPct(e) {
  const canvas = document.getElementById('region-canvas');
  const rect   = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (e.clientX - rect.left)  / rect.width)),
    y: Math.max(0, Math.min(1, (e.clientY - rect.top)   / rect.height)),
  };
}

function _onRgnStart(e) {
  const img    = document.getElementById('preview-img');
  const canvas = document.getElementById('region-canvas');
  // Re-size canvas if needed (ensure it matches displayed image)
  if (canvas.width !== img.offsetWidth || canvas.height !== img.offsetHeight) {
    canvas.width  = img.offsetWidth  || canvas.offsetWidth;
    canvas.height = img.offsetHeight || canvas.offsetHeight;
  }
  _rgnStart = _rgnPct(e);
}

function _onRgnMove(e) {
  if (!_rgnStart) return;
  const cur    = _rgnPct(e);
  const canvas = document.getElementById('region-canvas');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Normalise so width/height are always positive (handles any drag direction)
  const x = Math.min(_rgnStart.x, cur.x) * W;
  const y = Math.min(_rgnStart.y, cur.y) * H;
  const w = Math.abs(cur.x - _rgnStart.x) * W;
  const h = Math.abs(cur.y - _rgnStart.y) * H;

  // Dim outside using evenodd so the interior stays clear
  ctx.fillStyle = 'rgba(0,0,0,0.40)';
  ctx.beginPath();
  ctx.rect(0, 0, W, H);
  ctx.rect(x, y, w, h);
  ctx.fill('evenodd');

  // Dashed cyan border
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth   = 1.5;
  ctx.setLineDash([8, 5]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function _onRgnEnd(e) {
  if (!_rgnStart) return;
  const end = _rgnPct(e);

  const x1 = Math.min(_rgnStart.x, end.x);
  const y1 = Math.min(_rgnStart.y, end.y);
  const x2 = Math.max(_rgnStart.x, end.x);
  const y2 = Math.max(_rgnStart.y, end.y);

  // Ignore tiny accidental clicks
  if ((x2 - x1) < 0.02 || (y2 - y1) < 0.02) {
    _rgnStart = null;
    return;
  }

  appState.analysisRegion = { x1_pct: x1, y1_pct: y1, x2_pct: x2, y2_pct: y2 };
  _rgnStart = null;

  // Redraw finalized region on the canvas
  const canvas = document.getElementById('region-canvas');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const rx = x1 * W, ry = y1 * H, rw = (x2 - x1) * W, rh = (y2 - y1) * H;
  ctx.fillStyle = 'rgba(0,0,0,0.40)';
  ctx.beginPath();
  ctx.rect(0, 0, W, H);
  ctx.rect(rx, ry, rw, rh);
  ctx.fill('evenodd');

  ctx.beginPath();
  ctx.rect(rx, ry, rw, rh);
  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth   = 1.5;
  ctx.setLineDash([8, 5]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Label
  ctx.font = 'bold 9px "Space Mono", monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  const tag = 'ANALYSIS REGION';
  const tw  = ctx.measureText(tag).width + 8;
  ctx.fillStyle = 'rgba(8,12,18,0.85)';
  ctx.fillRect(rx, ry - 16, tw, 16);
  ctx.fillStyle = '#00d4ff';
  ctx.fillText(tag, rx + 4, ry - 2);
}
