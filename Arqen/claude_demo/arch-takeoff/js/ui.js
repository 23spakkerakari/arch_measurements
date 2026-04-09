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
  document.getElementById('manual-scale-field').style.display = m === 'manual' ? 'block' : 'none';
}

function setUnits(u) {
  appState.units = u;
  document.getElementById('radio-metric').classList.toggle('selected', u === 'metric');
  document.getElementById('radio-imperial').classList.toggle('selected', u === 'imperial');
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
}
