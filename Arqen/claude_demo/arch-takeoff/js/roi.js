/**
 * roi.js — drag a rectangle on the plan preview to mark the building footprint.
 */

function initBuildingRoi() {
  const wrap = document.getElementById('preview-wrap');
  const img = document.getElementById('preview-img');
  const canvas = document.getElementById('roi-canvas');
  if (!wrap || !img || !canvas) return;

  let dragging = false;
  let start = null;

  const syncCanvas = () => {
    const wrapRect = wrap.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();
    canvas.style.left = `${imgRect.left - wrapRect.left}px`;
    canvas.style.top = `${imgRect.top - wrapRect.top}px`;
    canvas.style.width = `${imgRect.width}px`;
    canvas.style.height = `${imgRect.height}px`;
    canvas.width = Math.round(imgRect.width);
    canvas.height = Math.round(imgRect.height);
    drawRoiPreview();
  };

  const displayToImagePct = (clientX, clientY) => {
    const r = img.getBoundingClientRect();
    if (!r.width || !r.height || !img.naturalWidth) return null;
    const x = ((clientX - r.left) / r.width) * img.naturalWidth;
    const y = ((clientY - r.top) / r.height) * img.naturalHeight;
    return {
      x_pct: x / img.naturalWidth,
      y_pct: y / img.naturalHeight,
    };
  };

  const drawRoiPreview = () => {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const roi = appState.buildingRoi;
    if (!roi) return;
    const r = img.getBoundingClientRect();
    const x0 = roi.x0_pct * r.width;
    const y0 = roi.y0_pct * r.height;
    const x1 = roi.x1_pct * r.width;
    const y1 = roi.y1_pct * r.height;
    ctx.strokeStyle = 'rgba(0, 200, 120, 0.95)';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 5]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(0, 200, 120, 0.08)';
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  };

  const onDown = e => {
    if (!img.naturalWidth) return;
    const p = displayToImagePct(e.clientX, e.clientY);
    if (!p) return;
    dragging = true;
    start = p;
    appState.buildingRoi = null;
    updateRoiStatus();
  };

  const onMove = e => {
    if (!dragging || !start) return;
    const p = displayToImagePct(e.clientX, e.clientY);
    if (!p) return;
    appState.buildingRoi = normalizeRoi(start, p);
    drawRoiPreview();
    updateRoiStatus();
  };

  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    const roi = appState.buildingRoi;
    if (roi) {
      const area = (roi.x1_pct - roi.x0_pct) * (roi.y1_pct - roi.y0_pct);
      if (area < 0.02) {
        appState.buildingRoi = null;
      }
    }
    drawRoiPreview();
    updateRoiStatus();
  };

  canvas.onmousedown = onDown;
  canvas.onmousemove = onMove;
  canvas.onmouseup = onUp;
  canvas.onmouseleave = onUp;

  img.addEventListener('load', syncCanvas);
  window.addEventListener('resize', syncCanvas);
  syncCanvas();
}

function normalizeRoi(a, b) {
  return {
    x0_pct: Math.max(0, Math.min(a.x_pct, b.x_pct)),
    y0_pct: Math.max(0, Math.min(a.y_pct, b.y_pct)),
    x1_pct: Math.min(1, Math.max(a.x_pct, b.x_pct)),
    y1_pct: Math.min(1, Math.max(a.y_pct, b.y_pct)),
    method: 'user-roi',
  };
}

function clearBuildingRoi() {
  appState.buildingRoi = null;
  const canvas = document.getElementById('roi-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  updateRoiStatus();
}

function updateRoiStatus() {
  const el      = document.getElementById('roi-status');
  const runBtn  = document.getElementById('btn-run-analysis');
  const clearBtn = document.getElementById('btn-clear-roi');
  const roi     = appState.buildingRoi;

  if (!roi) {
    if (el) {
      el.textContent = 'Draw a box around the floor plan to continue.';
      el.classList.remove('ok');
    }
    if (runBtn) {
      runBtn.disabled = true;
      runBtn.style.opacity = '0.45';
      runBtn.style.cursor = 'not-allowed';
    }
    if (clearBtn) clearBtn.style.display = 'none';
    return;
  }

  const w = ((roi.x1_pct - roi.x0_pct) * 100).toFixed(0);
  const h = ((roi.y1_pct - roi.y0_pct) * 100).toFixed(0);
  if (el) {
    el.textContent = `Region set (${w}% × ${h}% of sheet) — title block and margins excluded.`;
    el.classList.add('ok');
  }
  if (runBtn) {
    runBtn.disabled = false;
    runBtn.style.opacity = '';
    runBtn.style.cursor = '';
  }
  if (clearBtn) clearBtn.style.display = '';
}

function applyUserRoiToResult(parsed) {
  const hint = appState.buildingRoi;
  if (!hint) return parsed;

  parsed._userRoiHint = { ...hint };
  parsed._userRoi = true;

  // Server auto-expands the hint to the detected building envelope; keep that
  // footprint for display and do not re-filter walls by the drawn box.
  if (parsed.analysis_roi_pct) {
    const ar = parsed.analysis_roi_pct;
    parsed.footprint_bbox = {
      x0_pct: ar.x0_pct,
      y0_pct: ar.y0_pct,
      x1_pct: ar.x1_pct,
      y1_pct: ar.y1_pct,
      method: ar.method || 'auto-expanded',
    };
    parsed.footprint_bbox_cv = { ...parsed.footprint_bbox };
    parsed.footprint_polygon_pct = [
      [ar.x0_pct, ar.y0_pct],
      [ar.x1_pct, ar.y0_pct],
      [ar.x1_pct, ar.y1_pct],
      [ar.x0_pct, ar.y1_pct],
    ];
  }

  return parsed;
}

function updateExpandedRoiStatus(parsed) {
  const el = document.getElementById('roi-status');
  if (!el || !parsed?.analysis_roi_pct) return;
  const ar = parsed.analysis_roi_pct;
  const w = ((ar.x1_pct - ar.x0_pct) * 100).toFixed(0);
  const h = ((ar.y1_pct - ar.y0_pct) * 100).toFixed(0);
  el.textContent = `Expanded to detected building footprint (${w}% × ${h}% of sheet).`;
  el.classList.add('ok');
}
