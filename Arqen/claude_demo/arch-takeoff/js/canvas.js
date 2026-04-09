/**
 * canvas.js
 * Draws dimension lines and labels onto the result canvas.
 */

function drawCanvas() {
  const data = appState.analysisResult;
  if (!data) return;

  const img    = document.getElementById('result-img');
  const canvas = document.getElementById('overlay-canvas');

  const render = () => {
    canvas.width  = img.offsetWidth;
    canvas.height = img.offsetHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const W = canvas.width;
    const H = canvas.height;

    if (appState.layers.dims) drawDimLines(ctx, data.dimension_lines || [], W, H);
  };

  if (img.complete && img.naturalWidth) {
    render();
  } else {
    img.onload = render;
  }

  window.removeEventListener('resize', render);
  window.addEventListener('resize', render);
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
