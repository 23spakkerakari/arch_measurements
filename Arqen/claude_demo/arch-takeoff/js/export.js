/**
 * export.js
 * CSV and annotated-image export functions.
 */

function exportCSV() {
  const data = appState.analysisResult;
  if (!data) return;

  const rows = [
    ['Wall Name', 'Facing', 'Length', 'Notes'],
    ...(data.walls || []).map(w => [
      w.name || w.id, w.facing || '', w.length || '', w.notes || '',
    ]),
    [],
    ['Total Area',         data.total_area],
    ['Scale',              data.detected_scale],
    ['Scale Confidence',   data.scale_confidence],
    ['Units',              data.units],
  ];

  const csv  = rows.map(r => r.map(c => `"${(c || '').replace(/"/g, '""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  triggerDownload(URL.createObjectURL(blob), 'arch-takeoff.csv');
}

function exportImage() {
  const img    = document.getElementById('result-img');
  const canvas = document.getElementById('overlay-canvas');

  const exportCanvas = document.createElement('canvas');
  exportCanvas.width  = img.naturalWidth;
  exportCanvas.height = img.naturalHeight;

  const ctx    = exportCanvas.getContext('2d');
  const scaleX = img.naturalWidth  / canvas.width;
  const scaleY = img.naturalHeight / canvas.height;

  ctx.drawImage(img, 0, 0);
  ctx.scale(scaleX, scaleY);
  ctx.drawImage(canvas, 0, 0);

  triggerDownload(exportCanvas.toDataURL('image/png'), 'arch-takeoff-annotated.png');
}

function triggerDownload(href, filename) {
  const a = document.createElement('a');
  a.href     = href;
  a.download = filename;
  a.click();
}
