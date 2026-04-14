/**
 * state.js
 * Central application state object shared across all modules.
 */

const appState = {
  imageFile: null,
  imageDataUrl: null,       // display image (rendered first page for PDFs, original for images)
  fileDataUrl: null,         // raw uploaded file data URL (used for API calls)
  fileType: null,            // 'image' or 'pdf'
  scaleMode: 'auto',
  imageDpi: 300,
  units: 'metric',
  currentStep: 1,
  analysisResult: null,
  layers: { dims: true, labels: true },
  highlightedWall: null,
};

const WALL_STROKES = [
  '#00d4ff','#00e5a0','#f0a500','#a578ff','#ff6496','#50c8ff',
  '#ffb432','#64ffc8','#c864ff','#ff8250','#32dcb4','#ffc864',
];
