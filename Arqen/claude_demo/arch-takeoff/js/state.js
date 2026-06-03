/**
 * state.js
 * Central application state object shared across all modules.
 */

const appState = {
  imageFile: null,
  imageDataUrl: null,       // display + API image (cropped to drawing content)
  fileDataUrl: null,         // raw uploaded file data URL
  contentBBox: null,         // normalized 0–1 bbox of non-white content in original raster
  planCrop: null,            // pixel crop rect applied before analysis
  planImageSize: null,       // { w, h } natural pixels of image sent to API
  fileType: null,            // 'image' or 'pdf'
  scaleMode: 'auto',
  imageDpi: 300,
  units: 'imperial',
  currentStep: 1,
  analysisResult: null,
  layers: { dims: true, labels: true },
  highlightedWall: null,
  buildingRoi: null,         // user-drawn { x0_pct, y0_pct, x1_pct, y1_pct }
};

const WALL_STROKES = [
  '#00d4ff','#00e5a0','#f0a500','#a578ff','#ff6496','#50c8ff',
  '#ffb432','#64ffc8','#c864ff','#ff8250','#32dcb4','#ffc864',
];
