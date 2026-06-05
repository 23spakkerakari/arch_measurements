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
  layers: { dims: true, labels: true, mask: false },
  visibleWalls: new Set(),   // wall IDs currently pinned/visible on the canvas
  buildingRoi: null,         // user-drawn { x0_pct, y0_pct, x1_pct, y1_pct }
  drawWallMode: false,       // true when user is drawing a wall by clicking two endpoints
  drawWallFirstPoint: null,  // { x, y } in pct — first endpoint while drawing
  _drawCursor: null,         // { x, y } current cursor pct — rubber-band preview
  maskImage: null,           // loaded Image of the wall_pair_mask PNG (for CV mask overlay)
  _loadedMaskPath: null,     // path of the currently loaded mask (cache key)
  // Endpoint drag state — set while the user is dragging a wall endpoint to adjust length.
  // { wallId, endpointIdx (0=start,1=end), anchorXPct, anchorYPct, unitX, unitY }
  dragState: null,
  hoveredEndpoint: null,     // { wallId, endpointIdx } — endpoint under cursor (hover only)
  rooms: [],                 // [{ id, name, wallIds: [], color }]
  activeRoomId: null,        // room currently open for wall-assignment mode
};

const WALL_STROKES = [
  '#00d4ff','#00e5a0','#f0a500','#a578ff','#ff6496','#50c8ff',
  '#ffb432','#64ffc8','#c864ff','#ff8250','#32dcb4','#ffc864',
];
