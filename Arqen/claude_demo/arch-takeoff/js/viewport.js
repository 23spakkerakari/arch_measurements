/**
 * viewport.js
 * AutoCAD-style scroll-wheel zoom and middle-mouse pan for the results plan view.
 */

const _viewport = {
  container: null,
  stage: null,
  img: null,
  fitScale: 1,
  userZoom: 1,
  panX: 0,
  panY: 0,
  minUserZoom: 0.25,
  maxUserZoom: 8,
  panning: false,
  panStartX: 0,
  panStartY: 0,
  panOriginX: 0,
  panOriginY: 0,
  initialized: false,
  resizeObserver: null,
};

function _clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function _isInteractiveChrome(target) {
  return target.closest('.canvas-toolbar, .lasso-assign-bar, .scale-badge, button, a, input, select');
}

function _applyTransform() {
  const { stage, panX, panY, userZoom } = _viewport;
  if (!stage) return;
  stage.style.transform = `translate(${panX}px, ${panY}px) scale(${userZoom})`;
  stage.style.transformOrigin = '0 0';
}

function _computeFitScale() {
  const { container, img } = _viewport;
  if (!container || !img || !img.naturalWidth || !img.naturalHeight) return 1;

  const cw = container.clientWidth;
  const ch = container.clientHeight;
  if (cw <= 0 || ch <= 0) return 1;

  return Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
}

function _sizeStage() {
  const { stage, img, fitScale } = _viewport;
  if (!stage || !img || !img.naturalWidth) return;

  const w = Math.round(img.naturalWidth * fitScale);
  const h = Math.round(img.naturalHeight * fitScale);
  stage.style.width = `${w}px`;
  stage.style.height = `${h}px`;
  img.style.width = `${w}px`;
  img.style.height = `${h}px`;
}

function refitViewport({ resetZoom = true } = {}) {
  const { img } = _viewport;
  if (!img || !img.naturalWidth) return;

  _viewport.fitScale = _computeFitScale();
  _sizeStage();

  if (resetZoom) {
    _viewport.userZoom = 1;
    _viewport.panX = 0;
    _viewport.panY = 0;
  }

  _applyTransform();
  if (typeof drawCanvas === 'function') drawCanvas();
}

function resetViewport() {
  refitViewport({ resetZoom: true });
}

function _onWheel(e) {
  if (_isInteractiveChrome(e.target)) return;

  const { container } = _viewport;
  if (!container) return;

  e.preventDefault();

  const rect = container.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const { userZoom, panX, panY, minUserZoom, maxUserZoom } = _viewport;
  const factor = e.deltaY < 0 ? 1.1 : 0.9;

  const localX = (mx - panX) / userZoom;
  const localY = (my - panY) / userZoom;

  const newZoom = _clamp(userZoom * factor, minUserZoom, maxUserZoom);
  _viewport.userZoom = newZoom;
  _viewport.panX = mx - localX * newZoom;
  _viewport.panY = my - localY * newZoom;

  _applyTransform();
}

function _onMouseDown(e) {
  if (e.button !== 1) return;
  if (_isInteractiveChrome(e.target)) return;

  e.preventDefault();

  const { container } = _viewport;
  _viewport.panning = true;
  _viewport.panStartX = e.clientX;
  _viewport.panStartY = e.clientY;
  _viewport.panOriginX = _viewport.panX;
  _viewport.panOriginY = _viewport.panY;
  if (container) container.classList.add('panning');
}

function _onMouseMove(e) {
  if (!_viewport.panning) return;

  const dx = e.clientX - _viewport.panStartX;
  const dy = e.clientY - _viewport.panStartY;
  _viewport.panX = _viewport.panOriginX + dx;
  _viewport.panY = _viewport.panOriginY + dy;
  _applyTransform();
}

function _endPan() {
  if (!_viewport.panning) return;
  _viewport.panning = false;
  if (_viewport.container) _viewport.container.classList.remove('panning');
}

function _onAuxClick(e) {
  if (e.button === 1) e.preventDefault();
}

function initResultViewport() {
  if (_viewport.initialized) return;

  const container = document.getElementById('canvas-container');
  const stage = document.getElementById('result-plan-stage');
  const img = document.getElementById('result-img');
  if (!container || !stage || !img) return;

  _viewport.container = container;
  _viewport.stage = stage;
  _viewport.img = img;
  _viewport.initialized = true;

  container.addEventListener('wheel', _onWheel, { passive: false });
  container.addEventListener('mousedown', _onMouseDown);
  container.addEventListener('auxclick', _onAuxClick);
  document.addEventListener('mousemove', _onMouseMove);
  document.addEventListener('mouseup', _endPan);

  img.addEventListener('load', () => refitViewport({ resetZoom: true }));

  if (typeof ResizeObserver !== 'undefined') {
    _viewport.resizeObserver = new ResizeObserver(() => {
      refitViewport({ resetZoom: true });
    });
    _viewport.resizeObserver.observe(container);
  } else {
    window.addEventListener('resize', () => refitViewport({ resetZoom: true }));
  }

  if (img.complete && img.naturalWidth) {
    refitViewport({ resetZoom: true });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initResultViewport);
} else {
  initResultViewport();
}
