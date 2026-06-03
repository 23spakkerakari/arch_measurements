/**
 * upload.js
 * Handles file drag-and-drop and file input selection for images and PDFs.
 */

const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
const ACCEPTED_PDF_TYPE    = 'application/pdf';

const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

const PLAN_CROP_PAD = 0.015;
/** Only trim bottom title block — fixed % insets break overlays on varied sheet layouts. */
const PLAN_BOTTOM_TITLE_TRIM = 0.18;

function isInkPixel(r, g, b) {
  return r < 238 || g < 238 || b < 238;
}

async function loadImagePixels(dataUrl) {
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = dataUrl; });
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(img, 0, 0);
  return { data: canvas.getContext('2d').getImageData(0, 0, w, h).data, w, h, img };
}

/** Bounding box of all non-white pixels (fractional 0–1). */
function detectInkBBox(pixels) {
  const { data, w, h } = pixels;
  const step = Math.max(2, Math.floor(Math.min(w, h) / 800));
  let minX = w, minY = h, maxX = 0, maxY = 0;
  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      const i = (y * w + x) * 4;
      if (isInkPixel(data[i], data[i + 1], data[i + 2])) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX <= minX || maxY <= minY) return null;
  return {
    x0: minX / w,
    y0: minY / h,
    x1: (maxX + step) / w,
    y1: (maxY + step) / h,
  };
}

/** Trim title block at bottom only; keep full drawing width for consistent coordinates. */
function trimTitleBlock(bbox) {
  const ih = bbox.y1 - bbox.y0;
  return {
    x0: bbox.x0,
    x1: bbox.x1,
    y0: bbox.y0,
    y1: bbox.y1 - ih * PLAN_BOTTOM_TITLE_TRIM,
    method: 'ink-bottom-trim',
    inkBBox: bbox,
  };
}

/** Crop plan: ink bbox with bottom title trim only (no side insets). */
async function cropPlanToContent(dataUrl) {
  const pixels = await loadImagePixels(dataUrl);
  const ink = detectInkBBox(pixels);
  if (!ink) return { dataUrl, bbox: null, crop: null, skipped: true };

  const bbox = trimTitleBlock(ink);
  const area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
  if (area < 0.08 || area > 0.98) return { dataUrl, bbox: ink, crop: null, skipped: true };

  const { img, w, h } = pixels;
  const padX = Math.round(PLAN_CROP_PAD * w);
  const padY = Math.round(PLAN_CROP_PAD * h);
  const sx = Math.max(0, Math.floor(bbox.x0 * w) - padX);
  const sy = Math.max(0, Math.floor(bbox.y0 * h) - padY);
  const ex = Math.min(w, Math.ceil(bbox.x1 * w) + padX);
  const ey = Math.min(h, Math.ceil(bbox.y1 * h) + padY);
  const sw = ex - sx;
  const sh = ey - sy;
  if (sw < 32 || sh < 32) return { dataUrl, bbox, crop: null, skipped: true };

  const out = document.createElement('canvas');
  out.width = sw;
  out.height = sh;
  out.getContext('2d').drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  return {
    dataUrl: out.toDataURL('image/png'),
    bbox,
    crop: { sx, sy, sw, sh },
    skipped: false,
  };
}

async function applyPlanImage(dataUrl) {
  // Use the full raster as-is so AI coordinates match what is displayed.
  appState.buildingRoi = null;
  appState.imageDataUrl = dataUrl;
  appState.contentBBox  = null;
  appState.planCrop     = null;

  const previewImg = document.getElementById('preview-img');
  previewImg.src = dataUrl;
  document.getElementById('result-img').src = dataUrl;
  await new Promise(res => { previewImg.complete ? res() : (previewImg.onload = res); });
  appState.planImageSize = {
    w: previewImg.naturalWidth,
    h: previewImg.naturalHeight,
  };

  goToStep(2);
  requestAnimationFrame(() => {
    initBuildingRoi();
    updateRoiStatus();
  });
}

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragging');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragging');
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragging');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(f) {
  const isImage = ACCEPTED_IMAGE_TYPES.includes(f.type) || f.type.startsWith('image/');
  const isPdf   = f.type === ACCEPTED_PDF_TYPE || f.name.toLowerCase().endsWith('.pdf');

  if (!isImage && !isPdf) {
    alert('Please upload an image (PNG, JPG, GIF, WEBP) or a PDF file.');
    return;
  }

  appState.imageFile = f;
  appState.fileType  = isPdf ? 'pdf' : 'image';

  // Try to extract DPI from image metadata before loading the full data URL
  if (isImage) {
    const metaReader = new FileReader();
    metaReader.onload = ev => {
      const dpi = extractDpi(new Uint8Array(ev.target.result), f.type);
      if (dpi) {
        appState.imageDpi = dpi;
        document.getElementById('image-dpi').value = dpi;
      }
    };
    // Read only the first 64 KB — enough for EXIF/pHYs metadata
    metaReader.readAsArrayBuffer(f.slice(0, 65536));
  }

  const reader = new FileReader();
  reader.onload = ev => {
    appState.fileDataUrl = ev.target.result;

    if (isPdf) {
      renderPdfFirstPage(ev.target.result);
    } else {
      applyPlanImage(ev.target.result);
    }
  };
  reader.readAsDataURL(f);
}

/**
 * Extracts DPI from image file metadata.
 * - JPEG: scans APP0 (JFIF) and APP1 (EXIF) segments for resolution tags.
 * - PNG:  reads the pHYs chunk (pixels per unit).
 * Returns the detected DPI as an integer, or null if not found / unreliable.
 */
function extractDpi(bytes, mimeType) {
  try {
    if (mimeType === 'image/jpeg' || mimeType === 'image/jpg') {
      return extractJpegDpi(bytes);
    }
    if (mimeType === 'image/png') {
      return extractPngDpi(bytes);
    }
  } catch (e) {
    // Silently ignore parse errors
  }
  return null;
}

function extractJpegDpi(bytes) {
  let i = 2; // skip SOI marker FF D8
  while (i < bytes.length - 4) {
    if (bytes[i] !== 0xFF) break;
    const marker = bytes[i + 1];
    const segLen = (bytes[i + 2] << 8) | bytes[i + 3];

    // APP0 — JFIF
    if (marker === 0xE0 && segLen >= 14) {
      const unit = bytes[i + 11]; // 1 = DPI, 2 = dpcm
      const xDpi = (bytes[i + 12] << 8) | bytes[i + 13];
      if (unit === 1 && xDpi > 0) return xDpi;
      if (unit === 2 && xDpi > 0) return Math.round(xDpi * 2.54);
    }

    // APP1 — EXIF
    if (marker === 0xE1 && segLen > 6) {
      const exifHeader = String.fromCharCode(...bytes.slice(i + 4, i + 10));
      if (exifHeader.startsWith('Exif')) {
        const dpi = parseExifDpi(bytes, i + 10, segLen - 8);
        if (dpi) return dpi;
      }
    }

    i += 2 + segLen;
  }
  return null;
}

function parseExifDpi(bytes, offset, length) {
  if (length < 8) return null;
  const isLE = bytes[offset] === 0x49; // 'II' = little-endian
  const read16 = o => isLE ? (bytes[offset+o] | bytes[offset+o+1]<<8) : (bytes[offset+o]<<8 | bytes[offset+o+1]);
  const read32 = o => isLE ? (bytes[offset+o] | bytes[offset+o+1]<<8 | bytes[offset+o+2]<<16 | bytes[offset+o+3]<<24) : (bytes[offset+o]<<24 | bytes[offset+o+1]<<16 | bytes[offset+o+2]<<8 | bytes[offset+o+3]);

  const ifdOffset = read32(4);
  if (ifdOffset + 2 > length) return null;
  const entryCount = read16(ifdOffset);

  let xRes = null, resUnit = 2; // default unit = inch
  for (let e = 0; e < entryCount; e++) {
    const base = ifdOffset + 2 + e * 12;
    if (base + 12 > length) break;
    const tag = read16(base);
    if (tag === 0x011A) { // XResolution
      const numOffset = read32(base + 8);
      if (numOffset + 8 <= length) {
        const num = read32(numOffset);
        const den = read32(numOffset + 4);
        if (den > 0) xRes = num / den;
      }
    }
    if (tag === 0x0128) resUnit = read16(base + 8); // ResolutionUnit
  }

  if (!xRes || xRes <= 0) return null;
  if (resUnit === 3) return Math.round(xRes * 2.54); // dpcm → dpi
  return Math.round(xRes); // dpi
}

function extractPngDpi(bytes) {
  // PNG signature is 8 bytes, then chunks
  let i = 8;
  while (i < bytes.length - 12) {
    const chunkLen  = (bytes[i]<<24 | bytes[i+1]<<16 | bytes[i+2]<<8 | bytes[i+3]) >>> 0;
    const chunkType = String.fromCharCode(bytes[i+4], bytes[i+5], bytes[i+6], bytes[i+7]);

    if (chunkType === 'pHYs' && chunkLen === 9) {
      const xPpu  = (bytes[i+8]<<24 | bytes[i+9]<<16 | bytes[i+10]<<8 | bytes[i+11]) >>> 0;
      const unit  = bytes[i + 16]; // 1 = metre
      if (unit === 1 && xPpu > 0) {
        return Math.round(xPpu / 39.3701); // pixels/metre → DPI
      }
      return null; // unit 0 = unknown aspect ratio, not useful
    }

    if (chunkType === 'IDAT') break; // metadata chunks always precede image data
    i += 12 + chunkLen;
  }
  return null;
}

/**
 * Renders page 1 of a PDF to an off-screen canvas, converts it to a data URL,
 * and uses that as the preview / result image so the rest of the app works unchanged.
 */
async function renderPdfFirstPage(dataUrl) {
  try {
    const raw    = atob(dataUrl.split(',')[1]);
    const bytes  = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

    const pdf  = await pdfjsLib.getDocument({ data: bytes }).promise;
    const page = await pdf.getPage(1);

    const scale    = 2;
    const rasterDpi = Math.round(72 * scale);
    appState.imageDpi = rasterDpi;
    const dpiInput = document.getElementById('image-dpi');
    if (dpiInput) dpiInput.value = String(rasterDpi);

    const viewport = page.getViewport({ scale });
    const canvas   = document.createElement('canvas');
    canvas.width   = viewport.width;
    canvas.height  = viewport.height;

    await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;

    const imageUrl = canvas.toDataURL('image/png');

    const pageCount = pdf.numPages;
    if (pageCount > 1) {
      console.log(`PDF has ${pageCount} pages — showing page 1 as preview; all pages sent to API.`);
    }

    await applyPlanImage(imageUrl);
  } catch (err) {
    console.error('PDF render error:', err);
    alert('Could not render PDF preview. The file may be corrupted or password-protected.');
  }
}
