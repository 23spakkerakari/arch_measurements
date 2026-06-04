/**
 * coords.js
 * Maps wall coordinates from footprint-relative space to full-image percentages.
 */

/** @typedef {{ x0_pct: number, y0_pct: number, x1_pct: number, y1_pct: number }} FootprintBBox */

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

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
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  return { data: ctx.getImageData(0, 0, w, h).data, w, h };
}

/**
 * Detect main floor-plan drawing bounds via ink projections (excludes title block band).
 * Returns fractional bbox 0–1 or null.
 */
function detectBuildingFootprintBBox(pixels) {
  const { data, w, h } = pixels;
  const y0 = Math.floor(h * 0.04);
  const y1 = Math.floor(h * 0.74);

  const col = new Uint32Array(w);
  const row = new Uint32Array(h);
  for (let y = y0; y < y1; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      if (isInkPixel(data[i], data[i + 1], data[i + 2])) {
        col[x]++;
        row[y]++;
      }
    }
  }

  const maxCol = Math.max(...col);
  const maxRow = Math.max(...row);
  if (maxCol === 0 || maxRow === 0) return null;

  const colThresh = maxCol * 0.10;
  const rowThresh = maxRow * 0.10;

  let minX = w, maxX = 0, minY = h, maxY = 0;
  for (let x = Math.floor(w * 0.02); x < Math.floor(w * 0.98); x++) {
    if (col[x] >= colThresh) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
    }
  }
  for (let y = y0; y < y1; y++) {
    if (row[y] >= rowThresh) {
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }

  if (maxX <= minX || maxY <= minY) return null;

  const padX = Math.round(w * 0.008);
  const padY = Math.round(h * 0.008);
  return {
    x0_pct: clamp01((minX - padX) / w),
    y0_pct: clamp01((minY - padY) / h),
    x1_pct: clamp01((maxX + padX) / w),
    y1_pct: clamp01((maxY + padY) / h),
    method: 'cv-ink',
  };
}

function footprintFromAreaFields(data) {
  if (data.footprint_bbox &&
      data.footprint_bbox.x0_pct != null &&
      data.footprint_bbox.x1_pct != null) {
    return data.footprint_bbox;
  }
  if (data.area_x1_pct != null && data.area_x2_pct != null) {
    return {
      x0_pct: data.area_x1_pct,
      y0_pct: data.area_y1_pct,
      x1_pct: data.area_x2_pct,
      y1_pct: data.area_y2_pct,
    };
  }
  return null;
}

/** Exterior walls with cardinal facing only (ignore "West Wing" in name). */
function cardinalExteriorWalls(walls) {
  return (walls || []).filter(w => {
    if (!/exterior/i.test(w.name || '')) return false;
    const f = (w.facing || '').toLowerCase();
    return f === 'north' || f === 'south' || f === 'east' || f === 'west';
  });
}

/** Derive building footprint from cardinal exterior wall endpoints. */
function deriveFootprintFromWalls(walls) {
  const ext = cardinalExteriorWalls(walls);
  if (ext.length < 3) return null;

  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  let n = 0;
  for (const w of ext) {
    if (w.x1_pct == null || w.y1_pct == null || w.x2_pct == null || w.y2_pct == null) continue;
    x0 = Math.min(x0, w.x1_pct, w.x2_pct);
    y0 = Math.min(y0, w.y1_pct, w.y2_pct);
    x1 = Math.max(x1, w.x1_pct, w.x2_pct);
    y1 = Math.max(y1, w.y1_pct, w.y2_pct);
    n++;
  }
  if (n < 3 || x1 <= x0 || y1 <= y0) return null;
  const padX = (x1 - x0) * 0.02;
  const padY = (y1 - y0) * 0.02;
  return {
    x0_pct: clamp01(x0 - padX),
    y0_pct: clamp01(y0 - padY),
    x1_pct: clamp01(x1 + padX),
    y1_pct: clamp01(y1 + padY),
  };
}

function footprintSpan(fp) {
  return {
    w: fp.x1_pct - fp.x0_pct,
    h: fp.y1_pct - fp.y0_pct,
  };
}

/**
 * True only when exterior walls use 0–1 footprint space (west≈0) AND footprint is
 * clearly inset from the image — avoids double-mapping image-relative coords.
 */
function exteriorUsesFootprintUnitSquare(walls, fp) {
  const ext = cardinalExteriorWalls(walls);
  if (ext.length < 3) return false;

  const west  = ext.find(w => (w.facing || '').toLowerCase() === 'west');
  const east  = ext.find(w => (w.facing || '').toLowerCase() === 'east');
  const north = ext.find(w => (w.facing || '').toLowerCase() === 'north');
  const south = ext.find(w => (w.facing || '').toLowerCase() === 'south');

  if (west?.x1_pct == null) return false;

  // Already image-relative: west wall sits on footprint left edge in image space
  if (Math.abs(west.x1_pct - fp.x0_pct) < 0.04) return false;

  const westAtOrigin = west.x1_pct < 0.05;
  const eastAtOne    = east?.x1_pct != null && east.x1_pct > 0.95;
  const northAtTop   = north?.y1_pct != null && north.y1_pct < 0.05;
  const southAtBot   = south?.y1_pct != null && south.y1_pct > 0.95;

  const fpInset = fp.x0_pct > 0.08 || fp.y0_pct > 0.08;

  return westAtOrigin && eastAtOne && (northAtTop || southAtBot) && fpInset;
}

/**
 * True when wall endpoints already sit on footprint edges in image space (skip remap).
 */
function wallsAlreadyInImageSpace(walls, fp) {
  if (exteriorUsesFootprintUnitSquare(walls, fp)) return false;

  const ext = cardinalExteriorWalls(walls);
  if (ext.length < 3) return false;

  const west  = ext.find(w => (w.facing || '').toLowerCase() === 'west');
  const east  = ext.find(w => (w.facing || '').toLowerCase() === 'east');
  const north = ext.find(w => (w.facing || '').toLowerCase() === 'north');
  const south = ext.find(w => (w.facing || '').toLowerCase() === 'south');

  const tol = 0.06;
  const westOn  = west?.x1_pct  != null && Math.abs(west.x1_pct  - fp.x0_pct) < tol;
  const eastOn  = east?.x1_pct  != null && Math.abs(east.x1_pct  - fp.x1_pct) < tol;
  const northOn = north?.y1_pct != null && Math.abs(north.y1_pct - fp.y0_pct) < tol;
  const southOn = south?.y1_pct != null && Math.abs(south.y1_pct - fp.y1_pct) < tol;

  return [westOn, eastOn, northOn, southOn].filter(Boolean).length >= 3;
}

/** Map footprint-relative (0–1) wall point to image-relative (0–1). */
function footprintPctToImagePct(xPct, yPct, fp) {
  const { w: fw, h: fh } = footprintSpan(fp);
  return {
    x: clamp01(fp.x0_pct + xPct * fw),
    y: clamp01(fp.y0_pct + yPct * fh),
  };
}

function mapWallToImageSpace(wall, fp, alreadyImageSpace) {
  if (wall.x1_pct == null) return wall;
  if (alreadyImageSpace) return wall;
  const p1 = footprintPctToImagePct(wall.x1_pct, wall.y1_pct, fp);
  const p2 = footprintPctToImagePct(wall.x2_pct, wall.y2_pct, fp);
  return { ...wall, x1_pct: p1.x, y1_pct: p1.y, x2_pct: p2.x, y2_pct: p2.y };
}

/**
 * Normalize analysis JSON so wall x1_pct/y1_pct are always image-relative for canvas draw.
 */
async function normalizeAnalysisCoords(data, imageDataUrl) {
  if (!data) return data;

  const walls = data.walls || [];
  const aiFp = footprintFromAreaFields(data);
  const wallFp = deriveFootprintFromWalls(walls);

  let cvFp = null;
  if (imageDataUrl) {
    try {
      const pixels = await loadImagePixels(imageDataUrl);
      cvFp = detectBuildingFootprintBBox(pixels);
    } catch { /* ignore */ }
  }

  const displayFp = cvFp || wallFp || aiFp;
  const remapFp = aiFp || wallFp;

  const useFootprint = remapFp && exteriorUsesFootprintUnitSquare(walls, remapFp);
  const skipRemap = !useFootprint && (wallsAlreadyInImageSpace(walls, remapFp) || !remapFp);

  const mappedWalls = remapFp
    ? walls.map(w => mapWallToImageSpace(w, remapFp, skipRemap))
    : walls;

  return {
    ...data,
    footprint_bbox: displayFp,
    footprint_bbox_ai: aiFp,
    footprint_bbox_cv: cvFp,
    footprint_bbox_walls: wallFp,
    walls: mappedWalls,
    _coordSpace: useFootprint ? 'footprint-mapped' : (skipRemap ? 'image' : 'unknown'),
    _remapped: useFootprint,
  };
}

/** Validation metrics for debug / UI warnings. */
function validateWallGeometry(data) {
  const walls = data?.walls || [];
  const fp = data?.footprint_bbox;
  const withCoords = walls.filter(w => w.x1_pct != null);
  const exterior = cardinalExteriorWalls(walls);

  let exteriorConsistent = null;
  if (exterior.length >= 4 && fp) {
    const tol = 0.06;
    const west  = exterior.filter(w => (w.facing || '').toLowerCase() === 'west');
    const east  = exterior.filter(w => (w.facing || '').toLowerCase() === 'east');
    const north = exterior.filter(w => (w.facing || '').toLowerCase() === 'north');
    const south = exterior.filter(w => (w.facing || '').toLowerCase() === 'south');

    const westX  = west.length  ? west.every(w  => Math.abs(w.x1_pct - fp.x0_pct) < tol) : null;
    const eastX  = east.length  ? east.every(w  => Math.abs(w.x1_pct - fp.x1_pct) < tol) : null;
    const northY = north.length ? north.every(w => Math.abs(w.y1_pct - fp.y0_pct) < tol) : null;
    const southY = south.length ? south.every(w => Math.abs(w.y1_pct - fp.y1_pct) < tol) : null;
    exteriorConsistent = [westX, eastX, northY, southY].filter(v => v != null).every(Boolean);
  }

  return {
    wallCount: walls.length,
    wallsWithCoords: withCoords.length,
    exteriorCount: exterior.length,
    footprint: fp,
    coordSpace: data._coordSpace,
    exteriorConsistent,
  };
}

/** Convert preprocess.py JSON (px_coords) to web overlay format (x1_pct, …). */
function cvResultToAnalysis(cv) {
  if (cv.error) throw new Error(cv.error);
  const [imgW, imgH] = cv.image_size_px;
  if (!imgW || !imgH) throw new Error('CV result missing image_size_px');

  const walls = (cv.walls || []).map(w => ({
    id: w.id,
    name: w.name,
    facing: w.facing,
    length: w.length,
    length_raw: w.length_raw,
    angle_deg: w.angle_deg,
    px_coords: w.px_coords,
    x1_pct: w.px_coords[0] / imgW,
    y1_pct: w.px_coords[1] / imgH,
    x2_pct: w.px_coords[2] / imgW,
    y2_pct: w.px_coords[3] / imgH,
  }));

  let footprint_bbox = null;
  if (cv.footprint_bbox_px && cv.footprint_bbox_px.length === 4) {
    const [x0, y0, x1, y1] = cv.footprint_bbox_px;
    footprint_bbox = {
      x0_pct: x0 / imgW,
      y0_pct: y0 / imgH,
      x1_pct: x1 / imgW,
      y1_pct: y1 / imgH,
      method: 'cv-footprint-aabb',
    };
  }

  let footprint_polygon_pct = null;
  if (cv.footprint_polygon_px && cv.footprint_polygon_px.length >= 3) {
    footprint_polygon_pct = cv.footprint_polygon_px.map(([x, y]) => [x / imgW, y / imgH]);
  }

  const minSegPx = Math.max(70, Math.min(imgW, imgH) * 0.03);
  const minLenFt = 4;
  // Only exclude pixels right at the image edge (2 %). Python's
  // _filter_wall_segments already removed title-block and header content,
  // so a tighter band here was incorrectly dropping exterior walls whose
  // midpoints fall near the top/bottom of the sheet.
  const inExclusion = (mx, my) => (
    my < 0.02 || my > 0.98 || mx < 0.02 || mx > 0.98
  );
  const dimension_lines = walls.filter(w => {
    const dx = (w.x2_pct - w.x1_pct) * imgW;
    const dy = (w.y2_pct - w.y1_pct) * imgH;
    const mx = (w.x1_pct + w.x2_pct) / 2;
    const my = (w.y1_pct + w.y2_pct) / 2;
    if (inExclusion(mx, my)) return false;
    if ((w.length_raw || 0) < minLenFt) return false;
    return Math.hypot(dx, dy) >= minSegPx;
  }).map(w => ({
    x1_pct: w.x1_pct,
    y1_pct: w.y1_pct,
    x2_pct: w.x2_pct,
    y2_pct: w.y2_pct,
    label: w.length,
    wallId: w.id,
  }));

  return {
    detected_scale: cv.detected_scale,
    scale_confidence: 'high',
    total_area: cv.total_area,
    units: cv.units || 'imperial',
    walls,
    dimension_lines,
    footprint_bbox,
    footprint_bbox_cv: footprint_bbox,
    footprint_polygon_pct,
    image_size_px: cv.image_size_px,
    polygon_vertices: cv.polygon_vertices,
    mask_cache_path: cv.mask_cache_path || null,
    mask_base64: cv.mask_base64 || null,
    mask_roi_offset: cv.mask_roi_offset || [0, 0],
    _coordSpace: 'cv-px',
    _source: 'preprocess',
  };
}

async function runCvAnalyze(imageDataUrl, scaleStr, dpi, roi) {
  const body = { imageBase64: imageDataUrl, scale: scaleStr, dpi };
  if (roi && roi.x0_pct != null) body.roi = roi;
  const res = await fetch(CONFIG.CV_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `CV API error ${res.status}`);
  return cvResultToAnalysis(data);
}

