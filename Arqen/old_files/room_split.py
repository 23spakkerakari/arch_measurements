"""
room_wall_split.py — Subdivide exterior walls at interior-wall T-junctions.

For each exterior wall in out.json, detects where interior walls T-junction
into it and splits the wall into sub-segments, each labeled with a sequential
room_index (1, 2, 3 ...).

Usage
-----
  python room_wall_split.py test.pdf \\
      --json out.json --scale "1in=16ft" [--dpi 500] [--tol 15]

Output
------
  room_walls.json — same structure as out.json.  Walls that were split are
  replaced by their sub-segments; unsplit walls gain room_index = 1.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from non_seg_pipeline.preprocess import (
    _extract_wall_lines,
    find_footprint,
    find_footprint_contour,
    pdf_to_images,
    preprocess,
)
from non_seg_pipeline.scale_parse import parse_scale



def seg_angle_deg(x1, y1, x2, y2):
    """Angle of segment in [0, 180)."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def angle_diff(a, b):
    """Smallest angle between two directions in [0, 180)."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def project_point_onto_segment(px, py, x1, y1, x2, y2):
    """
    Project point (px, py) onto the line through (x1,y1)→(x2,y2).

    Returns
    -------
    t    : float — position along segment; 0 = start, 1 = end (not clamped)
    dist : float — perpendicular distance from point to the infinite line
    """
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return 0.0, math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return t, math.hypot(px - proj_x, py - proj_y)


def point_in_contour(px, py, contour):
    """Return True if (px, py) is inside or on the contour."""
    result = cv2.pointPolygonTest(
        contour.reshape(-1, 1, 2).astype(np.float32),
        (float(px), float(py)),
        measureDist=False,
    )
    return result >= 0


# ── Interior segment detection ────────────────────────────────────────────────

def detect_hough_segments(wall_mask, bbox, margin=30):
    """
    Run HoughLinesP on the wall mask cropped to the floor plan bounding box.
    Returns segments in full-image pixel coordinates.
    """
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(wall_mask.shape[1], x_max + margin)
    y_max = min(wall_mask.shape[0], y_max + margin)

    crop = wall_mask[y_min:y_max, x_min:x_max]

    lines = cv2.HoughLinesP(
        crop,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=15,
        maxLineGap=8,
    )
    if lines is None:
        return []

    # Translate crop-local coords back to full-image coords
    return [
        (
            int(l[0][0] + x_min), int(l[0][1] + y_min),
            int(l[0][2] + x_min), int(l[0][3] + y_min),
        )
        for l in lines
    ]


def find_interior_segments(all_segs, exterior_walls, footprint_contour, near_tol):
    """
    Filter Hough segments to those that are interior walls with at least one
    endpoint T-junctioning into an exterior wall.

    A segment is kept if ALL of the following hold:
      1. Its midpoint lies inside the floor plan footprint.
      2. It is not tracing the exterior boundary (not nearly parallel AND close
         to an existing exterior wall on both endpoints).
      3. At least one endpoint falls within near_tol pixels of an exterior wall.
    """
    interior = []

    for seg in all_segs:
        x1, y1, x2, y2 = seg
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # 1. Midpoint must be inside the footprint
        if not point_in_contour(mx, my, footprint_contour):
            continue

        seg_ang = seg_angle_deg(x1, y1, x2, y2)

        # 2. Reject if this segment is nearly parallel to AND very close to
        #    an exterior wall on both endpoints (i.e., it's tracing the boundary)
        tracing = False
        for ew in exterior_walls:
            ew_ang = seg_angle_deg(*ew)
            if angle_diff(seg_ang, ew_ang) < 15:           # nearly parallel
                _, d1 = project_point_onto_segment(x1, y1, *ew)
                _, d2 = project_point_onto_segment(x2, y2, *ew)
                if d1 < near_tol and d2 < near_tol:        # both endpoints close
                    tracing = True
                    break
        if tracing:
            continue

        # 3. At least one endpoint must be near an exterior wall
        def near_any_exterior(px, py):
            return any(
                project_point_onto_segment(px, py, *ew)[1] < near_tol
                for ew in exterior_walls
            )

        if near_any_exterior(x1, y1) or near_any_exterior(x2, y2):
            interior.append(seg)

    return interior


# ── T-junction splitting ──────────────────────────────────────────────────────

def find_split_t_values(ext_wall, interior_segs, near_tol, min_angle_diff=30,
                        t_margin=0.05, min_segment_px=None):
    """
    For one exterior wall segment, find the t-values (0–1) where interior
    walls T-junction into it.

    Parameters
    ----------
    ext_wall        : (x1,y1,x2,y2) of the exterior wall
    interior_segs   : candidate interior segments
    near_tol        : max perpendicular distance (px) to count as a hit
    min_angle_diff  : interior wall must differ from exterior wall by at least
                      this many degrees (filters out near-parallel glancing hits)
    t_margin        : ignore hits within this fraction of either endpoint
                      (avoids trivial corner splits)
    min_segment_px  : if set, discard any split that would create a sub-segment
                      shorter than this many pixels (noise / wall-thickness hits)

    Returns
    -------
    Sorted, deduplicated list of t-values in (t_margin, 1 - t_margin).
    """
    x1, y1, x2, y2 = ext_wall
    wall_len = math.hypot(x2 - x1, y2 - y1)
    ew_ang = seg_angle_deg(x1, y1, x2, y2)
    raw_ts = []

    for seg in interior_segs:
        ix1, iy1, ix2, iy2 = seg

        # Interior wall must be sufficiently non-parallel to the exterior wall
        seg_ang = seg_angle_deg(ix1, iy1, ix2, iy2)
        if angle_diff(ew_ang, seg_ang) < min_angle_diff:
            continue

        # Check each endpoint of the interior segment
        for px, py in [(ix1, iy1), (ix2, iy2)]:
            t, dist = project_point_onto_segment(px, py, x1, y1, x2, y2)
            if dist < near_tol and t_margin < t < (1.0 - t_margin):
                raw_ts.append(t)

    # Sort then deduplicate: merge any two hits within 5% of each other
    # (the two faces of a double-line interior wall both register)
    raw_ts.sort()
    deduped = []
    for t in raw_ts:
        if not deduped or (t - deduped[-1]) > 0.05:
            deduped.append(t)

    # Drop any split that would produce a sub-segment shorter than min_segment_px.
    # Works by repeatedly removing the split point responsible for the shortest
    # interval until all intervals are long enough.
    if min_segment_px is not None and wall_len > 0:
        min_t = min_segment_px / wall_len
        boundaries = [0.0] + deduped + [1.0]
        changed = True
        while changed:
            changed = False
            intervals = [boundaries[i+1] - boundaries[i]
                         for i in range(len(boundaries) - 1)]
            shortest = min(intervals)
            if shortest < min_t:
                idx = intervals.index(shortest)
                # Remove the interior boundary that created this short interval
                # (prefer removing the one that's less central)
                if idx == 0:
                    del boundaries[1]
                elif idx == len(intervals) - 1:
                    del boundaries[-2]
                else:
                    # Remove whichever neighbour boundary is less "interior"
                    del boundaries[idx + 1]
                changed = True
        deduped = boundaries[1:-1]

    return deduped


def split_wall_entry(wall, t_values, px_per_unit, unit_label):
    """
    Subdivide a wall dict at the given t-values along its length.

    Each sub-segment inherits facing/angle from the parent and gets:
      room_index    — 1-based position along the wall (left→right / top→bottom)
      room_count    — total number of rooms this wall was split into
      parent_wall_id — id of the original unsplit wall
    """
    x1, y1, x2, y2 = wall["px_coords"]
    dx, dy = x2 - x1, y2 - y1
    room_count = len(t_values) + 1

    if not t_values:
        # No T-junctions — return the wall unchanged, just add room fields
        return [{
            **wall,
            "room_index": 1,
            "room_count": 1,
            "parent_wall_id": wall["id"],
        }]

    boundaries = [0.0] + t_values + [1.0]
    sub_walls = []

    for room_idx, (ta, tb) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        sx1 = int(round(x1 + ta * dx))
        sy1 = int(round(y1 + ta * dy))
        sx2 = int(round(x1 + tb * dx))
        sy2 = int(round(y1 + tb * dy))

        px_len = math.hypot(sx2 - sx1, sy2 - sy1)
        real_len = px_len / px_per_unit

        sub_walls.append({
            "id": f"{wall['id']}.r{room_idx}",
            "name": f"{wall['name']} (Room {room_idx}/{room_count})",
            "facing": wall["facing"],
            "length": f"{real_len:.2f} {unit_label}",
            "length_raw": round(real_len, 2),
            "angle_deg": wall["angle_deg"],
            "px_coords": [sx1, sy1, sx2, sy2],
            "room_index": room_idx,
            "room_count": room_count,
            "parent_wall_id": wall["id"],
            "parent_length": wall["length"],
        })

    return sub_walls


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process(pdf_path, json_path, scale_str, dpi, near_tol):
    print("Loading data …", file=sys.stderr)
    with open(json_path) as f:
        data = json.load(f)

    if "walls" not in data:
        raise ValueError(
            f"{json_path} has no top-level 'walls' — expected single-plan output "
            "from preprocess.py. Re-run preprocess.py to regenerate it."
        )

    images = pdf_to_images(str(pdf_path), dpi=dpi)
    image = images[0]

    print("Extracting wall pixels …", file=sys.stderr)
    wall_mask = _extract_wall_lines(image)

    print("Re-detecting floor plan footprint …", file=sys.stderr)
    binary = preprocess(image)
    mask = find_footprint(binary)
    contour = find_footprint_contour(mask)
    if contour is None:
        raise RuntimeError("No building footprint detected in PDF")

    cal = parse_scale(scale_str, dpi, output_unit="ft")
    px_per_unit = cal["px_per_unit"]
    unit_label = cal["unit_label"]

    bbox = data["footprint_bbox_px"]   # [x_min, y_min, x_max, y_max]
    exterior_walls_coords = [w["px_coords"] for w in data["walls"]]

    all_segs = detect_hough_segments(wall_mask, bbox)
    interior_segs = find_interior_segments(
        all_segs, exterior_walls_coords, contour, near_tol=near_tol,
    )

    print(
        f"  {len(exterior_walls_coords)} exterior walls, "
        f"{len(interior_segs)} interior T-junction segments detected",
        file=sys.stderr,
    )

    # Discard splits that would create sub-segments shorter than 2 ft —
    # those are wall-thickness noise hits, not real room boundaries.
    min_segment_px = 2.0 * px_per_unit
    split_walls = []
    n_split_walls = 0
    for wall in data["walls"]:
        t_vals = find_split_t_values(
            wall["px_coords"], interior_segs, near_tol=near_tol,
            min_segment_px=min_segment_px,
        )
        subs = split_wall_entry(wall, t_vals, px_per_unit, unit_label)
        if len(subs) > 1:
            n_split_walls += 1
        split_walls.extend(subs)

    n_new = len(split_walls) - len(exterior_walls_coords)
    print(
        f"  {n_split_walls} walls split → "
        f"{len(split_walls)} total sub-segments ({n_new:+d})",
        file=sys.stderr,
    )

    return {
        **{k: v for k, v in data.items() if k != "walls"},
        "walls": split_walls,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split exterior walls at interior T-junctions and label by room index."
    )
    parser.add_argument("pdf", help="Source PDF (same one used with preprocess.py)")
    parser.add_argument("--json", default="out.json",
                        help="Exterior walls JSON produced by preprocess.py (default: out.json)")
    parser.add_argument("--scale", required=True,
                        help='Drawing scale, e.g. "1in=16ft"')
    parser.add_argument("--dpi", type=int, default=500,
                        help="Rasterization DPI — must match the value used in preprocess.py (default: 500)")
    parser.add_argument("--tol", type=int, default=15,
                        help="Near-miss tolerance in pixels: interior wall endpoint must be "
                             "within this many pixels of the exterior wall line (default: 15)")
    parser.add_argument("--output", default="room_walls.json",
                        help="Output JSON path (default: room_walls.json)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)
    if not Path(args.json).exists():
        print(f"Error: {args.json} not found — run preprocess.py first", file=sys.stderr)
        sys.exit(1)

    result = process(pdf_path, args.json, args.scale, args.dpi, args.tol)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()