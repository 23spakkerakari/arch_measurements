import math
import numpy as np

'''
Helper file to extract wall segments from a polygon.
'''

def segment_length(seg: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = seg
    return math.hypot(x2 - x1, y2 - y1)


def segment_angle_deg(seg: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = seg
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx)) % 180


#converting our polygon into a list of segments
def polygon_to_segments(polygon: np.ndarray) -> list[tuple[int, int, int, int]]:
    segments = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        segments.append((int(x1), int(y1), int(x2), int(y2)))
    return segments


#filtering out short segments -- making sure we're only working with the longest walls
def filter_short_segments(
    segments: list[tuple[int, int, int, int]],
    min_length_px: float = 15.0
) -> list[tuple[int, int, int, int]]:
    return [seg for seg in segments if segment_length(seg) >= min_length_px]

def filter_non_orthogonal_segments(segments, angle_tolerance_deg: float = 20.0): #outputs a list of segments that either run 0deg or 90deg (with some room for error ofc)
    result = []
    for seg in segments:
        angle = segment_angle_deg(seg)          # in [0, 180)
        dist_to_horiz = min(angle, 180.0 - angle)   # deviation from 0°/180°
        dist_to_vert  = abs(angle - 90.0)            # deviation from 90°
        if min(dist_to_horiz, dist_to_vert) <= angle_tolerance_deg:
            result.append(seg)
    return result


def angle_diff_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)


#for now, a 5 degree angle difference is good enough
#might be useless
def merge_collinear_segments(
    segments: list[tuple[int, int, int, int]],
    angle_threshold_deg: float = 8.0,
    gap_threshold_px: float = 10.0
) -> list[tuple[int, int, int, int]]:
    if not segments:
        return []

    merged = []
    current = segments[0]

    for nxt in segments[1:] + [segments[0]]:
        x1, y1, x2, y2 = current
        nx1, ny1, nx2, ny2 = nxt

        current_angle = segment_angle_deg(current)
        next_angle = segment_angle_deg(nxt)
        endpoint_gap = math.hypot(nx1 - x2, ny1 - y2)

        if (
            angle_diff_deg(current_angle, next_angle) <= angle_threshold_deg
            and endpoint_gap <= gap_threshold_px
        ):
            current = (x1, y1, nx2, ny2)
        else:
            merged.append(current)
            current = nxt

    if len(merged) > 1:
        first = merged[0]
        last = merged[-1]
        if (
            angle_diff_deg(segment_angle_deg(first), segment_angle_deg(last)) <= angle_threshold_deg
            and math.hypot(first[0] - last[2], first[1] - last[3]) <= gap_threshold_px
        ):
            merged[0] = (last[0], last[1], first[2], first[3])
            merged.pop()

    return merged


#puts all the above methods together
def extract_wall_segments(polygon: np.ndarray, min_length_px: float = 15.0, angle_threshold_deg: float = 8.0, gap_threshold_px: float = 10.0) -> list[tuple[int, int, int, int]]:
    raw_segments = polygon_to_segments(polygon)
    filtered = filter_short_segments(raw_segments, min_length_px=min_length_px)
    filtered = filter_non_orthogonal_segments(filtered)
    merged = merge_collinear_segments(
        filtered,
        angle_threshold_deg=angle_threshold_deg,
        gap_threshold_px=gap_threshold_px,
    )
    return merged