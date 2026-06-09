"""Verify wall facing (N/S/E/W) at 3/8 and 1/8 scales."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ARQEN = Path(__file__).resolve().parents[1]
CAPTURE = ARQEN / "debug_runs" / "20260609-153430"
REQ = json.loads((CAPTURE / "request.json").read_text())
roi = REQ["roi"]
roi_str = f"{roi['x0_pct']},{roi['y0_pct']},{roi['x1_pct']},{roi['y1_pct']}"
script = ARQEN / "claude_demo" / "arch-takeoff" / "scripts" / "cv_analyze.py"


def _crop_probe_to_full(box, roi_dict):
    """Map a probe defined in ROI-crop fractions to full-image fractions."""
    x0, y0, x1, y1 = box
    rw = roi_dict["x1_pct"] - roi_dict["x0_pct"]
    rh = roi_dict["y1_pct"] - roi_dict["y0_pct"]
    return (
        roi_dict["x0_pct"] + x0 * rw,
        roi_dict["y0_pct"] + y0 * rh,
        roi_dict["x0_pct"] + x1 * rw,
        roi_dict["y1_pct"] + y1 * rh,
    )


probes = {
    "topright": (0.68, 0.05, 0.92, 0.20),
    "west": _crop_probe_to_full((0.0, 0.10, 0.06, 0.90), roi),
    "east": _crop_probe_to_full((0.94, 0.10, 1.0, 0.90), roi),
    "bottomleft": (0.14, 0.82, 0.48, 0.96),
    "bottomctr": (0.48, 0.80, 0.78, 0.96),
}


def in_box(seg, box, img_w, img_h, orientation=None):
    x0, y0, x1, y1 = (
        int(box[0] * img_w), int(box[1] * img_h),
        int(box[2] * img_w), int(box[3] * img_h),
    )
    sx1, sy1, sx2, sy2 = seg
    is_horiz = abs(sx2 - sx1) >= abs(sy2 - sy1)
    if orientation == "horiz" and not is_horiz:
        return False
    if orientation == "vert" and is_horiz:
        return False
    if is_horiz:
        if not (y0 <= (sy1 + sy2) / 2 <= y1):
            return False
        lo, hi = max(min(sx1, sx2), x0), min(max(sx1, sx2), x1)
        return hi - lo >= 40
    if not (x0 <= (sx1 + sx2) / 2 <= x1):
        return False
    lo, hi = max(min(sy1, sy2), y0), min(max(sy1, sy2), y1)
    return hi - lo >= 40


def dominant_facing(walls):
    if not walls:
        return None
    return Counter(w["facing"] for w in walls).most_common(1)[0][0]


failed = False
for scale in ['3/8"=1ft', "1/8in=1ft"]:
    proc = subprocess.run(
        [sys.executable, str(script), "--image", str(CAPTURE / "image.png"),
         "--scale", scale, "--dpi", str(REQ["dpi"]), "--roi", roi_str],
        capture_output=True, text=True, cwd=str(ARQEN),
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        failed = True
        continue
    d = json.loads(proc.stdout)
    walls = d.get("walls", [])
    err = d.get("error")
    img_w, img_h = d.get("image_size_px", [1, 1])
    counts = Counter(w.get("facing") for w in walls)
    print(f"\n=== {scale} === err={err} walls={len(walls)} facings={dict(counts)}")

    for cardinal in ("North", "South", "East", "West"):
        if counts.get(cardinal, 0) == 0:
            print(f"  WARN: no {cardinal} walls")
            failed = True

    dedup = [line for line in proc.stderr.splitlines() if "dedup:" in line]
    if dedup:
        print(f"  {dedup[-1]}")

    expectations = {
        "topright": ("North", "horiz"),
        "west": ("West", "vert"),
        "east": ("East", "vert"),
        "bottomleft": ("South", "horiz"),
        "bottomctr": ("South", "horiz"),
    }
    for name, box in probes.items():
        exp_facing, orient = expectations[name]
        hits = [
            w for w in walls
            if in_box(w["px_coords"], box, img_w, img_h, orientation=orient)
        ]
        dom = dominant_facing(hits)
        ok = dom == exp_facing if hits else False
        status = "OK" if ok else "FAIL"
        if hits and not ok:
            failed = True
        print(f"  {name}: {len(hits)} hits dom={dom} expect={exp_facing} [{status}]")

if failed:
    sys.exit(1)
print("\nAll facing checks passed.")
