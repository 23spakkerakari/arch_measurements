"""Quick validation script for room-split output on a debug capture."""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import (  # noqa: E402
    _perp_band_tol,
    analyze_page,
    coaxial_spanning_wall_indices,
    dedup_axis_tol_px,
)


def assert_no_coaxial_spanning_duplicates(walls, px_per_unit, cover_frac=0.85):
    """Fail if any exterior wall is a spanning duplicate of shorter tiled segments."""
    axis_tol_px = dedup_axis_tol_px(px_per_unit)
    band_tol = _perp_band_tol(px_per_unit, axis_tol_px)
    offenders = coaxial_spanning_wall_indices(
        walls, axis_tol_px, cover_frac, perp_band_tol=band_tol,
    )
    exterior_offenders = [
        i for i in offenders
        if walls[i].get("is_exterior") and walls[i]["length_raw"] >= 30
    ]
    if not exterior_offenders:
        return
    names = [walls[i].get("name", walls[i].get("id")) for i in sorted(exterior_offenders)]
    raise AssertionError(
        f"{len(exterior_offenders)} exterior spanning duplicate(s) remain: {names[:8]}"
        + (" …" if len(names) > 8 else "")
    )


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "debug_runs/20260609-165134"
    )
    req_path = run_dir / "request.json"
    img_path = run_dir / "image.png"
    if not req_path.exists() or not img_path.exists():
        sys.exit(f"Missing capture in {run_dir}")

    with open(req_path) as f:
        req = json.load(f)

    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    result = analyze_page(
        image,
        req["scale"],
        req.get("dpi", 150),
        roi=req.get("roi"),
        doorway_close_ft=2.5,
        room_debug_dir=str(run_dir / "room_debug"),
    )

    if "error" in result:
        print("ERROR:", result["error"])
        sys.exit(1)

    rooms = result.get("rooms", [])
    walls = result.get("walls", [])
    exterior = [w for w in walls if w.get("is_exterior")]
    interior = [w for w in walls if not w.get("is_exterior")]
    px_per_unit = result.get("px_per_ft") or 18.0

    print(f"rooms: {len(rooms)}")
    print(f"walls: {len(walls)} ({len(exterior)} exterior sub-segs, {len(interior)} interior)")

    assert_no_coaxial_spanning_duplicates(walls, px_per_unit)
    print("OK: no coaxial spanning duplicates")

    # North-facing exterior sub-segments (top of building in image-up coords)
    north = [w for w in exterior if w.get("facing") == "North"]
    north_by_parent: dict[str, list] = {}
    for w in north:
        pid = w.get("parent_wall_id", w["id"])
        north_by_parent.setdefault(pid, []).append(w)

    print("\nNorth exterior walls (by parent):")
    for pid, segs in sorted(north_by_parent.items()):
        segs = sorted(segs, key=lambda s: s.get("segment_index", 0))
        print(f"  {pid}: {len(segs)} sub-segments")
        for s in segs:
            print(f"    {s['id']} room={s.get('room_id')} len={s.get('length_raw')}ft")

    out_path = run_dir / "room_split_result.json"
    with open(out_path, "w") as f:
        json.dump({"rooms": rooms, "walls": walls}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
