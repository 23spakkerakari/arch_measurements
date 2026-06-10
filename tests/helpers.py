"""Shared assertion helpers for the test suite."""

from __future__ import annotations


def assert_no_coaxial_spanning_duplicates(walls: list[dict], px_per_unit: float) -> None:
    """Port of the regression check from Arqen/validate_room_split.py:

    No exterior wall >= 30 ft may be a coaxial spanning duplicate of shorter
    segments tiling the same run (cleanup_wall_list should have dropped it).
    """
    from preprocess import coaxial_spanning_wall_indices, dedup_axis_tol_px

    exterior = [w for w in walls if w.get("is_exterior")]
    long_exterior = [w for w in exterior if w.get("length_raw", 0) >= 30.0]
    if not long_exterior:
        return

    axis_tol = dedup_axis_tol_px(px_per_unit)
    spanning = coaxial_spanning_wall_indices(exterior, axis_tol, cover_frac=0.85)
    offenders = [
        exterior[i]["id"] for i in spanning
        if exterior[i].get("length_raw", 0) >= 30.0
    ]
    assert not offenders, (
        f"coaxial spanning duplicate exterior walls survived cleanup: {offenders}"
    )


def wall_counts(walls: list[dict]) -> tuple[int, int]:
    """(exterior_count, interior_count)"""
    exterior = sum(1 for w in walls if w.get("is_exterior"))
    return exterior, len(walls) - exterior
