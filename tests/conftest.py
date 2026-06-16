"""Pytest bootstrap: puts Arqen/ and validation/ on sys.path, shared fixtures.

No production module is modified; the path insertion below is how the test
suite imports `preprocess`, `room_wall_split`, etc. exactly as the service
does (cv_service.py imports them as same-directory top-level modules).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARQEN_DIR = REPO_ROOT / "Arqen"
VALIDATION_DIR = REPO_ROOT / "validation"

for _p in (str(ARQEN_DIR), str(VALIDATION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def arqen_dir() -> Path:
    return ARQEN_DIR


@pytest.fixture(scope="session")
def two_room_plan():
    from arqen_validation.synth import render_two_room_plan

    return render_two_room_plan()


@pytest.fixture(scope="session")
def l_shape_plan():
    from arqen_validation.synth import render_l_shape_plan

    return render_l_shape_plan()


@pytest.fixture(scope="session")
def corridor_plan():
    from arqen_validation.synth import render_corridor_plan

    return render_corridor_plan()


@pytest.fixture(scope="session")
def symbol_window_plan():
    from arqen_validation.synth import render_symbol_window_plan

    return render_symbol_window_plan()
