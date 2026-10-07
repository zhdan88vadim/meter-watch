from __future__ import annotations

from pathlib import Path

import pytest


def _first_existing(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.is_file():
            return c
    return None


@pytest.fixture(scope="session")
def person_image_path() -> Path:
    p = _first_existing([
        Path("tests/assets/person.png"),
    ])
    if p is None:
        pytest.skip(
            "No person image found. Place a file at tests/assets/person.png "
            "or tests/assets/person2.png"
        )
    print(f"\n[conftest] person_image_path = {p.resolve()}")
    return p


@pytest.fixture(scope="session")
def empty_image_path() -> Path:
    p = _first_existing([
        Path("tests/assets/room.png"),
    ])
    if p is None:
        pytest.skip(
            "No empty image found. Place a file at tests/assets/room.png"
        )
    print(f"\n[conftest] empty_image_path = {p.resolve()}")
    return p