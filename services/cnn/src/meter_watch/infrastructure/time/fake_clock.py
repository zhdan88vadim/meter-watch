"""Deterministic clock for tests (also usable in production if desired)."""
from __future__ import annotations


class FakeClock:
    """A clock that returns a fixed, controllable time."""

    def __init__(self, t: float = 1000.0) -> None:
        self._t = t

    def now(self) -> float:
        return self._t

    def now_str(self) -> str:
        return "12:00 01:01:2025"

    def advance(self, dt: float) -> None:
        self._t += dt

    def set(self, t: float) -> None:
        self._t = t
