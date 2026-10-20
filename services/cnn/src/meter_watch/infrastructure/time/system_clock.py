"""System clock adapter."""
from __future__ import annotations

import time


class SystemClock:
    """Wall-clock implementation of the Clock protocol."""

    def now(self) -> float:
        return time.time()

    def now_str(self) -> str:
        return time.strftime("%H:%M %d:%m:%Y", time.localtime())
