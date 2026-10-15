"""Logger backed by Python's stdlib logging."""
from __future__ import annotations

import logging


class StdlibLogger:
    """Structured-ish logger using stdlib logging."""

    def __init__(self, name: str = "meter_watch") -> None:
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            self._log.addHandler(handler)
            self._log.setLevel(logging.INFO)

    def info(self, msg: str, **ctx: object) -> None:
        self._log.info("%s %s", msg, ctx or "")

    def warning(self, msg: str, **ctx: object) -> None:
        self._log.warning("%s %s", msg, ctx or "")

    def error(self, msg: str, **ctx: object) -> None:
        self._log.error("%s %s", msg, ctx or "")
