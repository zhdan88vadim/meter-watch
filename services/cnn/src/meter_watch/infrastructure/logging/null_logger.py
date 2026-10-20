"""No-op logger for tests."""
from __future__ import annotations


class NullLogger:
    """Collects log records in memory; useful for assertions in tests."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, object]]] = []

    def info(self, msg: str, **ctx: object) -> None:
        self.records.append(("info", msg, dict(ctx)))

    def warning(self, msg: str, **ctx: object) -> None:
        self.records.append(("warning", msg, dict(ctx)))

    def error(self, msg: str, **ctx: object) -> None:
        self.records.append(("error", msg, dict(ctx)))
