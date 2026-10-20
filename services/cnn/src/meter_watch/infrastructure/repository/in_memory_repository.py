"""In-memory repository for tests and dev."""
from __future__ import annotations

from meter_watch.domain.models import MeterReading


class InMemoryReadingRepository:
    """Stores readings in a list. Useful for tests."""

    def __init__(self) -> None:
        self.readings: list[tuple[MeterReading, bool]] = []

    def save(self, reading: MeterReading, is_anomaly: bool) -> None:
        self.readings.append((reading, is_anomaly))

    def clear(self) -> None:
        self.readings.clear()

    @property
    def count(self) -> int:
        return len(self.readings)
