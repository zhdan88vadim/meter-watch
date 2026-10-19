from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from meter_watch.application.monitor_meter import MonitorStatus
from meter_watch.domain.models import MeterReading, RecognitionResult


class Monitor(Protocol):
    def status(self) -> MonitorStatus: ...
    def history(self) -> list[MeterReading]: ...


class ConfigProvider(Protocol):
    def get(self, key: str) -> Any: ...
    def set(self, key: str, value: Any) -> bool: ...
    def update(
        self, updates: dict[str, Any]
    ) -> dict[str, dict[str, Any]]: ...
    def to_dict(self) -> dict[str, Any]: ...
    def has(self, key: str) -> bool: ...


class Recognizer(Protocol):
    def execute(self, image: np.ndarray) -> RecognitionResult: ...
