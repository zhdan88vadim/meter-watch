"""Interfaces the domain and application layers depend on.

Implementations live in infrastructure/.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from meter_watch.domain.models import DigitPrediction, MeterReading


@runtime_checkable
class DigitClassifier(Protocol):
    """Classify a single digit ROI into a DigitPrediction."""

    def predict(self, roi: np.ndarray) -> DigitPrediction:
        ...


@runtime_checkable
class ImageFetcher(Protocol):
    """Fetch an image from a source."""

    def fetch(self) -> Optional[np.ndarray]:
        ...


@runtime_checkable
class ReadingRepository(Protocol):
    """Persist meter readings."""

    def save(self, reading: MeterReading, is_anomaly: bool) -> None:
        ...


@runtime_checkable
class Clock(Protocol):
    """Injectable clock for deterministic tests."""

    def now(self) -> float:
        ...

    def now_str(self) -> str:
        ...


@runtime_checkable
class Logger(Protocol):
    """Structured logger."""

    def info(self, msg: str, **ctx: object) -> None:
        ...

    def warning(self, msg: str, **ctx: object) -> None:
        ...

    def error(self, msg: str, **ctx: object) -> None:
        ...


@runtime_checkable
class ImageAnnotator(Protocol):
    """Render heatmaps / digit crops to base64 strings."""

    def annotate(self, roi: np.ndarray, prediction: DigitPrediction) -> dict[str, str]:
        ...
