"""Reusable fakes for tests."""
from __future__ import annotations

from typing import Optional

import numpy as np

from meter_watch.domain.models import DigitPrediction


class FakeClassifier:
    """Returns a fixed DigitPrediction."""

    def __init__(self, value: int = 5, confidence: float = 0.99) -> None:
        self.value = value
        self.confidence = confidence
        self.calls = 0

    def predict(self, roi: np.ndarray) -> DigitPrediction:
        self.calls += 1
        return DigitPrediction(value=self.value, confidence=self.confidence)


class SequenceClassifier:
    """Returns digits in a repeating sequence — useful for multi-digit tests."""

    def __init__(self, values: list[int], confidence: float = 0.9) -> None:
        self._values = list(values)
        self._i = 0
        self._conf = confidence

    def predict(self, roi: np.ndarray) -> DigitPrediction:
        v = self._values[self._i % len(self._values)]
        self._i += 1
        return DigitPrediction(value=v, confidence=self._conf)


class FakeFetcher:
    """Returns a fixed image, or None if ``fail`` is True."""

    def __init__(self, image: Optional[np.ndarray] = None, fail: bool = False) -> None:
        self._image = image
        self.fail = fail
        self.calls = 0

    def fetch(self) -> Optional[np.ndarray]:
        self.calls += 1
        if self.fail:
            return None
        if self._image is None:
            return np.zeros((60, 200, 3), np.uint8)
        return self._image


class FakeClock:
    """Deterministic clock."""

    def __init__(self, t: float = 1000.0) -> None:
        self._t = t

    def now(self) -> float:
        return self._t

    def now_str(self) -> str:
        return "12:00 01:01:2025"

    def advance(self, dt: float) -> None:
        self._t += dt


class FakeRepo:
    """In-memory repository."""

    def __init__(self) -> None:
        self.saved: list[tuple] = []

    def save(self, reading, is_anomaly: bool) -> None:  # noqa: ANN001
        self.saved.append((reading, is_anomaly))


class NullLogger:
    """Logger that records messages in memory."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def info(self, msg: str, **ctx: object) -> None:
        self.records.append(("info", msg, dict(ctx)))

    def warning(self, msg: str, **ctx: object) -> None:
        self.records.append(("warning", msg, dict(ctx)))

    def error(self, msg: str, **ctx: object) -> None:
        self.records.append(("error", msg, dict(ctx)))


class NullAnnotator:
    """Annotator that returns empty strings."""

    def annotate(self, roi, prediction):  # noqa: ANN001
        return {"heatmap_gradcam": "", "heatmap_saliency": "", "digit_image": ""}
