"""Pure domain entities and value objects. No I/O, no frameworks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


# ---------- Value objects ----------


@dataclass(frozen=True)
class DigitPrediction:
    """A single digit prediction."""

    value: int
    confidence: float

    def __post_init__(self) -> None:
        if not 0 <= self.value <= 9:
            raise ValueError(f"Digit must be 0-9, got {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass(frozen=True)
class MeterNumber:
    """A number composed of digits, preserves leading zeros."""

    digits: Sequence[int]

    def __post_init__(self) -> None:
        for d in self.digits:
            if not 0 <= d <= 9:
                raise ValueError(f"Invalid digit {d}")

    @property
    def value(self) -> int:
        if not self.digits:
            return 0
        return int("".join(str(d) for d in self.digits))

    @property
    def text(self) -> str:
        return "".join(str(d) for d in self.digits)

    def __len__(self) -> int:
        return len(self.digits)

    def __str__(self) -> str:
        return self.text


@dataclass()
class MeterReading:
    """A recognized meter reading at a point in time."""

    number: MeterNumber
    min_confidence: float
    timestamp: float
    time_str: str
    image: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DigitRecognition:
    """Full recognition output for one digit slot (includes heatmaps)."""

    position: int
    prediction: DigitPrediction
    heatmap_gradcam: str = ""
    heatmap_saliency: str = ""
    digit_image_b64: str = ""


@dataclass(frozen=True)
class RecognitionResult:
    """Aggregate result of recognizing one image."""

    digits: List[DigitRecognition]
    number: MeterNumber
    min_confidence: float

    @property
    def is_confident(self) -> bool:
        return self.min_confidence >= 0.5


# ---------- Anomaly domain ----------


class AnomalyKind:
    """Constants for anomaly reason codes."""

    LOW_CONFIDENCE = "low_confidence"
    BIG_JUMP = "big_jump"
    DECREASE = "decrease"


@dataclass(frozen=True)
class AnomalyAssessment:
    """Result of assessing a reading for anomalies."""

    is_anomaly: bool
    reasons: List[str] = field(default_factory=list)
    confirmed_sequence: Optional[List[MeterReading]] = None
