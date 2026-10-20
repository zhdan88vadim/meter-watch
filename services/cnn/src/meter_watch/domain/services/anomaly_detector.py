"""Pure anomaly detection logic. No I/O, no state except a sequence buffer."""
from __future__ import annotations

from typing import List, Optional

from meter_watch.domain.models import AnomalyAssessment, MeterReading


class AnomalyDetector:
    """Detects anomalous readings.

    Detects:
      - low confidence
      - big jump
      - decrease

    Consecutive anomalies with small deltas are treated as a "sequence":
    if ``sequence_size`` anomalies in a row each differ by at most
    ``max_sequence_jump`` and are increasing, they are confirmed as real
    readings rather than anomalies.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_jump: int = 10,
        max_sequence_jump: int = 4,
        sequence_size: int = 3,
    ) -> None:
        self._min_confidence = min_confidence
        self._max_jump = max_jump
        self._max_sequence_jump = max_sequence_jump
        self._sequence_size = sequence_size
        self._sequence: List[MeterReading] = []

    def reset(self) -> None:
        """Clear the internal sequence buffer."""
        self._sequence.clear()

    def assess(
        self,
        reading: MeterReading,
        previous: Optional[MeterReading],
    ) -> AnomalyAssessment:
        """Assess a reading against the previous one."""
        reasons = self._reasons_for(reading, previous)
        if not reasons:
            self._sequence.clear()
            return AnomalyAssessment(is_anomaly=False)

        self._sequence = self._extend_sequence(reading)

        if self._is_sequence_confirmed():
            confirmed = list(self._sequence)
            self._sequence.clear()
            return AnomalyAssessment(
                is_anomaly=True,
                reasons=reasons,
                confirmed_sequence=confirmed,
            )

        return AnomalyAssessment(is_anomaly=True, reasons=reasons)

    # ---------- internals ----------

    def _reasons_for(
        self,
        reading: MeterReading,
        previous: Optional[MeterReading],
    ) -> List[str]:
        reasons: List[str] = []

        if reading.min_confidence < self._min_confidence:
            reasons.append("low_confidence")

        if previous is not None:
            delta = reading.number.value - previous.number.value
            if abs(delta) > self._max_jump:
                reasons.append(f"big_jump_{delta}")
            if delta < 0:
                reasons.append("decrease")

        return reasons

    def _extend_sequence(self, reading: MeterReading) -> List[MeterReading]:
        if not self._sequence:
            return [reading]

        last = self._sequence[-1]
        delta = reading.number.value - last.number.value

        if delta < 0 or delta > self._max_sequence_jump:
            return [reading]

        return self._sequence + [reading]

    def _is_sequence_confirmed(self) -> bool:
        if len(self._sequence) < self._sequence_size:
            return False

        for i in range(1, len(self._sequence)):
            delta = (
                self._sequence[i].number.value
                - self._sequence[i - 1].number.value
            )
            if delta < 1 or delta > self._max_sequence_jump:
                return False
        return True
