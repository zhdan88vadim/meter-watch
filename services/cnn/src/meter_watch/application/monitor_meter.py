"""Use case: continuous meter monitoring."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from meter_watch.application.recognize_image import RecognizeImageUseCase
from meter_watch.domain.errors import ImageFetchError, RecognitionError
from meter_watch.domain.models import AnomalyAssessment, MeterReading
from meter_watch.domain.protocols import (
    Clock,
    ImageFetcher,
    Logger,
    ReadingRepository,
)
from meter_watch.domain.services.anomaly_detector import AnomalyDetector


@dataclass(frozen=True)
class MonitorStatus:
    """Snapshot of the monitor's state."""

    total_readings: int
    last_reading: Optional[MeterReading]
    is_running: bool


class MonitorMeterUseCase:
    """Orchestrates fetch → recognize → assess anomaly → persist.

    Owns the background thread but nothing else. All collaborators
    are injected so tests can drive ``run_cycle`` directly.
    """

    def __init__(
        self,
        fetcher: ImageFetcher,
        recognizer: RecognizeImageUseCase,
        anomaly_detector: AnomalyDetector,
        repository: ReadingRepository,
        clock: Clock,
        logger: Logger,
        poll_interval: float = 20.0,
        max_history: int = 1000,
        max_consecutive_failures: int = 10,
    ) -> None:
        self._fetcher = fetcher
        self._recognizer = recognizer
        self._anomaly = anomaly_detector
        self._repo = repository
        self._clock = clock
        self._logger = logger
        self._poll = poll_interval
        self._max_history = max_history
        self._max_failures = max_consecutive_failures

        self._history: List[MeterReading] = []
        self._last: Optional[MeterReading] = None
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ---------- single cycle (unit-testable) ----------

    def run_cycle(self) -> MeterReading:
        """Fetch, recognize, assess, persist. One shot."""
        image = self._fetcher.fetch()
        if image is None:
            raise ImageFetchError("fetcher returned None")

        result = self._recognizer.execute(image)
        if not result.digits:
            raise RecognitionError("no digits recognized")

        reading = MeterReading(
            number=result.number,
            min_confidence=result.min_confidence,
            timestamp=self._clock.now(),
            time_str=self._clock.now_str(),
            image=image,
        )

        with self._lock:
            return self._process(reading)

    def _process(self, reading: MeterReading) -> MeterReading:
        assessment: AnomalyAssessment = self._anomaly.assess(
            reading, self._last
        )

        if assessment.confirmed_sequence:
            self._logger.info(
                "anomaly sequence confirmed",
                count=len(assessment.confirmed_sequence),
            )
            for confirmed in assessment.confirmed_sequence:
                self._store(confirmed, is_anomaly=False)
            return reading

        if assessment.is_anomaly:
            self._logger.warning(
                "anomaly detected", reasons=assessment.reasons
            )
            self._repo.save(reading, is_anomaly=True)
            return reading

        self._store(reading, is_anomaly=False)
        return reading

    def _store(self, reading: MeterReading, is_anomaly: bool) -> None:
        self._history.append(reading)
        self._last = reading
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        self._repo.save(reading, is_anomaly=is_anomaly)

    # ---------- background loop ----------

    def _run_forever(self) -> None:
        failures = 0
        while self._running:
            try:
                self.run_cycle()
                failures = 0
            except (ImageFetchError, RecognitionError) as exc:
                failures += 1
                self._logger.warning(
                    "temporary failure",
                    error=str(exc),
                    failures=failures,
                )
                if failures >= self._max_failures:
                    self._logger.error("too many failures, resetting")
                    failures = 0
            except Exception as exc:  # noqa: BLE001
                self._logger.error("unexpected error", error=str(exc))
                failures += 1
            time.sleep(self._poll)

    def start(self) -> None:
        """Start the background monitoring thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_forever,
            name="meter-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and join it."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ---------- queries ----------

    def status(self) -> MonitorStatus:
        with self._lock:
            return MonitorStatus(
                total_readings=len(self._history),
                last_reading=self._last,
                is_running=self._running,
            )

    def history(self) -> List[MeterReading]:
        with self._lock:
            return list(self._history)
