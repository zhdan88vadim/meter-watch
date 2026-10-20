import pytest

from meter_watch.domain.errors import ImageFetchError
from meter_watch.domain.models import DigitPrediction
from meter_watch.application.monitor_meter import MonitorMeterUseCase
from meter_watch.domain.models import MeterNumber, MeterReading
from meter_watch.domain.services.anomaly_detector import AnomalyDetector
from tests.fixtures.fakes import FakeFetcher

def build_monitor(recognizer, repo, clock, logger, anomaly_detector_sequence_size) -> MonitorMeterUseCase:
    return MonitorMeterUseCase(
        fetcher=FakeFetcher(),
        recognizer=recognizer,
        anomaly_detector=AnomalyDetector(min_confidence=0.5, sequence_size=anomaly_detector_sequence_size),
        repository=repo,
        clock=clock,
        logger=logger,
        poll_interval=0.01,
    )

def reading(value: int, conf: float = 0.9, ts: float = 0) -> MeterReading:
    return MeterReading(
        number=MeterNumber(digits=[int(c) for c in str(value)]),
        min_confidence=conf,
        timestamp=ts,
        time_str="",
    )

@pytest.mark.parametrize(
        "anomaly_detector_sequence_size, expected_anomalies",
        [
            (3, 2),
            (4, 3),
            (5, 4),
        ]
)
def test_anomaly_list(recognizer, repo, clock, logger, anomaly_detector_sequence_size, expected_anomalies) -> None:
    monitor = build_monitor(
        recognizer, 
        repo, 
        clock, 
        logger, 
        anomaly_detector_sequence_size=anomaly_detector_sequence_size
    )

    values = [101, 102, 103, 10, 12, 13, 14, 15, 17, 19]

    for v in values:
        monitor._process(reading(v, conf=0.8))

    saved_anomaly_items = [item for item in repo.saved if item[1] == 1]
    
    assert len(repo.saved) - expected_anomalies == len(values)
    assert len(saved_anomaly_items) == expected_anomalies

def test_run_cycle_persists_reading(monitor, repo) -> None:
    monitor._recognizer._classifier = _StubClassifier(DigitPrediction(5, 0.9))
    reading = monitor.run_cycle()
    assert reading.number.value == 5
    assert len(repo.saved) == 1
    assert repo.saved[0][1] is False


def test_run_cycle_raises_on_fetch_failure(monitor) -> None:
    monitor._fetcher.fail = True
    with pytest.raises(ImageFetchError):
        monitor.run_cycle()


def test_status_reports_counts(monitor) -> None:
    s = monitor.status()
    assert s.total_readings == 0
    assert s.is_running is False


class _StubClassifier:
    def __init__(self, pred):  # noqa: ANN001
        self._pred = pred

    def predict(self, roi):  # noqa: ANN001
        return self._pred
