import pytest

from meter_watch.domain.errors import ImageFetchError
from meter_watch.domain.models import DigitPrediction


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
