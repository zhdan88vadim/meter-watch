from meter_watch.domain.models import MeterNumber, MeterReading
from meter_watch.domain.services.anomaly_detector import AnomalyDetector


def reading(value: int, conf: float = 0.9, ts: float = 0) -> MeterReading:
    return MeterReading(
        number=MeterNumber(digits=[int(c) for c in str(value)]),
        min_confidence=conf,
        timestamp=ts,
        time_str="",
    )


def test_normal_reading_is_not_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.5)
    result = det.assess(reading(100), previous=reading(99))
    assert not result.is_anomaly
    assert result.reasons == []


def test_low_confidence_is_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.8)
    result = det.assess(reading(100, conf=0.3), previous=None)
    assert result.is_anomaly
    assert "low_confidence" in result.reasons


def test_big_jump_is_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.5, max_jump=10)
    result = det.assess(reading(200), previous=reading(100))
    assert result.is_anomaly
    assert any(r.startswith("big_jump") for r in result.reasons)


def test_decrease_is_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.5)
    result = det.assess(reading(50), previous=reading(100))
    assert result.is_anomaly
    assert "decrease" in result.reasons


def test_sequence_confirmation() -> None:
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    results = [
        det.assess(reading(v, conf=0.3), previous=None)
        for v in [101, 102, 103]
    ]
    last = results[-1]
    assert last.confirmed_sequence is not None
    assert len(last.confirmed_sequence) == 3


def test_sequence_broken_by_decrease() -> None:
    det = AnomalyDetector(min_confidence=0.5, sequence_size=3)
    det.assess(reading(101, conf=0.3), previous=None)
    det.assess(reading(102, conf=0.3), previous=None)
    result = det.assess(reading(50, conf=0.3), previous=None)
    assert result.confirmed_sequence is None


def test_sequence_broken_by_jump() -> None:
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    det.assess(reading(101, conf=0.3), previous=None)
    det.assess(reading(102, conf=0.3), previous=None)
    result = det.assess(reading(200, conf=0.3), previous=None)
    assert result.confirmed_sequence is None
