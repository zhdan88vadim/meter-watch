"""Tests for AnomalyDetector."""
from __future__ import annotations

import pytest

from meter_watch.domain.models import MeterNumber, MeterReading
from meter_watch.domain.services.anomaly_detector import AnomalyDetector


def reading(value: int, conf: float = 0.9, ts: float = 0) -> MeterReading:
    return MeterReading(
        number=MeterNumber(digits=[int(c) for c in str(value)]),
        min_confidence=conf,
        timestamp=ts,
        time_str="",
    )


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_normal_reading_is_not_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.5)
    result = det.assess(reading(100), previous=reading(99))
    assert not result.is_anomaly
    assert result.reasons == []
    assert result.confirmed_sequence is None


def test_low_confidence_is_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.8)
    result = det.assess(reading(100, conf=0.3), previous=None)
    assert result.is_anomaly
    assert "low_confidence" in result.reasons


def test_confidence_exactly_at_threshold_is_ok() -> None:
    """Boundary: confidence == min_confidence should NOT be an anomaly."""
    det = AnomalyDetector(min_confidence=0.6)
    result = det.assess(reading(100, conf=0.6), previous=None)
    assert not result.is_anomaly


def test_confidence_just_below_threshold_is_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.6)
    result = det.assess(reading(100, conf=0.599), previous=None)
    assert result.is_anomaly
    assert "low_confidence" in result.reasons


# ---------------------------------------------------------------------------
# Big jump
# ---------------------------------------------------------------------------


def test_big_jump_is_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.5, max_jump=10)
    result = det.assess(reading(200), previous=reading(100))
    assert result.is_anomaly
    assert any(r.startswith("big_jump") for r in result.reasons)


def test_jump_exactly_at_limit_is_not_anomaly() -> None:
    """Boundary: abs(delta) == max_jump should NOT be a big_jump."""
    det = AnomalyDetector(min_confidence=0.5, max_jump=10)
    result = det.assess(reading(110), previous=reading(100))
    assert "big_jump_10" not in result.reasons


def test_jump_one_over_limit_is_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.5, max_jump=10)
    result = det.assess(reading(111), previous=reading(100))
    assert any(r.startswith("big_jump") for r in result.reasons)


def test_negative_big_jump_reports_both_reasons() -> None:
    """A large decrease should report both big_jump and decrease."""
    det = AnomalyDetector(min_confidence=0.5, max_jump=10)
    result = det.assess(reading(50), previous=reading(100))
    assert "decrease" in result.reasons
    assert any(r.startswith("big_jump") for r in result.reasons)


def test_first_reading_never_reports_jump() -> None:
    """No previous → no jump or decrease, even if value is huge."""
    det = AnomalyDetector(min_confidence=0.5, max_jump=10)
    result = det.assess(reading(999_999), previous=None)
    assert not result.is_anomaly
    assert result.reasons == []


# ---------------------------------------------------------------------------
# Decrease
# ---------------------------------------------------------------------------


def test_decrease_is_anomaly() -> None:
    det = AnomalyDetector(min_confidence=0.5)
    result = det.assess(reading(50), previous=reading(100))
    assert result.is_anomaly
    assert "decrease" in result.reasons


def test_small_decrease_is_still_anomaly() -> None:
    """Any negative delta is a decrease, even by 1."""
    det = AnomalyDetector(min_confidence=0.5, max_jump=100)
    result = det.assess(reading(99), previous=reading(100))
    assert result.is_anomaly
    assert "decrease" in result.reasons


def test_equal_value_is_not_decrease() -> None:
    det = AnomalyDetector(min_confidence=0.5)
    result = det.assess(reading(100), previous=reading(100))
    assert not result.is_anomaly


def test_increase_within_range_is_ok() -> None:
    det = AnomalyDetector(min_confidence=0.5, max_jump=10)
    result = det.assess(reading(105), previous=reading(100))
    assert not result.is_anomaly


# ---------------------------------------------------------------------------
# Sequence confirmation
# ---------------------------------------------------------------------------

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
    assert [r.number.value for r in last.confirmed_sequence] == [101, 102, 103]


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


def test_sequence_not_confirmed_before_reaching_size() -> None:
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=5
    )
    results = [
        det.assess(reading(v, conf=0.3), previous=None)
        for v in [101, 102, 103, 104]
    ]
    for r in results:
        assert r.confirmed_sequence is None
        assert r.is_anomaly  # still flagged while pending


def test_sequence_with_equal_values_does_not_confirm() -> None:
    """Sequence requires strictly increasing deltas (delta >= 1)."""
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    for v in [100, 100, 100]:
        result = det.assess(reading(v, conf=0.3), previous=None)
    assert result.confirmed_sequence is None


def test_sequence_with_delta_at_max_sequence_jump_confirms() -> None:
    """Boundary: delta == max_sequence_jump should be allowed in sequence."""
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    for v in [100, 104, 108]:
        result = det.assess(reading(v, conf=0.3), previous=None)
    assert result.confirmed_sequence is not None
    assert len(result.confirmed_sequence) == 3


def test_sequence_with_delta_one_over_max_breaks() -> None:
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    det.assess(reading(100, conf=0.3), previous=None)
    det.assess(reading(104, conf=0.3), previous=None)
    result = det.assess(reading(109, conf=0.3), previous=None)  # delta=5
    assert result.confirmed_sequence is None


def test_sequence_buffer_resets_after_confirmation() -> None:
    """After confirming, the buffer should be empty so a new sequence starts fresh."""
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    for v in [101, 102, 103]:
        det.assess(reading(v, conf=0.3), previous=None)

    # Next anomaly should start a fresh sequence, not reuse the old buffer.
    result = det.assess(reading(200, conf=0.3), previous=None)
    assert result.confirmed_sequence is None


def test_sequence_cleared_by_normal_reading() -> None:
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    det.assess(reading(101, conf=0.3), previous=None)
    det.assess(reading(102, conf=0.3), previous=None)
    # A good reading clears the buffer.
    det.assess(reading(103, conf=0.9), previous=reading(102))
    # Now two more anomalies should not confirm.
    det.assess(reading(104, conf=0.3), previous=None)
    result = det.assess(reading(105, conf=0.3), previous=None)
    assert result.confirmed_sequence is None


def test_sequence_confirmation_resets_after_being_reported() -> None:
    """Two consecutive sequences should both confirm independently."""
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    first = None
    for v in [100, 101, 102]:
        first = det.assess(reading(v, conf=0.3), previous=None)
    assert first.confirmed_sequence is not None

    second = None
    for v in [200, 201, 202]:
        second = det.assess(reading(v, conf=0.3), previous=None)
    assert second.confirmed_sequence is not None
    assert [r.number.value for r in second.confirmed_sequence] == [200, 201, 202]


def test_sequence_restarts_on_decrease_then_increases() -> None:
    """[100, 101, 50, 51, 52] — the decrease resets, then a fresh 3-run confirms."""
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    det.assess(reading(100, conf=0.3), previous=None)
    det.assess(reading(101, conf=0.3), previous=None)
    det.assess(reading(50, conf=0.3), previous=None)   # reset
    det.assess(reading(51, conf=0.3), previous=None)
    result = det.assess(reading(52, conf=0.3), previous=None)
    assert result.confirmed_sequence is not None
    assert [r.number.value for r in result.confirmed_sequence] == [50, 51, 52]


# ---------------------------------------------------------------------------
# Interaction between reasons and sequence
# ---------------------------------------------------------------------------


def test_normal_reading_after_anomaly_clears_sequence() -> None:
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    det.assess(reading(100, conf=0.3), previous=None)
    det.assess(reading(101, conf=0.3), previous=None)
    det.assess(reading(102, conf=0.9), previous=reading(101))  # good
    result = det.assess(reading(103, conf=0.9), previous=reading(102))
    assert not result.is_anomaly


def test_is_anomaly_true_even_when_sequence_confirmed() -> None:
    """Documents current behavior: confirmed sequences still have is_anomaly=True.

    This may be surprising — a confirmed sequence is treated as real readings,
    not an anomaly, yet the flag is True. Kept as a regression test.
    """
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    for v in [100, 101, 102]:
        result = det.assess(reading(v, conf=0.3), previous=None)
    assert result.confirmed_sequence is not None
    assert result.is_anomaly is True  # documented quirk


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_pending_sequence() -> None:
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    det.assess(reading(101, conf=0.3), previous=None)
    det.assess(reading(102, conf=0.3), previous=None)
    det.reset()
    result = det.assess(reading(103, conf=0.3), previous=None)
    assert result.confirmed_sequence is None


def test_reset_is_idempotent() -> None:
    det = AnomalyDetector()
    det.reset()
    det.reset()
    result = det.assess(reading(100), previous=None)
    assert not result.is_anomaly


# ---------------------------------------------------------------------------
# Configuration boundaries
# ---------------------------------------------------------------------------


def test_sequence_size_one_confirms_immediately() -> None:
    """sequence_size=1 should confirm on the first anomaly."""
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=1
    )
    result = det.assess(reading(100, conf=0.3), previous=None)
    assert result.confirmed_sequence is not None
    assert len(result.confirmed_sequence) == 1


def test_max_sequence_jump_zero_only_allows_repeats() -> None:
    """With max_sequence_jump=0, only equal values stay in a sequence, but
    _is_sequence_confirmed requires delta >= 1, so it can never confirm."""
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=0, sequence_size=2
    )
    det.assess(reading(100, conf=0.3), previous=None)
    result = det.assess(reading(100, conf=0.3), previous=None)
    assert result.confirmed_sequence is None


def test_zero_value_readings() -> None:
    """A meter reading of 0 should be handled."""
    det = AnomalyDetector(min_confidence=0.5)
    result = det.assess(reading(0), previous=None)
    assert not result.is_anomaly


def test_leading_zeros_preserved_in_sequence() -> None:
    """MeterNumber preserves leading zeros; sequence comparisons use .value."""
    det = AnomalyDetector(
        min_confidence=0.5, max_sequence_jump=4, sequence_size=3
    )
    r1 = MeterReading(
        number=MeterNumber(digits=[0, 0, 1]),
        min_confidence=0.3, timestamp=0, time_str="",
    )
    r2 = MeterReading(
        number=MeterNumber(digits=[0, 0, 2]),
        min_confidence=0.3, timestamp=0, time_str="",
    )
    r3 = MeterReading(
        number=MeterNumber(digits=[0, 0, 3]),
        min_confidence=0.3, timestamp=0, time_str="",
    )
    det.assess(r1, previous=None)
    det.assess(r2, previous=None)
    result = det.assess(r3, previous=None)
    assert result.confirmed_sequence is not None
    assert [r.number.value for r in result.confirmed_sequence] == [1, 2, 3]
