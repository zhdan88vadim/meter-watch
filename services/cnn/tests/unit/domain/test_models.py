import pytest

from meter_watch.domain.models import MeterNumber, DigitPrediction


def test_meter_number_preserves_leading_zeros() -> None:
    n = MeterNumber(digits=[0, 0, 1, 2, 3])
    assert n.value == 123
    assert n.text == "00123"
    assert len(n) == 5


def test_meter_number_empty() -> None:
    n = MeterNumber(digits=[])
    assert n.value == 0
    assert n.text == ""


def test_meter_number_rejects_invalid_digits() -> None:
    with pytest.raises(ValueError):
        MeterNumber(digits=[10])


def test_digit_prediction_validates_range() -> None:
    with pytest.raises(ValueError):
        DigitPrediction(value=10, confidence=0.5)
    with pytest.raises(ValueError):
        DigitPrediction(value=5, confidence=1.5)
