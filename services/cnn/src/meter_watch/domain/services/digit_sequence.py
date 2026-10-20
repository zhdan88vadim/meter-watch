"""Number ↔ digit sequence conversions."""
from __future__ import annotations

from meter_watch.domain.models import MeterNumber


def digits_to_number(digits: list[int]) -> MeterNumber:
    """Build a MeterNumber from a list of digits."""
    return MeterNumber(digits=list(digits))


def number_to_digits(value: int) -> MeterNumber:
    """Build a MeterNumber from an integer."""
    if value < 0:
        raise ValueError("value must be non-negative")
    return MeterNumber(digits=[int(c) for c in str(value)])
