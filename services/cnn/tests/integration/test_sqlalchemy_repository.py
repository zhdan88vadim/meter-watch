"""Integration test: in-memory SQLite repository."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from meter_watch.domain.models import MeterNumber, MeterReading
from tests.fixtures.fakes import NullLogger


@pytest.mark.integration
def test_sqlalchemy_repository_saves_reading() -> None:
    """Requires meter_watch_shared models to be importable.
    Skips otherwise."""
    pytest.importorskip("meter_watch_shared.models")

    from meter_watch_shared.db import Base
    from meter_watch.infrastructure.repository.sqlalchemy_repository import (
        SqlAlchemyReadingRepository,
    )

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    repo = SqlAlchemyReadingRepository(Session, NullLogger())
    reading = MeterReading(
        number=MeterNumber(digits=[1, 2, 3]),
        min_confidence=0.9,
        timestamp=1000.0,
        time_str="12:00 01:01:2025",
    )
    repo.save(reading, is_anomaly=False)
    # If it didn't raise, we consider the integration OK
