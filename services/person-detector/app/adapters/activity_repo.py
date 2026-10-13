from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy.orm import sessionmaker

from meter_watch_shared.models import ActivityLog, EventTypeEnum, SourceEnum


class SqlActivityRepository:
    """Concrete repository backed by SessionLocal from meter_watch_shared."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def log_person_detected(self, person_data: dict) -> None:
        self._write(EventTypeEnum.PERSON_DETECTED, person_data)

    def log_person_left(self, person_data: dict) -> None:
        self._write(EventTypeEnum.PERSON_LEFT, person_data)

    def _write(self, event_type: EventTypeEnum, person_data: dict) -> None:
        db = self._session_factory()
        try:
            db.add(
                ActivityLog(
                    source=SourceEnum.PERSON_DETECTOR,
                    event_type=event_type,
                    data=json.dumps(person_data),
                    timestamp=datetime.utcnow(),
                )
            )
            db.commit()
        finally:
            db.close()
