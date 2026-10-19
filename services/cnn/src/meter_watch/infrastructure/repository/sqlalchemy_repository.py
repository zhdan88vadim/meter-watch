"""SQLAlchemy-backed repository adapter."""
from __future__ import annotations

import json
from typing import Any, Callable
from datetime import datetime

from meter_watch.domain.models import MeterReading
from meter_watch.domain.protocols import Logger


class SqlAlchemyReadingRepository:
    """Persists readings via meter_watch_shared SQLAlchemy models.

    The DB imports are isolated here so nothing else in the app
    depends on SQLAlchemy.
    """

    def __init__(
        self,
        session_factory: Callable[[], Any],
        logger: Logger,
    ) -> None:
        self._session_factory = session_factory
        self._logger = logger

    def save(self, reading: MeterReading, is_anomaly: bool) -> None:
        from meter_watch_shared.models import (  # noqa: PLC0415
            ActivityLog,
            EventTypeEnum,
            MeterReading as DbReading,
            SourceEnum,
        )

        db = self._session_factory()
        try:
            db.add(
                ActivityLog(
                    source=SourceEnum.METER,
                    event_type=EventTypeEnum.READING,
                    data=json.dumps({
                        "value": reading.number.value,
                        "is_anomaly": is_anomaly,
                    }),
                    meter_reading=reading.number.value,
                    timestamp=datetime.utcnow(),
                )
            )
            db.add(
                DbReading(
                    value=reading.number.value,
                    timestamp=datetime.utcnow(),
                    min_conf=reading.min_confidence,
                    is_anomaly=is_anomaly,
                )
            )
            db.commit()
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            self._logger.error("db save failed", error=str(exc))
            raise
        finally:
            db.close()
