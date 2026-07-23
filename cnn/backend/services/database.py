import json
from datetime import datetime
from models.monitoring_models import RecognitionResult
from meter_watch_shared.db import SessionLocal
from meter_watch_shared.models import (
    ActivityLog,
    MeterReading,
    SourceEnum,
    EventTypeEnum,
)


def save_meter_data_to_database(result: RecognitionResult, is_anomaly: bool) -> None:
    """Сохранить показания в БД"""
    db = SessionLocal()
    try:
        # Лог активности
        log = ActivityLog(
            source=SourceEnum.METER,
            event_type=EventTypeEnum.READING,
            data=json.dumps({"value": result.number, "is_anomaly": is_anomaly}),
            meter_reading=result.number,
            timestamp=datetime.utcnow(),
        )
        db.add(log)

        # Показание счетчика
        reading = MeterReading(
            value=result.number, timestamp=datetime.utcnow(), min_conf=result.min_conf, is_anomaly=is_anomaly
        )
        db.add(reading)

        db.commit()
        print(f"✅ Logged meter reading: {result.number}")

    except Exception as e:
        db.rollback()
        print(f"❌ Error logging meter reading: {e}")
    finally:
        db.close()
