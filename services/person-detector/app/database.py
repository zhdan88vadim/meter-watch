import json
from datetime import datetime
from meter_watch_shared.db import SessionLocal
from meter_watch_shared.models import ActivityLog, SourceEnum, EventTypeEnum


def log_person_detected_to_database(person_data):
    db = SessionLocal()
    try:
        log = ActivityLog(
            source=SourceEnum.PERSON_DETECTOR,
            event_type=EventTypeEnum.PERSON_DETECTED,
            data=json.dumps(person_data),
            timestamp=datetime.utcnow(),
        )
        db.add(log)
        db.commit()
    finally:
        db.close()


def log_person_left_to_database(person_data):
    db = SessionLocal()
    try:
        log = ActivityLog(
            source=SourceEnum.PERSON_DETECTOR,
            event_type=EventTypeEnum.PERSON_LEFT,
            data=json.dumps(person_data),
            timestamp=datetime.utcnow(),
        )
        db.add(log)
        db.commit()
    finally:
        db.close()
