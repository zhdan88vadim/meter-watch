from sqlalchemy import Column, Integer, DateTime, Float, Text, Boolean
from datetime import datetime
from sqlalchemy import Enum
import enum
from .db import Base


class SourceEnum(enum.Enum):
    METER = "meter"
    PERSON_DETECTOR = "person_detector"


class EventTypeEnum(enum.Enum):
    READING = "reading"
    PERSON_DETECTED = "person_detected"
    PERSON_LEFT = "person_left"


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(Enum(SourceEnum), nullable=False)
    event_type = Column(Enum(EventTypeEnum), nullable=False)
    data = Column(Text)  # JSON данные
    meter_reading = Column(Float, nullable=True)


class MeterReading(Base):
    __tablename__ = "meter_readings"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    value = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=True)
    min_conf = Column(Float, nullable=True)
