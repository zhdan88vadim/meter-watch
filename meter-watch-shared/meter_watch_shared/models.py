from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy import Enum
import enum

Base = declarative_base()

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
    source = Column(String(50))  # 'meter' или 'person_detector'
    event_type = Column(String(50))  # 'reading', 'person_detected', 'person_left'
    data = Column(Text)  # JSON данные
    meter_reading = Column(Float, nullable=True)
    person_detected = Column(Boolean, nullable=True)

class MeterReading(Base):
    __tablename__ = "meter_readings"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    value = Column(Float)
    min_conf = Column(Float, nullable=True) 