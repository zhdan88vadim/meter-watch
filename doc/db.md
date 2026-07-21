Ваш код хорош, но его можно улучшить. Давайте рассмотрим оба подхода и их преимущества:

## Текущий подход (функциональный)

**Плюсы:**
- Простота и прямота
- Легко понять новичкам
- Минимум абстракций

**Минусы:**
- Дублирование кода (SessionLocal, try/finally)
- Сложно тестировать (зависимости жестко закодированы)
- Трудно расширять
- Нарушение DRY (Don't Repeat Yourself)

## Классовый подход (рекомендуемый)

```python
# services/database_service.py
import json
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
from models.monitoring_models import RecognitionResult
from meter_watch_shared.db import SessionLocal
from meter_watch_shared.models import (
    ActivityLog,
    MeterReading,
    SourceEnum,
    EventTypeEnum,
)


@dataclass
class PersonData:
    """Data model for person detection"""
    confidence: float
    bbox: Optional[tuple] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "bbox": self.bbox,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class DatabaseService:
    """Service for database operations"""
    
    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        db = self.session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    
    def _create_activity_log(
        self,
        source: str,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        meter_reading: Optional[float] = None,
        person_detected: Optional[bool] = None
    ) -> ActivityLog:
        """Create activity log entry"""
        return ActivityLog(
            source=source,
            event_type=event_type,
            data=json.dumps(data) if data else None,
            meter_reading=meter_reading,
            person_detected=person_detected,
            timestamp=datetime.utcnow(),
        )
    
    def log_person_detected(self, person_data: PersonData) -> bool:
        """Log person detection event"""
        try:
            with self.get_session() as db:
                log = self._create_activity_log(
                    source="person_detector",
                    event_type="person_detected",
                    data=person_data.to_dict(),
                    person_detected=True
                )
                db.add(log)
                return True
        except Exception as e:
            print(f"❌ Error logging person detected: {e}")
            return False
    
    def log_person_left(self) -> bool:
        """Log person left event"""
        try:
            with self.get_session() as db:
                log = self._create_activity_log(
                    source="person_detector",
                    event_type="person_left",
                    person_detected=False
                )
                db.add(log)
                return True
        except Exception as e:
            print(f"❌ Error logging person left: {e}")
            return False
    
    def save_meter_reading(self, result: RecognitionResult) -> bool:
        """Save meter reading to database"""
        try:
            with self.get_session() as db:
                # Activity log
                log = self._create_activity_log(
                    source=SourceEnum.METER,
                    event_type=EventTypeEnum.READING,
                    data={"value": result.number},
                    meter_reading=result.number
                )
                db.add(log)
                
                # Meter reading
                reading = MeterReading(
                    value=result.number,
                    timestamp=datetime.utcnow(),
                    min_conf=result.min_conf
                )
                db.add(reading)
                
                print(f"✅ Logged meter reading: {result.number}")
                return True
                
        except Exception as e:
            print(f"❌ Error logging meter reading: {e}")
            return False
    
    def get_recent_readings(self, limit: int = 10) -> list:
        """Get recent meter readings"""
        try:
            with self.get_session() as db:
                readings = db.query(MeterReading)\
                    .order_by(MeterReading.timestamp.desc())\
                    .limit(limit)\
                    .all()
                return readings
        except Exception as e:
            print(f"❌ Error getting readings: {e}")
            return []
    
    def get_last_person_event(self) -> Optional[ActivityLog]:
        """Get last person detection event"""
        try:
            with self.get_session() as db:
                return db.query(ActivityLog)\
                    .filter(ActivityLog.source == "person_detector")\
                    .order_by(ActivityLog.timestamp.desc())\
                    .first()
        except Exception as e:
            print(f"❌ Error getting last person event: {e}")
            return None


# Singleton instance for backward compatibility
_db_service = None

def get_db_service() -> DatabaseService:
    """Get database service instance"""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service


# Backward compatibility functions
def log_person_detected_to_database(person_data):
    """Legacy function for backward compatibility"""
    service = get_db_service()
    if isinstance(person_data, dict):
        person_data = PersonData(**person_data)
    return service.log_person_detected(person_data)


def log_person_left_to_database():
    """Legacy function for backward compatibility"""
    return get_db_service().log_person_left()


def save_meter_data_to_database(result: RecognitionResult) -> None:
    """Legacy function for backward compatibility"""
    get_db_service().save_meter_reading(result)
```

## Альтернативный подход (репозиторий + сервис)

```python
# repositories/base_repository.py
from typing import TypeVar, Generic, Type, List, Optional
from sqlalchemy.orm import Session

T = TypeVar('T')


class BaseRepository(Generic[T]):
    """Base repository with CRUD operations"""
    
    def __init__(self, model: Type[T], session: Session):
        self.model = model
        self.session = session
    
    def add(self, entity: T) -> T:
        self.session.add(entity)
        return entity
    
    def get_by_id(self, id: int) -> Optional[T]:
        return self.session.query(self.model).filter_by(id=id).first()
    
    def get_all(self, limit: Optional[int] = None) -> List[T]:
        query = self.session.query(self.model)
        if limit:
            query = query.limit(limit)
        return query.all()


# repositories/activity_repository.py
from meter_watch_shared.models import ActivityLog


class ActivityRepository(BaseRepository[ActivityLog]):
    """Repository for ActivityLog"""
    
    def __init__(self, session: Session):
        super().__init__(ActivityLog, session)
    
    def get_by_source(self, source: str, limit: int = 10) -> List[ActivityLog]:
        return self.session.query(ActivityLog)\
            .filter_by(source=source)\
            .order_by(ActivityLog.timestamp.desc())\
            .limit(limit)\
            .all()


# services/database_service.py (упрощенная версия)
from repositories.activity_repository import ActivityRepository
from repositories.meter_reading_repository import MeterReadingRepository


class DatabaseService:
    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory
    
    @contextmanager
    def get_session(self):
        db = self.session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    
    def save_meter_reading(self, result: RecognitionResult) -> bool:
        try:
            with self.get_session() as db:
                activity_repo = ActivityRepository(db)
                reading_repo = MeterReadingRepository(db)
                
                # Create activity log
                log = ActivityLog(
                    source=SourceEnum.METER,
                    event_type=EventTypeEnum.READING,
                    data=json.dumps({"value": result.number}),
                    meter_reading=result.number,
                    timestamp=datetime.utcnow()
                )
                activity_repo.add(log)
                
                # Create meter reading
                reading = MeterReading(
                    value=result.number,
                    timestamp=datetime.utcnow(),
                    min_conf=result.min_conf
                )
                reading_repo.add(reading)
                
                return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
```

## Сравнение подходов

### Функциональный подход (ваш текущий)
```python
# Использование
log_person_detected_to_database({"confidence": 0.95})
save_meter_data_to_database(result)
```

**✅ Простота**
**❌** Сложно тестировать
**❌** Дублирование кода
**❌** Нет гибкости

### Классовый подход (рекомендуемый)
```python
# Использование
db_service = DatabaseService()
db_service.log_person_detected(PersonData(confidence=0.95))
db_service.save_meter_reading(result)

# Для тестирования
mock_service = DatabaseService(mock_session)
```

**✅** Легко тестировать (можно подменить зависимости)
**✅** Нет дублирования (context manager)
**✅** Расширяемость
**✅** Единый источник правды
**✅** Можно добавлять новые методы

## Ключевые преимущества классового подхода:

1. **Тестируемость**:
```python
def test_save_meter_reading():
    mock_session = Mock()
    service = DatabaseService(lambda: mock_session)
    result = service.save_meter_reading(test_result)
    assert result is True
    mock_session.add.assert_called()
```

2. **Гибкость**:
```python
# Можно легко добавить кеширование
class CachedDatabaseService(DatabaseService):
    def __init__(self, cache_ttl=300):
        super().__init__()
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    def get_recent_readings(self, limit=10):
        cache_key = f"readings_{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        result = super().get_recent_readings(limit)
        self.cache[cache_key] = result
        return result
```

3. **Управление ресурсами**:
```python
# Автоматическое управление сессиями через context manager
with db_service.get_session() as session:
    # Все операции в одной транзакции
    db_service.create_reading(session, data)
    db_service.create_log(session, data)
    # Автоматический commit/rollback
```

4. **Единая обработка ошибок**:
```python
class DatabaseService:
    def _handle_error(self, error, context):
        # Логирование, отправка в Sentry, etc.
        logger.error(f"Database error in {context}: {error}")
        # Можно добавить retry логику
        return False
```

## Рекомендация

**Используйте классовый подход**, особенно если:
- Проект будет расти
- Нужны тесты
- Планируется добавление новых функций
- Работает несколько разработчиков

**Оставьте функциональный подход**, если:
- Проект маленький и не будет расти
- Нужно быстрое решение
- Вы новичок в ООП

Для вашего случая, учитывая сложность проекта (распознавание, API, мониторинг), **рекомендую классовый подход** с добавлением:
- Context manager для работы с БД
- Dependency injection для тестирования
- Единую обработку ошибок
- Возможность расширения