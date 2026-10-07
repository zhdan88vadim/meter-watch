from __future__ import annotations

from typing import Dict, List, Optional

from app.domain.models import Detection, Frame


class FakeStore:
    def __init__(self) -> None:
        self.data: Dict[str, str] = {}
        self.ttls: Dict[str, int] = {}
        self.timestamps: Dict[str, float] = {}
        self.now: float = 0.0

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        self.data[key] = value
        if ttl is not None:
            self.ttls[key] = ttl
        return True

    def delete(self, key: str) -> bool:
        self.data.pop(key, None)
        self.ttls.pop(key, None)
        self.timestamps.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        return key in self.data

    def seconds_since(self, key: str) -> Optional[float]:
        if key not in self.timestamps:
            return None
        return self.now - self.timestamps[key]

    def set_timestamp(self, key: str, ttl: Optional[int] = None) -> float:
        self.timestamps[key] = self.now
        self.data[key] = str(self.now)
        if ttl is not None:
            self.ttls[key] = ttl
        return self.now


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: List[str] = []
        self.alerts: List[str] = []

    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        self.messages.append(message)
        return True

    def send_alert(self, alert_type: str, data: Optional[dict] = None) -> bool:
        self.alerts.append(alert_type)
        return True


class FakeRepository:
    def __init__(self) -> None:
        self.detected: List[dict] = []
        self.left: List[dict] = []

    def log_person_detected(self, person_data: dict) -> None:
        self.detected.append(person_data)

    def log_person_left(self, person_data: dict) -> None:
        self.left.append(person_data)


class FakeDetector:
    def __init__(
        self, detections_per_call: Optional[List[List[Detection]]] = None
    ) -> None:
        self._queue = detections_per_call or []
        self.calls = 0

    def detect(self, frame: Frame):
        self.calls += 1
        if self._queue:
            return self._queue.pop(0)
        return []


class FakeSource:
    def __init__(self, frames: List[Frame]) -> None:
        self._frames = frames
        self._i = 0
        self.released = False

    @property
    def fps(self) -> float:
        return 25.0

    def read(self):
        if self._i >= len(self._frames):
            return None
        f = self._frames[self._i]
        self._i += 1
        return f

    def release(self) -> None:
        self.released = True
