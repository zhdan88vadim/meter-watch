"""JSON-file-backed config provider."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AppConfigData:
    """Default config values."""

    camera_url: str = "rtsp://username:password@ip:port/stream"
    camera_request_pause: int = 5
    monitoring_enabled: bool = True
    save_threshold: float = 0.6
    save_bad_photos: bool = True


class FileConfigProvider:
    """Reads/writes JSON config. No global state, no import-time side effects."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._data = AppConfigData()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if hasattr(self._data, k):
                    setattr(self._data, k, v)
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading config: {exc}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._data), f, indent=4, ensure_ascii=False)

    def get(self, key: str) -> Any:
        return getattr(self._data, key)

    def set(self, key: str, value: Any) -> bool:
        if not hasattr(self._data, key):
            return False
        setattr(self._data, key, value)
        self.save()
        return True

    def update(self, updates: dict[str, Any]) -> dict[str, dict[str, Any]]:
        changes: dict[str, dict[str, Any]] = {}
        for k, v in updates.items():
            if hasattr(self._data, k):
                old = getattr(self._data, k)
                setattr(self._data, k, v)
                changes[k] = {"old": old, "new": v}
        if changes:
            self.save()
        return changes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self._data)

    def has(self, key: str) -> bool:
        return hasattr(self._data, key)
