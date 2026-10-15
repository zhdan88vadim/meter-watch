"""In-memory config provider for tests."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from meter_watch.infrastructure.config.file_provider import AppConfigData


class InMemoryConfigProvider:
    """Config provider that never touches the filesystem."""

    def __init__(self, **kwargs: Any) -> None:
        self._data = AppConfigData(**kwargs)

    def get(self, key: str) -> Any:
        return getattr(self._data, key)

    def set(self, key: str, value: Any) -> bool:
        if not hasattr(self._data, key):
            return False
        setattr(self._data, key, value)
        return True

    def update(self, updates: dict[str, Any]) -> dict[str, dict[str, Any]]:
        changes: dict[str, dict[str, Any]] = {}
        for k, v in updates.items():
            if hasattr(self._data, k):
                old = getattr(self._data, k)
                setattr(self._data, k, v)
                changes[k] = {"old": old, "new": v}
        return changes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self._data)

    def has(self, key: str) -> bool:
        return hasattr(self._data, key)
