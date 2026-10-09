"""
Adapter that wraps the existing RedisManager so it satisfies KeyValueStore.
RedisManager stays untouched as a class-based singleton in meter_watch_shared.
"""
from __future__ import annotations

from typing import Optional

from meter_watch_shared.redis_manager import RedisManager


class RedisKeyValueStore:
    def __init__(self, manager: type[RedisManager] = RedisManager) -> None:
        self._m = manager

    def get(self, key: str) -> str | None:
        return self._m.get_key(key)

    def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        return self._m.set_key(key, value, ttl)

    def delete(self, key: str) -> bool:
        return self._m.delete_key(key)

    def exists(self, key: str) -> bool:
        return self._m.key_exists(key)

    def seconds_since(self, key: str) -> float | None:
        return self._m.get_time_since(key)

    def set_timestamp(self, key: str, ttl: int | None = None) -> float:
        return self._m.set_timestamp_key(key, ttl)

    def hset(self, key: str, mapping: dict) -> bool:
        return self._m.hset(key, mapping)

    def expire(self, key: str, ttl: int) -> bool:
        return self._m.expire(key, ttl)
