import time
from typing import Callable


class SimpleRateLimiter:
    def __init__(
        self,
        min_interval_seconds: int = 30,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.min_interval = min_interval_seconds
        self._clock = clock
        self.last_save_time: float | None = None

    def can_save(self) -> bool:
        now = self._clock()
        if self.last_save_time is None or now - self.last_save_time >= self.min_interval:
            self.last_save_time = now
            return True
        return False

