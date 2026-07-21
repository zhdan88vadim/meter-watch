import time


class SimpleRateLimiter:
    def __init__(self, min_interval_seconds: int = 30):
        self.min_interval = min_interval_seconds
        self.last_save_time = 0

    def can_save(self) -> bool:
        """Проверяет, прошло ли время"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.min_interval:
            self.last_save_time = current_time
            return True
        return False
