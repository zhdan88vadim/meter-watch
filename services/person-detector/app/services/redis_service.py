
from meter_watch_shared.config import config
from meter_watch_shared.redis_manager import RedisManager
from app.protocol_models import Notifier
import logging

logger = logging.getLogger(__name__)

class RedisService:
    """Handles Redis cleanup and startup marking."""

    def __init__(self, notifier: Notifier, startup_duration: int = config.STARTUP_DURATION):
        self._notifier = notifier
        self.startup_duration = startup_duration

    def cleanup(self) -> None:
        """Clean stale Redis keys on startup."""
        try:
            conn = RedisManager.get_connection()
            # conn.delete(config.REDIS_KEYS['active_people'])
            conn.delete(config.REDIS_KEYS["alert_triggered"])
            conn.delete(config.REDIS_KEYS["alert_cooldown"])
            logger.info("✅ Redis cleaned")
        except Exception as exc:
            logger.warning("Redis cleanup failed: %s", exc)

    def mark_startup(self) -> None:
        """Mark service startup and notify Telegram."""
        RedisManager.set_timestamp_key(
            config.REDIS_KEYS["startup"],
            self.startup_duration,
        )
        self._notifier.send_alert("startup")
