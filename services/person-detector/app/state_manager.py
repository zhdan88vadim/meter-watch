from meter_watch_shared.config import config
from meter_watch_shared.redis_manager import RedisManager
import logging
import time

logger = logging.getLogger(__name__)

class StateManager:
    @staticmethod
    def reset_alert():
        time_str = time.strftime("%H:%M %d:%m:%Y", time.localtime(time.time()))

        RedisManager.delete_key(config.REDIS_KEYS['alert_triggered'])
        RedisManager.delete_key(config.REDIS_KEYS['alert_cooldown'])
        RedisManager.set_key(config.REDIS_KEYS['human_last_seen'], str(time.time()))
        RedisManager.set_key(config.REDIS_KEYS['human_last_seen_str'], time_str)
        logger.info("🔄 Alert reset")            