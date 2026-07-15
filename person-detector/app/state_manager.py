from meter_watch_shared.config import config
from meter_watch_shared.redis_manager import RedisManager
import logging
import time

logger = logging.getLogger(__name__)

class StateManager:
    @staticmethod
    def reset_alert_state():
        RedisManager.delete_key(config.REDIS_KEYS['alert_triggered'])
        RedisManager.delete_key(config.REDIS_KEYS['alert_cooldown'])
        RedisManager.set_key(config.REDIS_KEYS['human_last_seen'], str(time.time()))
        logger.info("🔄 Alert reset")            