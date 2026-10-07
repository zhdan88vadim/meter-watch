import threading
import logging
import signal
import sys

from meter_watch_shared.config import config
from meter_watch_shared.db import init_database

from app.person_tracker import PersonTracker
from app.services.redis_service import RedisService
from app.infra.yolo_detector import YoloDetector
from app.telegram_bot import TelegramBot
from app.api import start_api
from app.video_buffer import VideoBuffer
from app.rate_limiter import SimpleRateLimiter
from app.services.safety_monitor import SafetyMonitor
from app.adapters.redis_store import RedisKeyValueStore
from app.adapters.telegram_notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global tracker
tracker = None


def signal_handler(sig, frame):
    """Signal handling for graceful shutdown"""
    logger.info("🛑 Received shutdown signal")
    if tracker:
        tracker.cleanup()
    sys.exit(0)


def _thresholds() -> dict:
    return {
        "person_is_active_threshold": config.PERSON_IS_ACTIVE_THRESHOLD,
        "person_absence_threshold": config.PERSON_ABSENCE_THRESHOLD,
        "startup_person_timeout": config.STARTUP_PERSON_TIMEOUT,
        "alert_cooldown": config.ALERT_COOLDOWN,
    }

def main():

    # init_database()

    global tracker

    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("🚀 Starting Security System...")


    store = RedisKeyValueStore()
    bot = TelegramBot(config, store)
    bot.start()

    notifier = TelegramNotifier(config)

    safety_monitor = SafetyMonitor(
        store=store,
        notifier=notifier,
        keys=config.REDIS_KEYS,
        thresholds=_thresholds(),
        check_interval=config.CHECK_INTERVAL,
    )
    safety_monitor.start()

    # Start tracker in a separate thread
    tracker = PersonTracker(
        detector=YoloDetector("yolov8n.pt"),
        buffer=VideoBuffer(config.BUFFER_SECONDS, config.DEFAULT_FPS),
        rate_limiter=SimpleRateLimiter(30),
        source=config.RTSP_URL,
        post_roll_seconds=config.POST_ROLL_SECONDS,
        frame_skip=config.FRAME_SKIP,
    )

    redis_service = RedisService(notifier=notifier)
    redis_service.cleanup()
    redis_service.mark_startup()

    tracker_thread = threading.Thread(target=tracker.run, daemon=True)
    tracker_thread.start()
            

    # Start API in a separate thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)
    finally:
        safety_monitor.stop()
        bot.stop()

 
if __name__ == "__main__":
    main()