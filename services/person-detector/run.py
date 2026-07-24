import threading
import logging
import signal
import sys
from app.person_tracker import PersonTracker
from app.telegram_bot import telegram_bot
from app.api import start_api
from meter_watch_shared.config import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Глобальный трекер
tracker = None


def signal_handler(sig, frame):
    """Обработка сигналов для корректного завершения"""
    logger.info("🛑 Received shutdown signal")
    if tracker:
        tracker.cleanup()
    sys.exit(0)


def main():
    global tracker

    # Настройка обработки сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("🚀 Starting Security System...")

    # Запуск Telegram бота
    telegram_bot.start()

    # Запуск трекера в отдельном потоке
    tracker = PersonTracker(
        source=config.RTSP_URL,
        buffer_seconds=config.BUFFER_SECONDS,
        post_roll_seconds=config.POST_ROLL_SECONDS,
        frame_skip=config.FRAME_SKIP,
    )

    tracker_thread = threading.Thread(target=tracker.run, daemon=True)
    tracker_thread.start()

    # Запуск API в отдельном потоке
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()

    try:
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
