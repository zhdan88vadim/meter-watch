import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv


def load_environment():
    """Loading environment variables"""
    in_docker = os.path.exists("/.dockerenv")

    if in_docker:
        print("🐳 Running in Docker, using environment variables")
        # In Docker, variables are already in the environment
        return

    # Local development - load .env
    base_dir = Path(__file__).resolve().parent.parent.parent
    env_file = base_dir / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env from {env_file}")
    else:
        print(f"⚠️ No .env file found at {env_file}")


load_environment()


@dataclass(frozen=True)
class RedisKeys:
    startup: str = "system:startup:timestamp"
    gas_flow: str = "meter:gas:flow"
    gas_number: str = "meter:gas:number"
    gas_last_activity: str = "meter:gas:last_activity"
    human_last_seen: str = "human:last_seen"
    human_last_seen_str: str = "human:last_seen_str"
    alert_cooldown: str = "alert:telegram:cooldown"
    active_people: str = "active:people"
    recording_prefix: str = "recording:"
    alert_triggered: str = "alert:gas:triggered"


@dataclass(frozen=True)
class Config:
    # Redis
    RTSP_URL: str = os.getenv("RTSP_URL")

    DATABASE_URL: str = os.getenv("DATABASE_URL")
    DATABASE_URL_ASYNC: str = os.getenv("DATABASE_URL_ASYNC", 0)

    REDIS_HOST: str = os.getenv("REDIS_HOST")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 0))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = 0
    REDIS_TIMEOUT: int = 5

    # System
    STARTUP_DURATION: int = 5  # 5 sec
    PERSON_ABSENCE_THRESHOLD: int = 60 * 10  # 10 minutes
    PERSON_IS_ACTIVE_THRESHOLD: int = 5  # 5 sec
    ALERT_COOLDOWN: int = 30
    CHECK_INTERVAL: int = 3
    RECORDING_EXPIRE_TIME: int = 86400
    STARTUP_PERSON_TIMEOUT: int = 60

    # Video
    DEFAULT_FPS: int = 15
    DEFAULT_FRAME_WIDTH: int = 640
    DEFAULT_FRAME_HEIGHT: int = 480
    BUFFER_SECONDS: int = 4
    POST_ROLL_SECONDS: int = 4
    FRAME_SKIP: int = 30

    # Web
    WEB_HOST: str = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT: int = int(os.getenv("WEB_PORT", 5000))
    WEB_DEBUG: bool = os.getenv("WEB_DEBUG", "False").lower() == "true"

    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    TELEGRAM_API_URL: str = "https://api.telegram.org/bot{}/sendMessage"
    TELEGRAM_TIMEOUT: int = 5

    # API
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "your-secret-key-here")

    # Redis Keys
    REDIS_KEYS: RedisKeys = RedisKeys()


config = Config()