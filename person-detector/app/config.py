import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env ДО ВСЕГО
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

@dataclass
class Config:
    # Redis
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', 6379))
    REDIS_PASSWORD: str = os.getenv('REDIS_PASSWORD', '')
    REDIS_DB: int = 0
    REDIS_TIMEOUT: int = 5
    
    # System
    STARTUP_DURATION: int = 60 * 1  # 5 minutes
    PERSON_ABSENCE_THRESHOLD: int = 60 * 1  # 10 minutes
    ALERT_COOLDOWN: int = 30  # 2 minutes
    CHECK_INTERVAL: int = 30
    # PERSON_EXPIRE_TIME: int = 3600
    RECORDING_EXPIRE_TIME: int = 86400
    STARTUP_PERSON_TIMEOUT: int = 60
    
    # Video
    DEFAULT_FPS: int = 15
    DEFAULT_FRAME_WIDTH: int = 640
    DEFAULT_FRAME_HEIGHT: int = 480
    BUFFER_SECONDS: int = 4
    POST_ROLL_SECONDS: int = 4
    FRAME_SKIP: int = 30  # Process every 2nd frame
    
    # Web
    WEB_HOST: str = os.getenv('WEB_HOST', '0.0.0.0')
    WEB_PORT: int = int(os.getenv('WEB_PORT', 5000))
    WEB_DEBUG: bool = os.getenv('WEB_DEBUG', 'False').lower() == 'true'
    
    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv('TELEGRAM_CHAT_ID')
    TELEGRAM_API_URL: str = 'https://api.telegram.org/bot{}/sendMessage'
    TELEGRAM_TIMEOUT: int = 5
    
    # API
    API_SECRET_KEY: str = os.getenv('API_SECRET_KEY', 'your-secret-key-here')
    
    # Redis Keys
    REDIS_KEYS = {
        'startup': 'system:startup:timestamp',
        'gas_flow': 'meter:gas:flow',
        'human_last_seen': 'human:last_seen',
        'alert_cooldown': 'alert:telegram:cooldown',
        'active_people': 'active:people',
        'recording_prefix': 'recording:',
        'alert_triggered': 'alert:gas:triggered',
    }

config = Config()