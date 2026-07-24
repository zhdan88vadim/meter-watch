import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

class ConfigKeys(str, Enum):
    """Enum для всех ключей конфигурации - единый источник истины"""
    # Камера
    CAMERA_URL = "camera_url"
    CAMERA_REQUEST_PAUSE = "camera_request_pause"
    
    # Мониторинг
    MONITORING_ENABLED = "monitoring_enabled"
    SAVE_THRESHOLD = "save_threshold"
    SAVE_BAD_PHOTOS = "save_bad_photos"
    
    # Обрезка
    CROP_TOP = "crop_top"
    CROP_LEFT = "crop_left"
    CROP_RIGHT = "crop_right"
    CROP_BOTTOM = "crop_bottom"
    
    # Распознавание
    DIGIT_WIDTH = "digit_width"
    DIGIT_COUNT = "digit_count"
    
    # Другое
    DEBUG_MODE = "debug_mode"
    
    @classmethod
    def values(cls):
        return [key.value for key in cls]

@dataclass
class AppConfig:
    """Dataclass для конфигурации с типами и значениями по умолчанию"""
    # Камера
    camera_url: str = "rtsp://username:password@ip:port/stream"
    camera_request_pause: int = 5
    
    # Мониторинг
    monitoring_enabled: bool = True
    save_threshold: float = 0.6
    save_bad_photos: bool = True
    
    # Обрезка
    crop_top: int = 45
    crop_left: int = 8
    crop_right: int = 0
    crop_bottom: int = 35
    
    # Распознавание
    digit_width: int = 28
    digit_count: int = 5
    
    # Другое
    debug_mode: bool = False
    
    def __post_init__(self):
        self._config_file = "config.json"
        self._load_from_file()
    
    def _load_from_file(self):
        """Загружает из файла"""
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                print(f"Config loaded from {self._config_file}")
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def save(self):
        """Сохраняет в файл"""
        try:
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def to_dict(self) -> dict:
        """Возвращает словарь"""
        return asdict(self)

    def get(self, key: ConfigKeys) -> Any:
        return getattr(self, key.value)

    def set(self, key: ConfigKeys, value: Any) -> bool:
        if hasattr(self, key.value):
            setattr(self, key.value, value)
            self.save()
            return True
        return False

    def update(self, updates: dict) -> dict:
        """Обновляет несколько параметров"""
        changes = {}
        for key, value in updates.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                changes[key] = {"old": old_value, "new": value}
        
        if changes:
            self.save()
        
        return changes

# Глобальный экземпляр
config = AppConfig()
# config.save()