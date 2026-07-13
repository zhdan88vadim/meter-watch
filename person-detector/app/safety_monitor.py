import time
import threading
from datetime import datetime
from typing import Optional, Dict, Callable, List
from app.config import config
from app.redis_manager import RedisManager
from app.telegram_bot import telegram_bot
import logging

logger = logging.getLogger(__name__)

class SafetyMonitor:
    """Мониторинг безопасности: газ + человек"""
    
    def __init__(self, check_interval: int = config.CHECK_INTERVAL):
        self.check_interval = check_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.alert_count = 0
        self.last_check_time = 0
        
        # Callbacks
        self.on_alert_callbacks: List[Callable] = []
        self.on_person_detected_callbacks: List[Callable] = []
        self.on_person_missing_callbacks: List[Callable] = []
        
        logger.info(f"🔒 SafetyMonitor started (check every {check_interval}s)")
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.check_interval + 5)
    
    def _monitor_loop(self):
        while self.running:
            try:
                self._check()
            except Exception as e:
                logger.error(f"Check error: {e}")
            
            # Ждем interval секунд
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _check(self):
        """Основная проверка"""
        # 1. Газ не идет - безопасно
        if RedisManager.get_key(config.REDIS_KEYS['gas_flow']) != '1':
            return
        
        # 2. Режим запуска - ждем
        if RedisManager.key_exists(config.REDIS_KEYS['startup']):
            self._handle_startup()
            return
        
        # 3. Проверяем человека
        self._check_person()
    
    def _handle_startup(self):
        """Обработка режима запуска"""
        time_since_seen = RedisManager.get_time_since(config.REDIS_KEYS['human_last_seen'])
        
        # Если человек появился - выходим из режима запуска
        if time_since_seen is not None and time_since_seen < config.STARTUP_PERSON_TIMEOUT:
            RedisManager.delete_key(config.REDIS_KEYS['startup'])
            RedisManager.set_key(config.REDIS_KEYS['human_last_seen'], str(time.time()))
            logger.info("👤 Person detected - startup mode cleared")
            self._notify(self.on_person_detected_callbacks)
    
    def _check_person(self):
        """Проверка присутствия человека"""
        time_since_seen = RedisManager.get_time_since(config.REDIS_KEYS['human_last_seen'])
        
        # Человек есть (видели меньше минуты)
        if time_since_seen is not None and time_since_seen < 60:
            self._notify(self.on_person_detected_callbacks)
            # Сбрасываем тревогу если была
            if RedisManager.key_exists(config.REDIS_KEYS['alert_triggered']):
                RedisManager.delete_key(config.REDIS_KEYS['alert_triggered'])
                logger.info("✅ Alert cleared - person returned")
            return
        
        # Человек отсутствует
        if time_since_seen is None or time_since_seen >= config.PERSON_ABSENCE_THRESHOLD:
            minutes = int(time_since_seen / 60)
            logger.warning(f"⚠️ Person missing for {minutes} minutes!")
            self._send_alert()
            self._notify(self.on_person_missing_callbacks)
        else:
            # Человек отсутствует, но еще не критично
            minutes = int(time_since_seen / 60)
            logger.debug(f"👤 Person missing for {minutes} minutes")

    
    def _send_alert(self):
        """Отправка тревоги"""
        # Проверяем cooldown
        cooldown_key = config.REDIS_KEYS['alert_cooldown']
        if RedisManager.key_exists(cooldown_key):
            remaining = RedisManager.get_time_since(cooldown_key)
            if remaining:
                logger.debug(f"⏳ Alert cooldown: {int(remaining)}s remaining")
            return
        
        # Проверяем, не активна ли уже тревога
        if RedisManager.key_exists(config.REDIS_KEYS['alert_triggered']):
            logger.debug("⚠️ Alert already triggered")
            return
        
        # Проверяем cooldown и активную тревогу
        if (RedisManager.key_exists(config.REDIS_KEYS['alert_cooldown']) or 
            RedisManager.key_exists(config.REDIS_KEYS['alert_triggered'])):
            return
        
        # Отправляем
        logger.info("🚨 SENDING ALERT!")
        success = telegram_bot.send_alert('gas_alert')
        
        if success:
            self.alert_count += 1
            
            # Устанавливаем флаг тревоги
            RedisManager.set_key(config.REDIS_KEYS['alert_triggered'], '1')
            RedisManager.set_key(config.REDIS_KEYS['alert_cooldown'], '1', config.ALERT_COOLDOWN)
            
            logger.info(f"✅ Alert sent successfully (total: {self.alert_count})")
            
            # Вызываем колбэки
            self._notify(self.on_alert_callbacks)
        else:
            logger.error("❌ Failed to send alert")            
    
    def _notify(self, callbacks: List[Callable]):
        """Вызов колбэков"""
        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    # === Публичные методы ===
    
    def reset_alert(self):
        RedisManager.delete_key(config.REDIS_KEYS['alert_triggered'])
        RedisManager.delete_key(config.REDIS_KEYS['alert_cooldown'])
        RedisManager.set_key(config.REDIS_KEYS['human_last_seen'], str(time.time()))
        logger.info("🔄 Alert reset")

    def add_on_alert_callback(self, callback: Callable):
        self.on_alert_callbacks.append(callback)
    
    def add_on_person_detected_callback(self, callback: Callable):
        self.on_person_detected_callbacks.append(callback)
    
    def add_on_person_missing_callback(self, callback: Callable):
        self.on_person_missing_callbacks.append(callback)