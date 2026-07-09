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
    """
    Мониторинг безопасности: проверяет состояние газа и присутствие человека.
    Отправляет уведомления при обнаружении опасной ситуации.
    """
    
    def __init__(self, check_interval: int = config.CHECK_INTERVAL):
        """
        Инициализация монитора безопасности
        
        Args:
            check_interval: Интервал проверки в секундах
        """
        self.check_interval = check_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_check_time = 0
        self.alert_count = 0
        self.startup_check_count = 0
        
        # Статистика
        self.stats = {
            'checks_performed': 0,
            'gas_detected': 0,
            'alerts_sent': 0,
            'startup_mode_checks': 0,
            'person_detected': 0,
            'person_missing': 0
        }
        
        # Callback для внешних обработчиков
        self.on_alert_callbacks: List[Callable] = []
        self.on_person_detected_callbacks: List[Callable] = []
        self.on_person_missing_callbacks: List[Callable] = []
        
        logger.info(f"🔒 SafetyMonitor initialized (check interval: {check_interval}s)")
    
    def start(self):
        """Запускает монитор в отдельном потоке"""
        if self.running:
            logger.warning("⚠️ SafetyMonitor already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("🔒 SafetyMonitor started")
    
    def stop(self):
        """Останавливает монитор"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.check_interval + 5)
        logger.info("🔓 SafetyMonitor stopped")
    
    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self.running:
            try:
                self._check_safety()
                self.stats['checks_performed'] += 1
                self.last_check_time = time.time()
            except Exception as e:
                logger.error(f"❌ Error in safety check: {e}")
            
            # Ожидание с возможностью прерывания
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _check_safety(self):
        """Проверяет состояние безопасности"""
        # Шаг 1: Проверка газа
        gas_flowing = self._is_gas_flowing()
        if not gas_flowing:
            logger.debug("✅ Gas is not flowing, safe")
            return
        
        logger.debug("🔥 Gas is flowing, checking person...")
        self.stats['gas_detected'] += 1
        
        # Шаг 2: Проверка режима запуска
        if self._is_startup_mode():
            self._handle_startup_mode()
            return
        
        # Шаг 3: Проверка присутствия человека
        self._check_person_presence()
    
    def _is_gas_flowing(self) -> bool:
        """
        Проверяет, идет ли газ
        
        Returns:
            bool: True если газ идет
        """
        gas_status = RedisManager.get_key(config.REDIS_KEYS['gas_flow'])
        return gas_status == '1'
    
    def _is_startup_mode(self) -> bool:
        """
        Проверяет, находится ли система в режиме запуска
        
        Returns:
            bool: True если система в режиме запуска
        """
        return RedisManager.key_exists(config.REDIS_KEYS['startup'])
    
    def _handle_startup_mode(self):
        """Обрабатывает режим запуска системы"""
        self.stats['startup_mode_checks'] += 1
        logger.debug("⏳ System in startup mode, waiting...")
        
        # Проверяем, не появился ли человек
        time_since_seen = RedisManager.get_time_since(config.REDIS_KEYS['human_last_seen'])
        
        if time_since_seen is not None and time_since_seen < config.STARTUP_PERSON_TIMEOUT:
            # Человек обнаружен во время запуска
            RedisManager.delete_key(config.REDIS_KEYS['startup'])
            RedisManager.set_key(config.REDIS_KEYS['human_last_seen'], str(time.time()))
            logger.info("👤 Person detected during startup - startup mode cleared")
            
            # Вызываем колбэки
            for callback in self.on_person_detected_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        else:
            # Продолжаем ждать
            remaining = RedisManager.get_time_since(config.REDIS_KEYS['startup'])
            if remaining:
                remaining = max(0, config.STARTUP_DURATION - remaining)
                if int(remaining) % 30 == 0:  # Логируем каждые 30 секунд
                    logger.info(f"⏳ Startup mode: {int(remaining)}s remaining")
    
    def _check_person_presence(self):
        """Проверяет присутствие человека"""
        time_since_seen = RedisManager.get_time_since(config.REDIS_KEYS['human_last_seen'])
        
        if time_since_seen is None:
            # Человек никогда не был обнаружен
            self.stats['person_missing'] += 1
            logger.warning("⚠️ Person never detected")
            self._send_alert_if_needed()
            return
        
        if time_since_seen < 60:  # Видели меньше минуты назад
            self.stats['person_detected'] += 1
            logger.debug(f"👤 Person detected {int(time_since_seen)}s ago")
            
            # Вызываем колбэки
            for callback in self.on_person_detected_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            return
        
        if time_since_seen >= config.PERSON_ABSENCE_THRESHOLD:
            # Человек отсутствует дольше порога
            self.stats['person_missing'] += 1
            minutes = int(time_since_seen / 60)
            logger.warning(f"⚠️ Person missing for {minutes} minutes!")
            self._send_alert_if_needed()
            
            # Вызываем колбэки
            for callback in self.on_person_missing_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        else:
            # Человек отсутствует, но еще не критично
            minutes = int(time_since_seen / 60)
            logger.debug(f"👤 Person missing for {minutes} minutes")
    
    def _send_alert_if_needed(self):
        """Отправляет тревогу, если не было отправлено недавно"""
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
        
        # Отправляем тревогу
        logger.info("🚨 SENDING ALERT!")
        success = telegram_bot.send_alert('gas_alert')
        
        if success:
            self.alert_count += 1
            self.stats['alerts_sent'] += 1
            
            # Устанавливаем флаг тревоги
            RedisManager.set_key(config.REDIS_KEYS['alert_triggered'], '1')
            RedisManager.set_key(config.REDIS_KEYS['alert_cooldown'], '1', config.ALERT_COOLDOWN)
            
            logger.info(f"✅ Alert sent successfully (total: {self.alert_count})")
            
            # Вызываем колбэки
            for callback in self.on_alert_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        else:
            logger.error("❌ Failed to send alert")
    
    def reset_alert(self):
        """Сбрасывает состояние тревоги"""
        RedisManager.delete_key(config.REDIS_KEYS['alert_triggered'])
        RedisManager.delete_key(config.REDIS_KEYS['alert_cooldown'])
        RedisManager.set_key(config.REDIS_KEYS['human_last_seen'], str(time.time()))
        logger.info("🔄 Alert reset manually")
    
    def get_status(self) -> Dict:
        """
        Получает текущий статус монитора
        
        Returns:
            dict: Статус монитора
        """
        gas_flowing = self._is_gas_flowing()
        startup_mode = self._is_startup_mode()
        time_since_seen = RedisManager.get_time_since(config.REDIS_KEYS['human_last_seen'])
        alert_active = RedisManager.key_exists(config.REDIS_KEYS['alert_triggered'])
        cooldown = RedisManager.key_exists(config.REDIS_KEYS['alert_cooldown'])
        
        return {
            'running': self.running,
            'last_check': self.last_check_time,
            'last_check_str': datetime.fromtimestamp(self.last_check_time).strftime('%H:%M:%S') if self.last_check_time else 'Never',
            'gas_flowing': gas_flowing,
            'startup_mode': startup_mode,
            'person': {
                'last_seen': time_since_seen,
                'last_seen_str': f"{int(time_since_seen)}s ago" if time_since_seen else 'Never',
                'is_present': time_since_seen is not None and time_since_seen < 60
            },
            'alert': {
                'active': alert_active,
                'cooldown': cooldown,
                'count': self.alert_count
            },
            'stats': self.stats,
            'check_interval': self.check_interval
        }
    
    def add_on_alert_callback(self, callback: Callable):
        """Добавляет callback при тревоге"""
        self.on_alert_callbacks.append(callback)
        logger.debug(f"Added alert callback: {callback.__name__}")
    
    def add_on_person_detected_callback(self, callback: Callable):
        """Добавляет callback при обнаружении человека"""
        self.on_person_detected_callbacks.append(callback)
        logger.debug(f"Added person detected callback: {callback.__name__}")
    
    def add_on_person_missing_callback(self, callback: Callable):
        """Добавляет callback при отсутствии человека"""
        self.on_person_missing_callbacks.append(callback)
        logger.debug(f"Added person missing callback: {callback.__name__}")
    
    def get_stats(self) -> Dict:
        """Получает статистику"""
        return {
            **self.stats,
            'total_alerts': self.alert_count,
            'uptime': time.time() - self.last_check_time if self.last_check_time else 0
        }
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self.stats = {
            'checks_performed': 0,
            'gas_detected': 0,
            'alerts_sent': 0,
            'startup_mode_checks': 0,
            'person_detected': 0,
            'person_missing': 0
        }
        self.alert_count = 0
        logger.info("📊 Stats reset")