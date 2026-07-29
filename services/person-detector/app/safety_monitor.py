import time
import threading
from datetime import datetime
from typing import Optional, Dict, Callable, List
from meter_watch_shared.config import config
from meter_watch_shared.redis_manager import RedisManager
from app.telegram_bot import telegram_bot
import logging

logger = logging.getLogger(__name__)

class SafetyMonitor:
    """Safety monitoring: gas + person"""
    
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
            self.thread.join(timeout=self.check_interval)
    
    def _monitor_loop(self):
        while self.running:
            try:
                self._check()
            except Exception as e:
                logger.error(f"Check error: {e}")
            
            # Wait interval seconds
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _check(self):
        """Main check"""
        # 1. Gas is not flowing - safe
        if RedisManager.get_key(config.REDIS_KEYS['gas_flow']) != '1':
            return
        
        # 2. Startup mode - waiting
        if RedisManager.key_exists(config.REDIS_KEYS['startup']):
            self._handle_startup()
            return
        
        # 3. Check person
        self._check_person()
    
    def _handle_startup(self):
        """Handle startup mode"""
        time_since_seen = RedisManager.get_time_since(config.REDIS_KEYS['human_last_seen'])
        
        # If person appeared - exit startup mode
        if time_since_seen is not None and time_since_seen < config.STARTUP_PERSON_TIMEOUT:
            time_str = time.strftime("%H:%M %d:%m:%Y", time.localtime(time.time()))

            RedisManager.delete_key(config.REDIS_KEYS['startup'])
            RedisManager.set_key(config.REDIS_KEYS['human_last_seen'], str(time.time()))
            RedisManager.set_key(config.REDIS_KEYS['human_last_seen_str'], time_str)
            logger.info("👤 Person detected - startup mode cleared")
            self._notify(self.on_person_detected_callbacks)
    
    def _check_person(self):
        """Check person presence"""
        time_since_seen = RedisManager.get_time_since(config.REDIS_KEYS['human_last_seen'])
        
        # Person is present (seen less than PERSON_IS_ACTIVE_THRESHOLD)
        if time_since_seen is not None and time_since_seen < config.PERSON_IS_ACTIVE_THRESHOLD:
            self._notify(self.on_person_detected_callbacks)
            # Clear alert if it was active
            if RedisManager.key_exists(config.REDIS_KEYS['alert_triggered']):
                RedisManager.delete_key(config.REDIS_KEYS['alert_triggered'])
                logger.info("✅ Alert cleared - person returned")
            return
        
        # Person is missing
        if time_since_seen is None or time_since_seen >= config.PERSON_ABSENCE_THRESHOLD:
            minutes = int(time_since_seen / 60)
            logger.warning(f"⚠️ Person missing for {minutes} minutes!")
            self._send_alert()
            self._notify(self.on_person_missing_callbacks)
        else:
            # Person is missing but not critical yet
            minutes = int(time_since_seen / 60)
            logger.debug(f"👤 Person missing for {minutes} minutes")

    
    def _send_alert(self):
        """Send alert"""
        # Check cooldown
        cooldown_key = config.REDIS_KEYS['alert_cooldown']
        if RedisManager.key_exists(cooldown_key):
            remaining = RedisManager.get_time_since(cooldown_key)
            if remaining:
                logger.debug(f"⏳ Alert cooldown: {int(remaining)}s remaining")
            return
        
        # Check if alert is already active
        if RedisManager.key_exists(config.REDIS_KEYS['alert_triggered']):
            logger.debug("⚠️ Alert already triggered")
            return
        
        # Check cooldown and active alert
        if (RedisManager.key_exists(config.REDIS_KEYS['alert_cooldown']) or 
            RedisManager.key_exists(config.REDIS_KEYS['alert_triggered'])):
            return
        
        # Send
        logger.info("🚨 SENDING ALERT!")
        success = telegram_bot.send_alert('gas_alert')
        
        if success:
            self.alert_count += 1
            
            # Set alert flags
            RedisManager.set_key(config.REDIS_KEYS['alert_triggered'], '1')
            RedisManager.set_key(config.REDIS_KEYS['alert_cooldown'], '1', config.ALERT_COOLDOWN)
            
            logger.info(f"✅ Alert sent successfully (total: {self.alert_count})")
            
            # Call callbacks
            self._notify(self.on_alert_callbacks)
        else:
            logger.error("❌ Failed to send alert")            
    
    def _notify(self, callbacks: List[Callable]):
        """Call callbacks"""
        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    # === Public methods ===

    def add_on_alert_callback(self, callback: Callable):
        self.on_alert_callbacks.append(callback)
    
    def add_on_person_detected_callback(self, callback: Callable):
        self.on_person_detected_callbacks.append(callback)
    
    def add_on_person_missing_callback(self, callback: Callable):
        self.on_person_missing_callbacks.append(callback)