import time
import threading
from typing import Callable
from app.protocol_models import KeyValueStore, Notifier
from app.domain.models import Thresholds
from meter_watch_shared.config import RedisKeys

import logging

logger = logging.getLogger(__name__)

class SafetyMonitor:
    """Safety monitoring: gas + person"""

    def __init__(
        self,
        store: KeyValueStore,
        notifier: Notifier,
        keys: RedisKeys,
        thresholds: Thresholds,
        check_interval: int,
        clock: Callable[[], float] = time.time,
    ):
        self._store = store
        self._notifier = notifier
        self._clock = clock
        self._keys = keys
        self._thresholds = thresholds

        self.check_interval = check_interval
        self.running = False
        self.thread: threading.Thread | None = None
        self.alert_count = 0
        self.last_check_time = 0

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
                self.check_once()
            except Exception as e:
                logger.error(f"Check error: {e}")

            # Wait interval seconds
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)

    def check_once(self):
        """Main check"""
        # 1. Gas is not flowing - safe
        if self._store.get(self._keys.gas_flow) != '1':
            return

        # 2. Startup mode - waiting
        if self._store.exists(self._keys.startup):
            self._handle_startup()
            return

        # 3. Check person
        self._check_person()

    def _handle_startup(self):
        """Handle startup mode"""
        time_since_seen = self._store.seconds_since(self._keys.human_last_seen)

        # If person appeared - exit startup mode
        if time_since_seen is not None and time_since_seen < self._thresholds.startup_person_timeout:
            time_str = time.strftime("%H:%M %d:%m:%Y", time.localtime(self._clock()))

            self._store.delete(self._keys.startup)
            self._store.set(self._keys.human_last_seen, str(self._clock()))
            self._store.set(self._keys.human_last_seen_str, time_str)
            logger.info("👤 Person detected - startup mode cleared")

    def _check_person(self):
        """Check person presence"""
        time_since_seen = self._store.seconds_since(self._keys.human_last_seen)

        # Person is present (seen less than PERSON_IS_ACTIVE_THRESHOLD)
        if time_since_seen is not None and time_since_seen < self._thresholds.person_is_active_threshold:

            # Clear alert if it was active
            if self._store.exists(self._keys.alert_triggered):
                self._store.delete(self._keys.alert_triggered)
                logger.info("✅ Alert cleared - person returned")
            return

        # Person is missing
        if time_since_seen is None or time_since_seen >= self._thresholds.person_absence_threshold:
            minutes = int(time_since_seen / 60)
            logger.warning(f"⚠️ Person missing for {minutes} minutes!")
            self._send_alert()
        else:
            # Person is missing but not critical yet
            minutes = int(time_since_seen / 60)
            logger.debug(f"👤 Person missing for {minutes} minutes")

    def _send_alert(self):
        """Send alert"""
        # Check cooldown
        cooldown_key = self._keys.alert_cooldown
        if self._store.exists(cooldown_key):
            remaining = self._store.seconds_since(cooldown_key)
            if remaining:
                logger.debug(f"⏳ Alert cooldown: {int(remaining)}s remaining")
            return

        # Check if alert is already active
        if self._store.exists(self._keys.alert_triggered):
            logger.debug("⚠️ Alert already triggered")
            return

        # Check cooldown and active alert
        if (self._store.exists(self._keys.alert_cooldown) or
                self._store.exists(self._keys.alert_triggered)):
            return

        # Send
        logger.info("🚨 SENDING ALERT!")
        success = self._notifier.send_alert('gas_alert')

        if success:
            self.alert_count += 1

            # Set alert flags
            self._store.set(self._keys.alert_triggered, '1')
            self._store.set(self._keys.alert_cooldown, '1', self._thresholds.alert_cooldown)

            logger.info(f"✅ Alert sent successfully (total: {self.alert_count})")
        else:
            logger.error("❌ Failed to send alert")