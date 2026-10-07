"""
Telegram notifier. Wraps requests; satisfies the Notifier protocol.
Message formatting is pure and can be tested without network access.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import requests

from meter_watch_shared.config import config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, cfg=config) -> None:
        self._cfg = cfg

    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        if not self._cfg.TELEGRAM_BOT_TOKEN or not self._cfg.TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials not configured")
            return False
        try:
            url = self._cfg.TELEGRAM_API_URL.format(self._cfg.TELEGRAM_BOT_TOKEN)
            response = requests.post(
                url,
                data={
                    "chat_id": self._cfg.TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": parse_mode,
                },
                timeout=self._cfg.TELEGRAM_TIMEOUT,
            )
            if response.status_code == 200:
                logger.info("Message sent to Telegram")
                return True
            logger.error("Telegram send failed: %s", response.status_code)
            return False
        except Exception as exc:
            logger.error("Telegram send error: %s", exc)
            return False

    def send_alert(self, alert_type: str, data: Optional[dict] = None) -> bool:
        if alert_type == "startup":
            message = format_startup_message(self._cfg)
        elif alert_type == "gas_alert":
            message = format_gas_alert_message(self._cfg, self._read_state)
        else:
            message = str(data)
        return self.send_message(message)

    def _read_state(self, key: str) -> Optional[str]:
        # Deliberately minimal: only used to enrich the alert message.
        from meter_watch_shared.redis_manager import RedisManager
        return RedisManager.get_key(key)


def format_startup_message(cfg) -> str:
    return (
        f"🔄 **System restarted**\n"
        f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n"
        f"⏳ Waiting mode: {cfg.STARTUP_DURATION // 60} minutes\n"
        f"📡 Service active\n"
        f"🤖 Use /help for management"
    )


def format_gas_alert_message(cfg, read_state=None) -> str:
    gas_status = read_state(cfg.REDIS_KEYS["gas_flow"]) if read_state else None
    last_seen = read_state(cfg.REDIS_KEYS["human_last_seen"]) if read_state else None
    return (
        f"⚠️ **WARNING! GAS LEAK DETECTED!** ⚠️\n\n"
        f"🔥 **Gas flowing**: {'YES' if gas_status == '1' else 'NO'}\n"
        f"👤 **Last seen**: {last_seen or 'Never'}\n"
        f"🕐 **Alert time**: {datetime.now().strftime('%H:%M:%S')}\n\n"
        f"🚨 **IMMEDIATELY CHECK THE ROOM!**\n\n"
        f"🤖 Use /silence, /reset, /status"
    )
