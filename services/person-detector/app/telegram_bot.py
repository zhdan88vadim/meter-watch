from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

import requests

from app.protocol_models import KeyValueStore

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Command handling and long-polling. poll_once() is a single iteration and
    process_update() is pure enough to be called directly in tests.
    """

    def __init__(self, cfg, store: KeyValueStore) -> None:
        self._cfg = cfg
        self._store = store
        self.last_update_id = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Callable[[], str]] = {
            "/start": self._handle_start,
            "/status": self._handle_status,
            "/silence": self._handle_silence,
            "/silence_alert": self._handle_silence,
            "/reset": self._handle_reset,
            "/help": self._handle_help,
        }

    # ---- lifecycle ----
    def start(self) -> None:
        if not self._cfg.TELEGRAM_BOT_TOKEN or not self._cfg.TELEGRAM_CHAT_ID:
            logger.warning("Telegram bot not configured")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                logger.error("Telegram poll error: %s", exc)
                self._stop_event.wait(5)

    # ---- testable core ----
    def poll_once(self) -> None:
        url = f"https://api.telegram.org/bot{self._cfg.TELEGRAM_BOT_TOKEN}/getUpdates"
        params = {"offset": self.last_update_id + 1, "timeout": 30}
        response = requests.get(url, params=params, timeout=35)
        if response.status_code != 200:
            logger.error("Bot poll status: %s", response.status_code)
            return
        for update in response.json().get("result", []):
            self.process_update(update)
            self.last_update_id = update["update_id"]

    def process_update(self, update: dict) -> Optional[str]:
        message = update.get("message")
        if not message or "text" not in message:
            return None
        chat_id = str(message["chat"]["id"])
        if chat_id != self._cfg.TELEGRAM_CHAT_ID:
            logger.warning("Unauthorized chat: %s", chat_id)
            return None
        text = message["text"].strip()
        command = text.split()[0] if text else ""
        handler = self._handlers.get(command)
        if not handler:
            return None
        reply = handler()
        self._send(reply)
        return reply

    def _send(self, text: str) -> None:
        if not text:
            return
        try:
            url = self._cfg.TELEGRAM_API_URL.format(self._cfg.TELEGRAM_BOT_TOKEN)
            requests.post(
                url,
                data={
                    "chat_id": self._cfg.TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=self._cfg.TELEGRAM_TIMEOUT,
            )
        except Exception as exc:
            logger.error("Telegram reply error: %s", exc)

    # ---- handlers ----
    def _handle_start(self) -> str:
        return (
            "👋 **Welcome!**\n\n"
            "/status - system status\n"
            "/silence - mute sound\n"
            "/reset - reset alert\n"
            "/help - help"
        )

    def _handle_status(self) -> str:
        k = self._cfg.REDIS_KEYS
        gas = self._store.get(k["gas_flow"])
        last = self._store.get(k["human_last_seen"])
        alert = self._store.exists(k["alert_triggered"])
        lines = ["📊 **System status**\n"]
        lines.append(f"🔥 Gas: {'🟢 Flowing' if gas == '1' else '🔴 Not flowing'}")
        if last:
            secs = self._store.seconds_since(k["human_last_seen"])
            mins = int(secs / 60) if secs is not None else 0
            lines.append(f"👤 Person last seen: {mins} min ago")
        else:
            lines.append("👤 Person: ⚪ Not detected")
        lines.append(f"🚨 Alert: {'🔴 Active' if alert else '🟢 None'}")
        return "\n".join(lines)

    def _handle_silence(self) -> str:
        k = self._cfg.REDIS_KEYS
        self._store.set(k["alert_cooldown"], "1", ttl=self._cfg.ALERT_COOLDOWN)
        self._store.delete(k["alert_triggered"])
        return "🔇 Sound muted. Alert reset."

    def _handle_reset(self) -> str:
        k = self._cfg.REDIS_KEYS
        time_str = time.strftime("%H:%M %d:%m:%Y", time.localtime(time.time()))

        self._store.delete(k["alert_triggered"])
        self._store.delete(k["alert_cooldown"])
        self._store.set(k["human_last_seen"], str(time.time()))
        self._store.set(k['human_last_seen_str'], time_str)
        return "🔄 System reset."

    def _handle_help(self) -> str:
        return "🤖 **Commands:**\n/start /status /silence /reset /help"
