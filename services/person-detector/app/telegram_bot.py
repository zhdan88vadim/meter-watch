import time
import requests
import threading
from datetime import datetime
from typing import Optional
from meter_watch_shared.config import config
from meter_watch_shared.redis_manager import RedisManager
from app.state_manager import StateManager
import logging

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.state_manager = StateManager()
        self.last_update_id = 0
        self.running = False
        self.thread = None
        self.command_handlers = {}
        self._register_commands()
    
    def _register_commands(self):
        """Registers command handlers"""
        self.command_handlers = {
            '/start': self._handle_start,
            '/status': self._handle_status,
            '/silence_alert': self._handle_silence_alert,
            '/reset': self._handle_reset,
            '/help': self._handle_help
        }
    
    def send_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """Sends a message to Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.warning("⚠️ Telegram credentials not configured")
            return False
        
        try:
            url = config.TELEGRAM_API_URL.format(self.bot_token)
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            response = requests.post(url, data=data, timeout=config.TELEGRAM_TIMEOUT)
            
            if response.status_code == 200:
                logger.info("📨 Message sent to Telegram")
                return True
            else:
                logger.error(f"❌ Error sending message: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            return False
    
    def send_alert(self, alert_type: str, data: Optional[dict] = None) -> bool:
        """Sends a system notification"""
        if alert_type == 'startup':
            message = self._format_startup_message(data)
        elif alert_type == 'gas_alert':
            message = self._format_gas_alert_message()
        else:
            message = str(data)
        
        return self.send_message(message)
    
    def _format_startup_message(self, data: dict) -> str:
        return (
            f"🔄 **System restarted**\n"
            f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n"
            f"⏳ Waiting mode: {config.STARTUP_DURATION//60} minutes\n"
            f"📡 Service active\n"
            f"🤖 Use /help for management"
        )
    
    def _format_gas_alert_message(self) -> str:
        gas_status = RedisManager.get_key(config.REDIS_KEYS['gas_flow'])
        last_seen = RedisManager.get_key(config.REDIS_KEYS['human_last_seen'])
        
        if last_seen:
            last_seen_time = datetime.fromtimestamp(float(last_seen)).strftime('%H:%M:%S')
            minutes_ago = int((time.time() - float(last_seen)) / 60)
        else:
            last_seen_time = "Never"
            minutes_ago = "more than 10"
        
        return (
            f"⚠️ **WARNING! GAS LEAK DETECTED!** ⚠️\n\n"
            f"🔥 **Gas flowing**: {'YES' if gas_status == '1' else 'NO'}\n"
            f"👤 **Person not detected**: {minutes_ago} minutes\n"
            f"⏰ **Last detection**: {last_seen_time}\n"
            f"🕐 **Alert time**: {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"🚨 **IMMEDIATELY CHECK THE ROOM!**\n\n"
            f"🤖 Use commands:\n"
            f"/silence - mute sound\n"
            f"/reset - reset alert\n"
            f"/status - current status"
        )
    
    # === Bot commands ===
    
    def _handle_start(self, args=None) -> str:
        return (
            f"👋 **Welcome to the security system!**\n\n"
            f"🤖 Available commands:\n"
            f"/status - system status\n"
            f"/silence - mute sound\n"
            f"/reset - reset alert\n"
            f"/help - help"
        )
    
    def _handle_status(self, args=None) -> str:
        gas_status = RedisManager.get_key(config.REDIS_KEYS['gas_flow'])
        last_seen = RedisManager.get_key(config.REDIS_KEYS['human_last_seen'])
        alert_active = RedisManager.key_exists(config.REDIS_KEYS['alert_triggered'])
        
        status = f"📊 **System status**\n\n"
        status += f"🔥 Gas: {'🟢 Flowing' if gas_status == '1' else '🔴 Not flowing'}\n"
        
        if last_seen:
            last_seen_time = datetime.fromtimestamp(float(last_seen)).strftime('%H:%M:%S')
            minutes_ago = int((time.time() - float(last_seen)) / 60)
            status += f"👤 Person: {'🟢 Present' if minutes_ago < 5 else '🔴 Absent'}\n"
            status += f"⏰ Last seen: {last_seen_time} ({minutes_ago} min ago)\n"
        else:
            status += f"👤 Person: ⚪ Not detected\n"
        
        status += f"🚨 Alert: {'🔴 Active' if alert_active else '🟢 None'}\n"
        status += f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}"
        
        return status
    
    def _handle_silence_alert(self, args=None) -> str:
        RedisManager.set_key(config.REDIS_KEYS['alert_cooldown'], '1', config.ALERT_COOLDOWN)
        RedisManager.delete_key(config.REDIS_KEYS['alert_triggered'])
        return "🔇 Sound muted for 10 minutes. Alert reset."
    
    def _handle_reset(self, args=None) -> str:
        self.state_manager.reset_alert()
        return "🔄 System reset. Alert deactivated."
    
    def _handle_help(self, args=None) -> str:
        return (
            f"🤖 **Available commands:**\n\n"
            f"/start - welcome\n"
            f"/status - current system status\n"
            f"/silence - mute sound and reset alert\n"
            f"/reset - reset system state\n"
            f"/help - this help"
        )
    
    def start(self):
        """Starts the bot in a separate thread"""
        if not self.bot_token or not self.chat_id:
            logger.warning("⚠️ Telegram bot not configured")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._poll_messages, daemon=True)
        self.thread.start()
        logger.info("🤖 Telegram bot started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _poll_messages(self):
        """Checks for new messages"""
        while self.running:
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
                params = {
                    'offset': self.last_update_id + 1,
                    'timeout': 30
                }
                response = requests.get(url, params=params, timeout=35)
                
                if response.status_code == 200:
                    updates = response.json().get('result', [])
                    for update in updates:
                        self._process_update(update)
                        self.last_update_id = update['update_id']
                else:
                    logger.error(f"❌ Bot poll error: {response.status_code}")
                    time.sleep(5)
                    
            except Exception as e:
                logger.error(f"❌ Bot poll error: {e}")
                time.sleep(5)
    
    def _process_update(self, update):
        """Processes incoming message"""
        if 'message' not in update:
            return
        
        message = update['message']
        if 'text' not in message:
            return
        
        text = message['text']
        chat_id = str(message['chat']['id'])
        
        # Check that message is from the correct chat_id
        if chat_id != self.chat_id:
            logger.warning(f"⚠️ Message from unauthorized chat: {chat_id}")
            return
        
        # Process command
        for command, handler in self.command_handlers.items():
            if text.startswith(command):
                response = handler()
                self.send_message(response)
                break

telegram_bot = TelegramBot()