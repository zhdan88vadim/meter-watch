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
        """Регистрирует обработчики команд"""
        self.command_handlers = {
            '/start': self._handle_start,
            '/status': self._handle_status,
            '/silence_alert': self._handle_silence_alert,
            '/reset': self._handle_reset,
            '/help': self._handle_help
        }
    
    def send_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """Отправляет сообщение в Telegram"""
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
        """Отправляет системное уведомление"""
        if alert_type == 'startup':
            message = self._format_startup_message(data)
        elif alert_type == 'gas_alert':
            message = self._format_gas_alert_message()
        else:
            message = str(data)
        
        return self.send_message(message)
    
    def _format_startup_message(self, data: dict) -> str:
        return (
            f"🔄 **Система перезагружена**\n"
            f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}\n"
            f"⏳ Режим ожидания: {config.STARTUP_DURATION//60} минут\n"
            f"📡 Сервис активен\n"
            f"🤖 Используйте /help для управления"
        )
    
    def _format_gas_alert_message(self) -> str:
        gas_status = RedisManager.get_key(config.REDIS_KEYS['gas_flow'])
        last_seen = RedisManager.get_key(config.REDIS_KEYS['human_last_seen'])
        
        if last_seen:
            last_seen_time = datetime.fromtimestamp(float(last_seen)).strftime('%H:%M:%S')
            minutes_ago = int((time.time() - float(last_seen)) / 60)
        else:
            last_seen_time = "Никогда"
            minutes_ago = "более 10"
        
        return (
            f"⚠️ **ВНИМАНИЕ! ОБНАРУЖЕНА УТЕЧКА ГАЗА!** ⚠️\n\n"
            f"🔥 **Газ идет**: {'ДА' if gas_status == '1' else 'НЕТ'}\n"
            f"👤 **Человек не обнаружен**: {minutes_ago} минут\n"
            f"⏰ **Последнее обнаружение**: {last_seen_time}\n"
            f"🕐 **Время тревоги**: {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"🚨 **НЕМЕДЛЕННО ПРОВЕРЬТЕ ПОМЕЩЕНИЕ!**\n\n"
            f"🤖 Для управления используйте:\n"
            f"/silence - отключить звук\n"
            f"/reset - сбросить тревогу\n"
            f"/status - текущий статус"
        )
    
    # === Команды бота ===
    
    def _handle_start(self, args=None) -> str:
        return (
            f"👋 **Добро пожаловать в систему безопасности!**\n\n"
            f"🤖 Доступные команды:\n"
            f"/status - статус системы\n"
            f"/silence - отключить звук\n"
            f"/reset - сбросить тревогу\n"
            f"/help - помощь"
        )
    
    def _handle_status(self, args=None) -> str:
        gas_status = RedisManager.get_key(config.REDIS_KEYS['gas_flow'])
        last_seen = RedisManager.get_key(config.REDIS_KEYS['human_last_seen'])
        alert_active = RedisManager.key_exists(config.REDIS_KEYS['alert_triggered'])
        
        status = f"📊 **Статус системы**\n\n"
        status += f"🔥 Газ: {'🟢 Идет' if gas_status == '1' else '🔴 Не идет'}\n"
        
        if last_seen:
            last_seen_time = datetime.fromtimestamp(float(last_seen)).strftime('%H:%M:%S')
            minutes_ago = int((time.time() - float(last_seen)) / 60)
            status += f"👤 Человек: {'🟢 Есть' if minutes_ago < 5 else '🔴 Нет'}\n"
            status += f"⏰ Последний раз: {last_seen_time} ({minutes_ago} мин назад)\n"
        else:
            status += f"👤 Человек: ⚪ Не обнаружен\n"
        
        status += f"🚨 Тревога: {'🔴 Активна' if alert_active else '🟢 Нет'}\n"
        status += f"🕐 Время: {datetime.now().strftime('%H:%M:%S')}"
        
        return status
    
    def _handle_silence_alert(self, args=None) -> str:
        RedisManager.set_key(config.REDIS_KEYS['alert_cooldown'], '1', config.ALERT_COOLDOWN)
        RedisManager.delete_key(config.REDIS_KEYS['alert_triggered'])
        return "🔇 Звук отключен на 10 минут. Тревога сброшена."
    
    def _handle_reset(self, args=None) -> str:
        self.state_manager.reset_alert()
        return "🔄 Система сброшена. Тревога деактивирована."
    
    def _handle_help(self, args=None) -> str:
        return (
            f"🤖 **Доступные команды:**\n\n"
            f"/start - приветствие\n"
            f"/status - текущий статус системы\n"
            f"/silence - отключить звук и сбросить тревогу\n"
            f"/reset - сбросить состояние системы\n"
            f"/help - эта справка"
        )
    
    def start(self):
        """Запускает бота в отдельном потоке"""
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
        """Проверяет новые сообщения"""
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
        """Обрабатывает входящее сообщение"""
        if 'message' not in update:
            return
        
        message = update['message']
        if 'text' not in message:
            return
        
        text = message['text']
        chat_id = str(message['chat']['id'])
        
        # Проверяем, что сообщение от правильного chat_id
        if chat_id != self.chat_id:
            logger.warning(f"⚠️ Message from unauthorized chat: {chat_id}")
            return
        
        # Обрабатываем команду
        for command, handler in self.command_handlers.items():
            if text.startswith(command):
                response = handler()
                self.send_message(response)
                break

telegram_bot = TelegramBot()