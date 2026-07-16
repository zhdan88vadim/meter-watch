import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import redis
import json
import threading
import numpy as np
from collections import deque
import os
import logging
from typing import Optional, Dict, Set, List, Any, Union
import requests

# ==================== CONSTANTS ====================
class Config:
    # Redis
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
    REDIS_DB = 0
    REDIS_TIMEOUT = 5
    
    # System
    STARTUP_DURATION = 300  # 5 minutes in seconds
    PERSON_ABSENCE_THRESHOLD = 600  # 10 minutes in seconds
    ALERT_COOLDOWN = 600  # 10 minutes in seconds
    CHECK_INTERVAL = 30  # seconds between safety checks
    PERSON_EXPIRE_TIME = 3600  # 1 hour in seconds
    RECORDING_EXPIRE_TIME = 86400  # 24 hours in seconds
    STARTUP_PERSON_TIMEOUT = 60  # seconds to consider person seen during startup
    
    # Video
    DEFAULT_FPS = 30
    DEFAULT_FRAME_WIDTH = 640
    DEFAULT_FRAME_HEIGHT = 480
    BUFFER_SECONDS = 4
    POST_ROLL_SECONDS = 4
    
    # Redis Keys
    REDIS_KEYS = {
        'startup': 'system:startup:timestamp',
        'gas_flow': 'meter:gas:flow',
        'human_last_seen': 'human:detect:last_seen',
        'alert_cooldown': 'alert:telegram:cooldown',
        'detection_events': 'detection:events',
        'detection_history': 'detection:history',
        'active_people': 'active:people',
        'recording_prefix': 'recording:',
        'person_prefix': 'person:'
    }
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    TELEGRAM_API_URL = 'https://api.telegram.org/bot{}/sendMessage'
    TELEGRAM_TIMEOUT = 5

# ==================== LOGGING SETUP ====================
def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('person_detector.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ==================== REDIS SETUP ====================
def get_redis_connection():
    """Создание подключения к Redis"""
    return redis.Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        db=Config.REDIS_DB,
        password=Config.REDIS_PASSWORD if Config.REDIS_PASSWORD else None,
        decode_responses=True,
        socket_connect_timeout=Config.REDIS_TIMEOUT,
        socket_timeout=Config.REDIS_TIMEOUT
    )

r = get_redis_connection()

# ==================== HELPER FUNCTIONS ====================
def convert_to_serializable(obj: Any) -> Any:
    """Рекурсивно преобразует numpy типы в стандартные Python типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def safe_int(value: Any) -> int:
    """Безопасное преобразование в int"""
    if isinstance(value, (np.integer, np.floating)):
        return int(value)
    return int(value) if value is not None else 0

def safe_float(value: Any) -> float:
    """Безопасное преобразование в float"""
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return float(value) if value is not None else 0.0

# ==================== REDIS HELPERS ====================
class RedisManager:
    """Управление операциями с Redis"""
    
    @staticmethod
    def get_key(key: str) -> Optional[str]:
        """Получить значение ключа"""
        return r.get(key)
    
    @staticmethod
    def set_key(key: str, value: str, expire: Optional[int] = None):
        """Установить значение ключа"""
        if expire:
            r.setex(key, expire, value)
        else:
            r.set(key, value)
    
    @staticmethod
    def delete_key(key: str):
        """Удалить ключ"""
        r.delete(key)
    
    @staticmethod
    def key_exists(key: str) -> bool:
        """Проверить существование ключа"""
        return r.exists(key) > 0
    
    @staticmethod
    def get_timestamp_key(key: str) -> Optional[float]:
        """Получить timestamp из ключа"""
        value = r.get(key)
        return float(value) if value else None
    
    @staticmethod
    def set_timestamp_key(key: str, expire: Optional[int] = None):
        """Установить текущий timestamp в ключ"""
        current_time = time.time()
        if expire:
            r.setex(key, expire, str(current_time))
        else:
            r.set(key, str(current_time))
        return current_time
    
    @staticmethod
    def get_time_since(key: str) -> Optional[float]:
        """Получить время, прошедшее с момента установки ключа"""
        timestamp = RedisManager.get_timestamp_key(key)
        if timestamp is None:
            return None
        return time.time() - timestamp

# ==================== SYSTEM STATE HELPERS ====================
def is_system_in_startup_mode() -> bool:
    """Проверяет, находится ли система в режиме запуска"""
    return RedisManager.key_exists(Config.REDIS_KEYS['startup'])

def mark_system_startup():
    """Отмечает время запуска системы"""
    startup_time = RedisManager.set_timestamp_key(
        Config.REDIS_KEYS['startup'], 
        Config.STARTUP_DURATION
    )
    logger.info(f"🔄 Система запущена. Режим ожидания активирован на {Config.STARTUP_DURATION//60} минут")
    send_telegram_notification('startup', startup_time)

def clear_startup_mode():
    """Принудительно очищает режим запуска"""
    RedisManager.delete_key(Config.REDIS_KEYS['startup'])
    logger.info("✅ Режим ожидания завершен досрочно")

def update_human_last_seen():
    """Обновляет время последнего обнаружения человека"""
    RedisManager.set_key(Config.REDIS_KEYS['human_last_seen'], str(time.time()))

def get_human_absence_duration() -> Optional[float]:
    """Получает длительность отсутствия человека"""
    return RedisManager.get_time_since(Config.REDIS_KEYS['human_last_seen'])

def is_gas_flowing() -> bool:
    """Проверяет, идет ли газ"""
    return RedisManager.get_key(Config.REDIS_KEYS['gas_flow']) == '1'

# ==================== TELEGRAM HELPERS ====================
def send_telegram_notification(notification_type: str, data: Any = None) -> bool:
    """Отправляет уведомление в Telegram"""
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not configured")
        return False
    
    try:
        messages = {
            'startup': lambda: (
                f"🔄 **Система перезагружена**\n"
                f"⏰ Время запуска: {datetime.fromtimestamp(data).strftime('%H:%M:%S')}\n"
                f"⏳ Режим ожидания: {Config.STARTUP_DURATION//60} минут\n"
                f"📡 Сервис детекции человека активен"
            ),
            'gas_alert': lambda: _build_gas_alert_message()
        }
        
        message = messages.get(notification_type, lambda: str(data))()
        return _send_telegram_message(message)
    except Exception as e:
        logger.error(f"❌ Ошибка отправки уведомления: {e}")
        return False

def _build_gas_alert_message() -> str:
    """Формирует сообщение о тревоге"""
    gas_status = RedisManager.get_key(Config.REDIS_KEYS['gas_flow'])
    last_seen = RedisManager.get_key(Config.REDIS_KEYS['human_last_seen'])
    
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
        f"🚨 **НЕМЕДЛЕННО ПРОВЕРЬТЕ ПОМЕЩЕНИЕ!**"
    )

def _send_telegram_message(message: str) -> bool:
    """Отправляет сообщение в Telegram"""
    try:
        url = Config.TELEGRAM_API_URL.format(Config.TELEGRAM_BOT_TOKEN)
        data = {
            'chat_id': Config.TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, data=data, timeout=Config.TELEGRAM_TIMEOUT)
        
        if response.status_code == 200:
            logger.info("📨 Уведомление отправлено в Telegram")
            return True
        else:
            logger.error(f"❌ Ошибка отправки: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка отправки: {e}")
        return False

def send_gas_alert() -> bool:
    """Отправляет уведомление о тревоге"""
    # Проверяем cooldown
    if RedisManager.key_exists(Config.REDIS_KEYS['alert_cooldown']):
        logger.info("⏳ Cooldown активен, уведомление не отправлено")
        return False
    
    # Отправляем сообщение
    success = send_telegram_notification('gas_alert')
    
    if success:
        # Устанавливаем cooldown
        RedisManager.set_key(
            Config.REDIS_KEYS['alert_cooldown'], 
            '1', 
            Config.ALERT_COOLDOWN
        )
    
    return success

# ==================== VIDEO BUFFER CLASS ====================
class VideoBuffer:
    """Хранит кадры в циклическом буфере для пре-ролла"""
    
    def __init__(self, buffer_seconds: int = Config.BUFFER_SECONDS, fps: int = Config.DEFAULT_FPS):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_frames = int(buffer_seconds * fps)
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        self.is_recording = False
        self.video_writer = None
        self.current_session_id = None
        self.recording_start_time = None
        
    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Добавляет кадр в буфер"""
        if timestamp is None:
            timestamp = time.time()
        
        self.frames.append(frame.copy())
        self.timestamps.append(timestamp)
        
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
    
    def start_recording(self, session_id: Optional[str] = None) -> Optional[str]:
        """Начинает запись видео"""
        if self.is_recording:
            return None
        
        self.is_recording = True
        self.current_session_id = session_id or f"session_{int(time.time())}"
        self.recording_start_time = time.time()
        
        # Создаем файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}_ID{self.current_session_id}.mp4"
        
        os.makedirs("recordings", exist_ok=True)
        filepath = os.path.join("recordings", filename)
        
        # Получаем размеры кадра
        if self.frames:
            h, w = self.frames[0].shape[:2]
        else:
            h, w = Config.DEFAULT_FRAME_HEIGHT, Config.DEFAULT_FRAME_WIDTH
        
        # Инициализируем видеозапись
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))
        
        # Записываем пре-ролл
        for frame in self.frames:
            self.video_writer.write(frame)
        
        logger.info(f"📹 Started recording: {filename}")
        
        # Сохраняем информацию в Redis
        r.hset(f"{Config.REDIS_KEYS['recording_prefix']}{self.current_session_id}", mapping={
            'filename': filename,
            'start_time': self.recording_start_time,
            'person_id': session_id or 'unknown',
            'pre_roll_frames': len(self.frames)
        })
        
        return filepath
    
    def stop_recording(self):
        """Останавливает запись"""
        if not self.is_recording or not self.video_writer:
            return
        
        self.is_recording = False
        self.video_writer.release()
        self.video_writer = None
        
        duration = time.time() - self.recording_start_time
        logger.info(f"📹 Stopped recording. Duration: {duration:.2f}s")
        
        # Обновляем информацию в Redis
        if self.current_session_id:
            key = f"{Config.REDIS_KEYS['recording_prefix']}{self.current_session_id}"
            r.hset(key, 'duration', duration)
            r.hset(key, 'end_time', time.time())
            r.expire(key, Config.RECORDING_EXPIRE_TIME)
        
        self.current_session_id = None
        self.recording_start_time = None

# ==================== SAFETY MONITOR ====================
class SafetyMonitor:
    """Мониторинг безопасности (газ + человек)"""
    
    def __init__(self, check_interval: int = Config.CHECK_INTERVAL):
        self.check_interval = check_interval
        self.running = False
        self.thread = None
    
    def start(self):
        """Запускает мониторинг"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("🔒 Поток безопасности запущен")
    
    def stop(self):
        """Останавливает мониторинг"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("🔓 Поток безопасности остановлен")
    
    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self.running:
            try:
                self._check_safety()
            except Exception as e:
                logger.error(f"❌ Ошибка в мониторинге: {e}")
            
            # Ожидание с возможностью прерывания
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _check_safety(self):
        """Проверяет безопасность"""
        # Проверяем газ
        if not is_gas_flowing():
            return
        
        logger.debug("🔥 Газ идет. Проверяем человека...")
        
        # Проверяем режим запуска
        if is_system_in_startup_mode():
            self._handle_startup_mode()
            return
        
        # Проверяем человека
        self._check_person_absence()
    
    def _handle_startup_mode(self):
        """Обрабатывает режим запуска"""
        logger.info("⏳ Система в режиме запуска. Ожидаем...")
        
        # Проверяем, не появился ли человек
        time_since_seen = get_human_absence_duration()
        if time_since_seen is not None and time_since_seen < Config.STARTUP_PERSON_TIMEOUT:
            clear_startup_mode()
            update_human_last_seen()
            logger.info("👤 Человек обнаружен. Режим запуска завершен досрочно.")
    
    def _check_person_absence(self):
        """Проверяет отсутствие человека"""
        time_since_seen = get_human_absence_duration()
        
        if time_since_seen is None:
            logger.warning("⚠️ Человек не обнаружен ни разу")
            self._send_alert_if_needed()
            return
        
        if time_since_seen >= Config.PERSON_ABSENCE_THRESHOLD:
            logger.warning(f"⚠️ Человек не обнаружен {int(time_since_seen/60)} минут!")
            self._send_alert_if_needed()
        else:
            logger.debug(f"👤 Человек обнаружен {int(time_since_seen)} секунд назад")
    
    def _send_alert_if_needed(self):
        """Отправляет тревогу при необходимости"""
        if send_gas_alert():
            logger.info("🚨 Тревога отправлена!")

# ==================== PERSON TRACKER ====================
class PersonTracker:
    """Основной класс отслеживания людей"""
    
    def __init__(
        self, 
        source: Union[int, str] = 0,
        buffer_seconds: int = Config.BUFFER_SECONDS,
        post_roll_seconds: int = Config.POST_ROLL_SECONDS
    ):
        self.source = source
        self.post_roll_seconds = post_roll_seconds
        self.entry_times: Dict[int, float] = {}
        self.recording_sessions: Dict[int, Dict] = {}
        
        # Инициализация моделей
        self.model = YOLO('yolov8n.pt')
        
        # Инициализация видео
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or Config.DEFAULT_FPS
        
        # Инициализация буфера и монитора
        self.buffer = VideoBuffer(buffer_seconds, self.fps)
        self.safety_monitor = SafetyMonitor()
        
        # Маркируем запуск
        mark_system_startup()
    
    def _get_track_id(self, track_id: Any) -> int:
        """Преобразует track_id в int"""
        return safe_int(track_id)
    
    def _publish_event(self, event_type: str, track_id: int, data: Optional[Dict] = None):
        """Публикует событие в Redis"""
        event = {
            'type': event_type,
            'track_id': track_id,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'data': convert_to_serializable(data or {})
        }
        
        # Публикуем в канал
        r.publish(Config.REDIS_KEYS['detection_events'], json.dumps(event))
        
        # Сохраняем историю
        r.lpush(Config.REDIS_KEYS['detection_history'], json.dumps(event))
        r.ltrim(Config.REDIS_KEYS['detection_history'], 0, 999)
    
    def _update_person_in_redis(self, track_id: int, bbox: List[int], confidence: float):
        """Обновляет информацию о человеке в Redis"""
        person_key = f"{Config.REDIS_KEYS['person_prefix']}{track_id}"
        current_time = time.time()
        
        r.hset(person_key, 'last_seen', current_time)
        r.hset(person_key, 'bbox', str(bbox))
        r.expire(person_key, Config.PERSON_EXPIRE_TIME)
        
        # Обновляем глобальное время
        update_human_last_seen()
    
    def log_person_entry(self, track_id: int, bbox: List[int], confidence: float):
        """Обрабатывает вход человека"""
        entry_time = time.time()
        track_id = self._get_track_id(track_id)
        
        self.entry_times[track_id] = entry_time
        
        # Сохраняем в Redis
        person_key = f"{Config.REDIS_KEYS['person_prefix']}{track_id}"
        r.hset(person_key, mapping={
            'first_seen': entry_time,
            'last_seen': entry_time,
            'bbox': str(bbox),
            'confidence': confidence,
            'status': 'active'
        })
        r.expire(person_key, Config.PERSON_EXPIRE_TIME)
        r.sadd(Config.REDIS_KEYS['active_people'], track_id)
        
        # Обновляем время последнего обнаружения
        update_human_last_seen()
        
        # Выходим из режима запуска
        if is_system_in_startup_mode():
            clear_startup_mode()
            logger.info("👤 Режим запуска завершен досрочно")
        
        # Начинаем запись
        self.buffer.start_recording(str(track_id))
        self.recording_sessions[track_id] = {
            'start_time': entry_time,
            'active': True
        }
        
        # Публикуем событие
        self._publish_event('person_entered', track_id, {
            'bbox': bbox,
            'confidence': confidence
        })
        
        logger.info(f"👤 Person {track_id} entered")
    
    def log_person_exit(self, track_id: int):
        """Обрабатывает выход человека"""
        track_id = self._get_track_id(track_id)
        
        if track_id not in self.entry_times:
            return
        
        entry_time = self.entry_times.pop(track_id)
        exit_time = time.time()
        duration = exit_time - entry_time
        
        # Логируем в CSV
        with open("tracking_log.csv", "a") as f:
            f.write(f"{datetime.fromtimestamp(entry_time)}, {datetime.fromtimestamp(exit_time)}, {duration:.2f}s, ID:{track_id}\n")
        
        # Обновляем Redis
        person_key = f"{Config.REDIS_KEYS['person_prefix']}{track_id}"
        r.hset(person_key, mapping={
            'status': 'exited',
            'exit_time': exit_time,
            'duration': duration
        })
        r.srem(Config.REDIS_KEYS['active_people'], track_id)
        
        # Останавливаем запись с задержкой
        if track_id in self.recording_sessions:
            def delayed_stop():
                time.sleep(self.post_roll_seconds)
                if self.buffer.is_recording:
                    self.buffer.stop_recording()
                    self.recording_sessions[track_id]['active'] = False
                    logger.info(f"🛑 Recording stopped for person {track_id}")
            
            threading.Thread(target=delayed_stop, daemon=True).start()
        
        # Публикуем событие
        self._publish_event('person_exited', track_id, {'duration': duration})
        
        logger.info(f"🚶 Person {track_id} exited. Duration: {duration:.2f}s")
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """Обрабатывает один кадр"""
        self.buffer.add_frame(frame)
        
        # Детекция
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml",
            classes=[0], 
            verbose=False
        )
        
        current_ids = set()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            current_ids = set(track_ids)
            
            for i, track_id in enumerate(track_ids):
                track_id_int = self._get_track_id(track_id)
                current_time = time.time()
                
                self.entry_times[track_id_int] = current_time
                
                # Обновляем информацию
                bbox_list = [int(x) for x in boxes[i].tolist()]
                self._update_person_in_redis(track_id_int, bbox_list, confidences[i])
                
                # Рисуем bounding box
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    f"ID: {track_id_int}", 
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (0, 255, 0), 
                    2
                )
                
                # Новый человек
                if track_id_int not in self.recording_sessions:
                    self.log_person_entry(track_id_int, boxes[i], confidences[i])
        
        # Проверяем выход людей
        for track_id in list(self.recording_sessions.keys()):
            if (track_id not in current_ids and 
                self.recording_sessions[track_id].get('active', False)):
                self.log_person_exit(track_id)
        
        return True
    
    def run(self):
        """Основной цикл работы"""
        logger.info("🎯 Starting person tracking...")
        logger.info(f"📹 Buffer: {Config.BUFFER_SECONDS}s pre-roll, {self.post_roll_seconds}s post-roll")
        
        self.safety_monitor.start()
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break
                
                self.process_frame(frame)
                
                # Отображение
                cv2.imshow("YOLOv8 + ByteTrack", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.safety_monitor.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        
        if self.buffer.is_recording:
            self.buffer.stop_recording()
        
        logger.info("👋 Application stopped")

# ==================== REDIS MONITOR ====================
def monitor_detections():
    """Мониторинг событий из Redis"""
    pubsub = r.pubsub()
    pubsub.subscribe(Config.REDIS_KEYS['detection_events'])
    
    logger.info("👀 Monitoring Redis events...")
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                event = json.loads(message['data'])
                if event['type'] == 'person_entered':
                    logger.info(f"🔔 Person {event['track_id']} entered at {event['datetime']}")
                elif event['type'] == 'person_exited':
                    logger.info(f"🔔 Person {event['track_id']} exited after {event['data']['duration']:.2f}s")
            except Exception as e:
                logger.error(f"Error processing event: {e}")

# ==================== MAIN ====================
if __name__ == "__main__":
    # Запускаем монитор Redis
    monitor_thread = threading.Thread(target=monitor_detections, daemon=True)
    monitor_thread.start()
    
    # Запускаем трекер
    tracker = PersonTracker(
        source=0,
        buffer_seconds=Config.BUFFER_SECONDS,
        post_roll_seconds=Config.POST_ROLL_SECONDS
    )
    tracker.run()