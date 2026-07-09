import cv2
from ultralytics import YOLO
import time
import threading
import numpy as np
import json
from typing import Union, Optional, Dict, Set
from app.config import config
from app.redis_manager import RedisManager
from app.video_buffer import VideoBuffer
from app.safety_monitor import SafetyMonitor
from app.telegram_bot import telegram_bot
import logging

logger = logging.getLogger(__name__)

class PersonTracker:
    def __init__(
        self, 
        source: Union[int, str] = 0,
        buffer_seconds: int = config.BUFFER_SECONDS,
        post_roll_seconds: int = config.POST_ROLL_SECONDS,
        frame_skip: int = config.FRAME_SKIP
    ):
        self.source = source
        self.post_roll_seconds = post_roll_seconds
        self.frame_skip = frame_skip
        
        # Простые переменные состояния
        self.is_recording = False          # Идет ли запись
        self.current_person_id = None      # ID текущего записываемого человека
        self.recording_start_time = None   # Время начала записи
        self.frame_count = 0
        self.running = False
        
        # Храним время последнего появления каждого человека
        self.last_seen: Dict[int, float] = {}
        
        # Модель
        self.model = YOLO('yolov8n.pt')
        
        # Видео
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
        
        # Буфер и монитор
        self.buffer = VideoBuffer(buffer_seconds, self.fps)
        self.safety_monitor = SafetyMonitor()
        
        # Очищаем Redis при старте
        self._cleanup_redis()
        
        # Маркируем запуск
        self._mark_startup()
        
        # Подписываемся на команды
        self._subscribe_commands()
    
    def _cleanup_redis(self):
        """Простая очистка Redis"""
        try:
            conn = RedisManager.get_connection()
            
            # Удаляем активных людей
            conn.delete(config.REDIS_KEYS['active_people'])
            
            # Удаляем флаги тревоги
            conn.delete(config.REDIS_KEYS['alert_triggered'])
            conn.delete(config.REDIS_KEYS['alert_cooldown'])
            
            # Помечаем незавершенные записи
            for key in conn.keys(f"{config.REDIS_KEYS['recording_prefix']}*"):
                conn.hset(key, 'status', 'interrupted')
                conn.hset(key, 'end_time', time.time())
            
            # Устанавливаем время старта
            conn.set(config.REDIS_KEYS['human_last_seen'], str(time.time()))
            
            logger.info("✅ Redis cleaned")
        except Exception as e:
            logger.error(f"❌ Redis cleanup error: {e}")
    
    def _mark_startup(self):
        """Отмечает запуск"""
        RedisManager.set_timestamp_key(
            config.REDIS_KEYS['startup'], 
            config.STARTUP_DURATION
        )
        logger.info(f"🔄 System started, startup mode {config.STARTUP_DURATION//60} min")
        telegram_bot.send_alert('startup')
    
    def _subscribe_commands(self):
        """Подписка на команды"""
        def listener():
            while self.running:
                try:
                    pubsub = RedisManager.get_connection().pubsub()
                    pubsub.subscribe('system:commands')
                    
                    for msg in pubsub.listen():
                        if not self.running:
                            break
                        if msg['type'] == 'message':
                            try:
                                cmd = json.loads(msg['data'])
                                if cmd['command'] == 'stop_recording':
                                    self._stop_recording()
                            except:
                                pass
                except:
                    time.sleep(5)
        
        threading.Thread(target=listener, daemon=True).start()
    
    def _start_recording(self, person_id: int):
        """Начинает запись"""
        if self.is_recording:
            return  # Уже идет запись
        
        if not self.buffer.start_recording(str(person_id)):
            return
        
        self.is_recording = True
        self.current_person_id = person_id
        self.recording_start_time = time.time()
        
        logger.info(f"📹 Recording started for person {person_id}")
    
    def _stop_recording(self):
        """Останавливает запись"""
        if not self.is_recording:
            return
        
        # Ждем post_roll
        time.sleep(self.post_roll_seconds)
        
        self.buffer.stop_recording()
        self.is_recording = False
        self.current_person_id = None
        self.recording_start_time = None
        
        logger.info("🛑 Recording stopped")
    
    def process_frame(self, frame: np.ndarray):
        """Обрабатывает кадр"""
        self.buffer.add_frame(frame)
        
        # Пропуск кадров
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return
        
        # Детекция
        try:
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml",
                classes=[0], 
                verbose=False
            )
        except:
            return
        
        current_ids = set()
        current_time = time.time()
        
        # Если есть обнаружения
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            current_ids = set(track_ids)
            
            # Обновляем время последнего появления для каждого
            for track_id in track_ids:
                track_id = int(track_id)
                self.last_seen[track_id] = current_time
                
                # Обновляем Redis
                RedisManager.set_key(
                    config.REDIS_KEYS['human_last_seen'], 
                    str(current_time)
                )
                
                # Рисуем рамку
                idx = list(track_ids).index(track_id)
                x1, y1, x2, y2 = boxes[idx]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    f"ID: {track_id}", 
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
            
            # ПРОСТАЯ ЛОГИКА ЗАПИСИ:
            # Если запись не идет - начинаем с первым обнаруженным человеком
            if not self.is_recording:
                first_person = min(current_ids)  # Берем первого
                self._start_recording(first_person)
            
            # Если запись идет, но текущего человека нет в кадре - ищем замену
            elif self.is_recording and self.current_person_id not in current_ids:
                # Проверяем, не вернулся ли он (если прошло мало времени)
                last_time = self.last_seen.get(self.current_person_id, 0)
                if current_time - last_time > 3.0:  # 3 секунды нет - меняем
                    logger.info(f"🔄 Person {self.current_person_id} lost, switching to {min(current_ids)}")
                    self._stop_recording()
                    self._start_recording(min(current_ids))
        
        # Если нет людей в кадре
        else:
            # Если запись идет и человека нет больше 3 секунд - останавливаем
            if self.is_recording:
                if self.current_person_id:
                    last_time = self.last_seen.get(self.current_person_id, 0)
                    if current_time - last_time > 3.0:
                        logger.info(f"🚶 Person {self.current_person_id} left")
                        threading.Thread(target=self._stop_recording, daemon=True).start()
        
        # Показываем статус
        status = "🔴 REC" if self.is_recording else "⏸ IDLE"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if self.is_recording else (0, 255, 255), 2)
        
        if self.is_recording and self.current_person_id:
            cv2.putText(frame, f"Person: {self.current_person_id}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Основной цикл"""
        self.running = True
        logger.info("🎯 Starting tracker...")
        
        self.safety_monitor.start()
        
        try:
            while self.running and self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    time.sleep(1)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                    continue
                
                self.process_frame(frame)
                cv2.imshow("Tracker", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка"""
        self.running = False
        self.safety_monitor.stop()
        
        if self.is_recording:
            self.buffer.stop_recording()
        
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("👋 Stopped")