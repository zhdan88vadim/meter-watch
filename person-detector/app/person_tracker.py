from typing import Union

import cv2
from ultralytics import YOLO
import time
from meter_watch_shared.config import config
from meter_watch_shared.redis_manager import RedisManager
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
        
        # Состояние
        self.is_recording = False
        self.last_seen = {}           # Когда видели каждого
        self.frame_count = 0
        self.running = False
        
        # Модель
        self.model = YOLO('yolov8n.pt')
        
        # Видео
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
        
        # Буфер
        self.buffer = VideoBuffer(buffer_seconds, self.fps)
        self.safety_monitor = SafetyMonitor()
        
        # Очистка при старте
        self._cleanup_redis()
        self._mark_startup()
    
    def _cleanup_redis(self):
        """Очистка Redis"""
        try:
            conn = RedisManager.get_connection()
            # conn.delete(config.REDIS_KEYS['active_people'])
            conn.delete(config.REDIS_KEYS['alert_triggered'])
            conn.delete(config.REDIS_KEYS['alert_cooldown'])
            logger.info("✅ Redis cleaned")
        except:
            pass
    
    def _mark_startup(self):
        """Отметка запуска"""
        RedisManager.set_timestamp_key(
            config.REDIS_KEYS['startup'], 
            config.STARTUP_DURATION
        )
        telegram_bot.send_alert('startup')
    
    def _start_recording(self):
        """Начать запись"""
        if self.is_recording:
            return
        
        self.buffer.start_recording("recording")
        self.is_recording = True
        logger.info("📹 Recording STARTED")
    
    def _stop_recording(self):
        """Остановить запись"""
        if not self.is_recording:
            return
        
        # Ждем post_roll
        time.sleep(self.post_roll_seconds)
        
        self.buffer.stop_recording()
        self.is_recording = False
        logger.info("🛑 Recording STOPPED")
    
    def process_frame(self, frame):
        """Обработка кадра - простая логика"""
        # Добавляем в буфер
        self.buffer.add_frame(frame)
        
        # Пропускаем кадры
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return
        
        # Детекция людей
        try:
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml",
                classes=[0],  # Только люди
                verbose=False
            )
        except:
            return
        
        current_time = time.time()
        current_people = set()
        
        # Получаем ID людей в кадре
        if results and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            current_people = set(int(x) for x in track_ids)
            
            # Обновляем время появления каждого
            for person_id in current_people:
                self.last_seen[person_id] = current_time
                time_str = time.strftime("%H:%M %d:%m:%Y", time.localtime(time.time()))

                RedisManager.set_key(
                    config.REDIS_KEYS['human_last_seen_str'], 
                    time_str
                )
                RedisManager.set_key(
                    config.REDIS_KEYS['human_last_seen'], 
                    str(current_time)
                )
            
            # Рисуем рамки
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            for idx, person_id in enumerate(track_ids):
                x1, y1, x2, y2 = boxes[idx]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    f"ID: {int(person_id)}", 
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
        
        # ===== ПРОСТАЯ ЛОГИКА ЗАПИСИ =====
        
        # Есть люди в кадре
        if current_people:
            pass
            # Если запись не идет - начинаем
            # if not self.is_recording:
                # self._start_recording()
            # Если запись идет - продолжаем
            # (ничего не делаем)
        
        # Нет людей в кадре
        else:
            # Если запись идет - проверяем, может кто-то вышел
            if self.is_recording:
                # Проверяем всех, кого видели
                people_to_remove = []
                for person_id, last_time in self.last_seen.items():
                    # Если человека нет больше 3 секунд - удаляем
                    if current_time - last_time > 3.0:
                        people_to_remove.append(person_id)
                
                # Удаляем тех, кого давно нет
                for person_id in people_to_remove:
                    del self.last_seen[person_id]
                    logger.info(f"🚶 Person {person_id} left")
                
                # Если больше нет активных людей - останавливаем запись
                if not self.last_seen:
                    self._stop_recording()
        
        # Отображаем статус
        status = "🔴 REC" if self.is_recording else "⏸ IDLE"
        cv2.putText(
            frame, 
            status, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,
            (0, 0, 255) if self.is_recording else (0, 255, 255), 
            2
        )
        
        # Количество людей
        cv2.putText(
            frame,
            f"People: {len(current_people)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    def run(self):
        """Запуск"""
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
                # cv2.imshow("Tracker", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка"""
        self.running = False
        self.safety_monitor.stop()
        
        if self.is_recording:
            self._stop_recording()
        
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("👋 Stopped")