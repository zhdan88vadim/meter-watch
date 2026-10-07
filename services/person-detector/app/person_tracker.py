from typing import Callable, Union

import cv2
import time
from app.database import log_person_left_to_database, log_person_detected_to_database
from meter_watch_shared.config import config
from meter_watch_shared.redis_manager import RedisManager
from app.video_buffer import VideoBuffer
from app.rate_limiter import SimpleRateLimiter
import logging

from app.protocol_models import Detector

logger = logging.getLogger(__name__)

class PersonTracker:
    def __init__(
        self,
        detector: Detector,
        buffer: VideoBuffer,
        rate_limiter: SimpleRateLimiter | None = None,
        clock: Callable[[], float] = time.time,
        source: Union[int, str] = 0,
        post_roll_seconds: int = config.POST_ROLL_SECONDS,
        frame_skip: int = config.FRAME_SKIP
    ):
        print("source: ", source)


        self._detector = detector
        self.buffer = buffer
        self.rate_limiter = rate_limiter
        self._clock = clock

        self.source = source
        self.post_roll_seconds = post_roll_seconds
        self.frame_skip = frame_skip
        
        # Состояние
        self.is_recording = False
        self.last_seen = {}           # Когда видели каждого
        self.frame_count = 0
        self.running = False
        
        # Видео
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {self.source}")
    
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
        
        detections = list(self._detector.detect(frame))

        print("detections: ", len(detections))
        
        current_time = self._clock()
        current_people = set()
        
        # Получаем ID людей в кадре
        if detections:
            current_people = [d.track_id for d in detections]
                        
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


            for det in detections:
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID: {det.track_id}",
                    (det.x1, det.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        
        # ===== ПРОСТАЯ ЛОГИКА ЗАПИСИ =====
        
        # Есть люди в кадре
        if current_people:

            if self.rate_limiter.can_save():
                log_person_detected_to_database({'ids': list(current_people)})


            if not self.is_recording:
                self.is_recording = True
                logger.info("Person detected")
                
                # self._start_recording()
        
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
                    log_person_left_to_database({'id': person_id})
                
                # Если больше нет активных людей - останавливаем запись
                if not self.last_seen:
                    self.is_recording = False
                    # self._stop_recording()
        
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

        except Exception as ex:
            import traceback
            traceback.print_exc()
            print("Error: ", ex)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка"""
        self.running = False
        
        if self.is_recording:
            self._stop_recording()
        
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("👋 Stopped")