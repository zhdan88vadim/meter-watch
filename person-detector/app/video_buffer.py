import cv2
import time
import os
from collections import deque
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import numpy as np
from meter_watch_shared.config import config
from meter_watch_shared.redis_manager import RedisManager
import logging

logger = logging.getLogger(__name__)

class VideoBuffer:
    """
    Хранит видеокадры в циклическом буфере для пре-ролла.
    Позволяет начать запись с сохранением кадров до момента обнаружения.
    """
    
    def __init__(self, buffer_seconds: int = config.BUFFER_SECONDS, fps: int = config.DEFAULT_FPS):
        """
        Инициализация буфера видео
        
        Args:
            buffer_seconds: Количество секунд для хранения в буфере (пре-ролл)
            fps: Частота кадров видео
        """
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_frames = int(buffer_seconds * fps)
        
        # Буферы для кадров и временных меток
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        
        # Состояние записи
        self.is_recording = False
        self.video_writer = None
        self.current_session_id = None
        self.recording_start_time = None
        self.recording_filepath = None
        
        # Статистика
        self.frames_dropped = 0
        self.total_frames_added = 0
        
        # Создаем директорию для записей
        os.makedirs("recordings", exist_ok=True)
        
        logger.info(f"📹 VideoBuffer initialized: {buffer_seconds}s buffer at {fps}fps")
    
    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Добавляет кадр в буфер
        
        Args:
            frame: Кадр видео (numpy array)
            timestamp: Временная метка кадра (если None, используется текущее время)
            
        Returns:
            bool: True если кадр успешно добавлен
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            # Добавляем копию кадра в буфер
            self.frames.append(frame.copy())
            self.timestamps.append(timestamp)
            self.total_frames_added += 1
            
            # Если идет запись, записываем кадр в файл
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding frame to buffer: {e}")
            self.frames_dropped += 1
            return False
    
    def start_recording(self, session_id: Optional[str] = None) -> Optional[str]:
        """
        Начинает запись видео с пре-роллом из буфера
        
        Args:
            session_id: Идентификатор сессии (обычно ID человека)
            
        Returns:
            str: Путь к файлу записи или None в случае ошибки
        """
        if self.is_recording:
            logger.warning("⚠️ Recording already in progress")
            return None
        
        try:
            self.is_recording = True
            self.current_session_id = session_id or f"session_{int(time.time())}"
            self.recording_start_time = time.time()
            
            # Создаем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}_ID{self.current_session_id}.mp4"
            self.recording_filepath = os.path.join("recordings", filename)
            
            # Определяем размеры кадра
            if self.frames:
                h, w = self.frames[0].shape[:2]
            else:
                h, w = config.DEFAULT_FRAME_HEIGHT, config.DEFAULT_FRAME_WIDTH
                logger.warning(f"⚠️ No frames in buffer, using default size {w}x{h}")
            
            # Инициализируем видеозапись
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.recording_filepath, 
                fourcc, 
                self.fps, 
                (w, h)
            )
            
            if not self.video_writer.isOpened():
                raise Exception("Failed to open video writer")
            
            # Записываем пре-ролл из буфера
            pre_roll_frames = 0
            for frame in self.frames:
                self.video_writer.write(frame)
                pre_roll_frames += 1
            
            # Сохраняем информацию в Redis
            redis_data = {
                'filename': filename,
                'filepath': self.recording_filepath,
                'start_time': self.recording_start_time,
                'person_id': session_id or 'unknown',
                'pre_roll_frames': pre_roll_frames,
                'fps': self.fps,
                'status': 'recording'
            }
            
            RedisManager.hset(
                f"{config.REDIS_KEYS['recording_prefix']}{self.current_session_id}", 
                redis_data
            )
            
            logger.info(f"📹 Started recording: {filename} (pre-roll: {pre_roll_frames} frames)")
            return self.recording_filepath
            
        except Exception as e:
            logger.error(f"❌ Error starting recording: {e}")
            self.is_recording = False
            self.video_writer = None
            return None
    
    def stop_recording(self) -> Optional[Dict]:
        """
        Останавливает запись и сохраняет видео
        
        Returns:
            dict: Информация о записи или None в случае ошибки
        """
        if not self.is_recording or not self.video_writer:
            logger.warning("⚠️ No recording in progress")
            return None
        
        try:
            # Закрываем видео writer
            self.video_writer.release()
            self.video_writer = None
            
            duration = time.time() - self.recording_start_time
            logger.info(f"📹 Stopped recording. Duration: {duration:.2f}s")
            
            # Обновляем информацию в Redis
            if self.current_session_id:
                key = f"{config.REDIS_KEYS['recording_prefix']}{self.current_session_id}"
                RedisManager.hset(key, {
                    'duration': duration,
                    'end_time': time.time(),
                    'status': 'completed'
                })
                RedisManager.get_connection().expire(key, config.RECORDING_EXPIRE_TIME)
            
            recording_info = {
                'session_id': self.current_session_id,
                'filepath': self.recording_filepath,
                'duration': duration,
                'start_time': self.recording_start_time,
                'end_time': time.time()
            }
            
            # Сбрасываем состояние
            self.is_recording = False
            self.current_session_id = None
            self.recording_start_time = None
            self.recording_filepath = None
            
            return recording_info
            
        except Exception as e:
            logger.error(f"❌ Error stopping recording: {e}")
            self.is_recording = False
            return None
    
    def get_buffer_frames(self, seconds: Optional[float] = None) -> List[np.ndarray]:
        """
        Получает кадры из буфера за последние N секунд
        
        Args:
            seconds: Количество секунд (если None - весь буфер)
            
        Returns:
            list: Список кадров
        """
        if seconds is None:
            return list(self.frames)
        
        # Вычисляем, сколько кадров нужно
        frames_needed = int(seconds * self.fps)
        frames_needed = min(frames_needed, len(self.frames))
        
        if frames_needed == 0:
            return []
        
        # Берем последние N кадров
        return list(self.frames)[-frames_needed:]
    
    def get_buffer_timestamps(self, seconds: Optional[float] = None) -> List[float]:
        """
        Получает временные метки из буфера за последние N секунд
        
        Args:
            seconds: Количество секунд (если None - весь буфер)
            
        Returns:
            list: Список временных меток
        """
        if seconds is None:
            return list(self.timestamps)
        
        frames_needed = int(seconds * self.fps)
        frames_needed = min(frames_needed, len(self.timestamps))
        
        if frames_needed == 0:
            return []
        
        return list(self.timestamps)[-frames_needed:]
    
    def clear_buffer(self):
        """Очищает буфер"""
        self.frames.clear()
        self.timestamps.clear()
        logger.info("🔄 Buffer cleared")
    
    def get_buffer_info(self) -> Dict:
        """
        Получает информацию о буфере
        
        Returns:
            dict: Информация о буфере
        """
        return {
            'buffer_seconds': self.buffer_seconds,
            'fps': self.fps,
            'max_frames': self.max_frames,
            'current_frames': len(self.frames),
            'buffer_utilization': len(self.frames) / self.max_frames if self.max_frames > 0 else 0,
            'is_recording': self.is_recording,
            'total_frames_added': self.total_frames_added,
            'frames_dropped': self.frames_dropped,
            'recording_session': self.current_session_id
        }
    
    def save_snapshot(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Сохраняет последний кадр из буфера как изображение
        
        Args:
            filename: Имя файла (если None - генерируется автоматически)
            
        Returns:
            str: Путь к сохраненному изображению или None
        """
        if not self.frames:
            logger.warning("⚠️ No frames in buffer for snapshot")
            return None
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
            
            filepath = os.path.join("recordings", filename)
            cv2.imwrite(filepath, self.frames[-1])
            logger.info(f"📸 Snapshot saved: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error saving snapshot: {e}")
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Получает последний кадр из буфера
        
        Returns:
            numpy.ndarray: Последний кадр или None
        """
        if self.frames:
            return self.frames[-1]
        return None
    
    def get_recording_status(self) -> Dict:
        """
        Получает статус текущей записи
        
        Returns:
            dict: Статус записи
        """
        return {
            'is_recording': self.is_recording,
            'session_id': self.current_session_id,
            'start_time': self.recording_start_time,
            'duration': time.time() - self.recording_start_time if self.recording_start_time else 0,
            'filepath': self.recording_filepath,
            'frames_in_buffer': len(self.frames)
        }