import cv2
import time
import os
from collections import deque
from datetime import datetime
from typing import Callable, Optional
import numpy as np
from meter_watch_shared.config import config
import logging

logger = logging.getLogger(__name__)

class VideoBuffer:
    """
    Stores video frames in a circular buffer for pre-roll.
    Allows recording to start with frames saved up to the moment of detection.
    """
    
    def __init__(
            self, 
            buffer_seconds: int = config.BUFFER_SECONDS, 
            fps: int = config.DEFAULT_FPS,
            clock: Callable[[], float] = time.time,
            ):
        """
        Initialize video buffer
        
        Args:
            buffer_seconds: Number of seconds to store in buffer (pre-roll)
            fps: Video frame rate
        """
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_frames = int(buffer_seconds * fps)

        self._clock = clock
        
        # Buffers for frames and timestamps
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.current_session_id = None
        self.recording_start_time = None
        self.recording_filepath = None
                
        # Create directory for recordings
        os.makedirs("recordings", exist_ok=True)
        
        logger.info(f"📹 VideoBuffer initialized: {buffer_seconds}s buffer at {fps}fps")
    
    def add_frame(self, frame: np.ndarray, timestamp: float | None = None) -> bool:
        """
        Adds a frame to the buffer
        
        Args:
            frame: Video frame (numpy array)
            timestamp: Frame timestamp (if None, current time is used)
            
        Returns:
            bool: True if frame was successfully added
        """
        timestamp = timestamp if timestamp is not None else self._clock()
        
        try:
            # Add a copy of the frame to the buffer
            self.frames.append(frame.copy())
            self.timestamps.append(timestamp)
            
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding frame to buffer: {e}")
            return False
    
    def start_recording(self, session_id: str | None = None) -> str | None:
        """
        Starts video recording with pre-roll from buffer
        
        Args:
            session_id: Session identifier (usually person ID)
            
        Returns:
            str: Path to recording file or None on error
        """
        if self.is_recording:
            logger.warning("⚠️ Recording already in progress")
            return None
        
        try:
            self.is_recording = True
            self.current_session_id = session_id or f"session_{int(self._clock())}"
            self.recording_start_time = self._clock()
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}_ID{self.current_session_id}.mp4"
            self.recording_filepath = os.path.join("recordings", filename)
            
            # Determine frame dimensions
            if self.frames:
                h, w = self.frames[0].shape[:2]
            else:
                h, w = config.DEFAULT_FRAME_HEIGHT, config.DEFAULT_FRAME_WIDTH
                logger.warning(f"⚠️ No frames in buffer, using default size {w}x{h}")
            
            # Initialize video recording
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.recording_filepath, 
                fourcc, 
                self.fps, 
                (w, h)
            )
            
            if not self.video_writer.isOpened():
                raise Exception("Failed to open video writer")
            
            # Write pre-roll from buffer
            pre_roll_frames = 0
            for frame in self.frames:
                self.video_writer.write(frame)
                pre_roll_frames += 1
            
            logger.info(f"📹 Started recording: {filename} (pre-roll: {pre_roll_frames} frames)")
            return self.recording_filepath
            
        except Exception as e:
            logger.error(f"❌ Error starting recording: {e}")
            self.is_recording = False
            self.video_writer = None
            return None
    
    def stop_recording(self) -> None:
        """
        Stops recording and saves the video
        
        Returns:
            dict: Recording information or None on error
        """
        if not self.is_recording or not self.video_writer:
            logger.warning("⚠️ No recording in progress")
            return
        
        try:
            # Close video writer
            self.video_writer.release()
            self.video_writer = None
            
            duration = self._clock() - self.recording_start_time
            logger.info(f"📹 Stopped recording. Duration: {duration:.2f}s")

            # Reset state
            self.is_recording = False
            self.current_session_id = None
            self.recording_start_time = None
            self.recording_filepath = None
            
        except Exception as e:
            logger.error(f"❌ Error stopping recording: {e}")
            self.is_recording = False
            return
        
    
    def save_snapshot(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Saves the last frame from the buffer as an image
        
        Args:
            filename: File name (if None - auto-generated)
            
        Returns:
            str: Path to saved image or None
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

    