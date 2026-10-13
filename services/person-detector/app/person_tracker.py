from typing import Callable

import cv2
import time
import numpy as np

from app.video_buffer import VideoBuffer
from app.rate_limiter import SimpleRateLimiter
from app.protocol_models import KeyValueStore
from app.protocol_models import ActivityRepository, Detector

from meter_watch_shared.config import RedisKeys

import logging

logger = logging.getLogger(__name__)

class PersonTracker:
    def __init__(
        self,
        detector: Detector,
        buffer: VideoBuffer,
        db_repository: ActivityRepository,
        keys: RedisKeys,
        store: KeyValueStore,
        post_roll_seconds: int,
        frame_skip: int,
        rate_limiter: SimpleRateLimiter | None = None,
        clock: Callable[[], float] = time.time,
        source: int | str = 0,
    ):
        print("source: ", source)

        self._detector = detector
        self._keys = keys
        self.buffer = buffer
        self.rate_limiter = rate_limiter
        self._clock = clock
        self._db_repository = db_repository
        self._store = store

        self.source = source
        self.post_roll_seconds = post_roll_seconds
        self.frame_skip = frame_skip
        
        # State
        self.is_recording = False
        self.last_seen = {}           # When each person was last seen
        self.frame_count = 0
        self.running = False
        
        self.cap = None
    
    def _start_recording(self):
        """Start recording"""
        if self.is_recording:
            return
        
        self.buffer.start_recording("recording")
        self.is_recording = True
        logger.info("📹 Recording STARTED")
    
    def _stop_recording(self):
        """Stop recording"""
        if not self.is_recording:
            return
        
        # Wait for post_roll
        time.sleep(self.post_roll_seconds)
        
        self.buffer.stop_recording()
        self.is_recording = False
        logger.info("🛑 Recording STOPPED")
    
    def process_frame(self, frame: np.ndarray):
        """Process frame"""
        self.buffer.add_frame(frame)
        
        # skip frame
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return
        
        detections = list(self._detector.detect(frame))

        print("detections: ", len(detections))
        
        current_time = self._clock()
        current_people = set()
        
        # get person ID
        if detections:
            current_people = [d.track_id for d in detections]
                        
            # Update appearance time for each person
            for person_id in current_people:
                self.last_seen[person_id] = current_time
                time_str = time.strftime("%H:%M %d:%m:%Y", time.localtime(self._clock()))

                self._store.set(self._keys.human_last_seen_str, time_str)
                self._store.set(self._keys.human_last_seen, str(current_time))

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

            if self.rate_limiter.can_save():
                self._db_repository.log_person_detected({'ids': list(current_people)})                

            if not self.is_recording:
                # COMMENT THIS WHEN _start_recording is uncommneted
                # self.is_recording = True
                logger.info("Person detected")
                
                self._start_recording()
        
        # No people in frame
        else:
            # If recording is in progress - check if someone left
            if self.is_recording:
                # Check everyone we've seen
                people_to_remove = []
                for person_id, last_time in self.last_seen.items():
                    # If person hasn't been seen for more than 3 seconds - remove them
                    if current_time - last_time > 3.0:
                        people_to_remove.append(person_id)
                
                # Remove those who have been gone for a while
                for person_id in people_to_remove:
                    del self.last_seen[person_id]
                    logger.info(f"🚶 Person {person_id} left, now last_seen: {self.last_seen}" )
                    self._db_repository.log_person_left({'id': person_id})                    
                
                # If there are no more active people - stop recording
                if not self.last_seen:
                    # COMMENT THIS WHEN _stop_recording is uncommneted
                    # self.is_recording = False

                    self._stop_recording()
        
        # Display status
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
        
        # Number of people
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
        """Start"""
        self.running = True
        logger.info("Starting tracker...")
        
        try:           
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                print(f"Error: Could not open video source: {self.source}")
                            
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
        """Cleanup"""
        self.running = False
        
        if self.is_recording:
            self._stop_recording()
        
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("👋 Stopped")