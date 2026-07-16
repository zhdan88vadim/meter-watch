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

# ==================== REDIS SETUP ====================

# FOR DOCKER
# REDIS_HOST = os.getenv('REDIS_HOST', 'redis')

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost') 
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

r = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    db=0, 
    password=REDIS_PASSWORD if REDIS_PASSWORD else None,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5
)

# r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# ==================== HELPER FUNCTIONS ====================
def convert_to_serializable(obj):
    """Рекурсивно преобразует numpy типы в стандартные Python типы для сохранения в Redis"""
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
    else:
        return obj

# ==================== VIDEO BUFFER CLASS ====================
class VideoBuffer:
    """Stores video frames in a circular buffer for pre-roll recording"""
    
    def __init__(self, buffer_seconds=4, fps=30):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_frames = int(buffer_seconds * fps)
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        self.is_recording = False
        self.video_writer = None
        self.current_session_id = None
        self.recording_start_time = None
        self.last_detection_time = {}
        self.person_in_frame = False
        
    def add_frame(self, frame, timestamp=None):
        """Add frame to buffer with timestamp"""
        if timestamp is None:
            timestamp = time.time()
        self.frames.append(frame.copy())
        self.timestamps.append(timestamp)
        
        # If recording, also write to video file
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
    
    def start_recording(self, session_id=None):
        """Start recording video (will include pre-roll frames)"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.current_session_id = session_id or f"session_{int(time.time())}"
        self.recording_start_time = time.time()
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}_ID{session_id}.mp4"
        
        # Ensure recordings directory exists
        os.makedirs("recordings", exist_ok=True)
        filepath = os.path.join("recordings", filename)
        
        # Get frame dimensions
        if self.frames:
            h, w = self.frames[0].shape[:2]
        else:
            h, w = 480, 640  # Default fallback
            
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))
        
        # Write pre-roll frames from buffer
        for frame in self.frames:
            self.video_writer.write(frame)
            
        print(f"📹 Started recording: {filename}")
        
        # Store recording info in Redis
        r.hset(f"recording:{self.current_session_id}", mapping={
            'filename': filename,
            'start_time': self.recording_start_time,
            'person_id': session_id or 'unknown',
            'pre_roll_frames': len(self.frames)
        })
        
        return filepath
    
    def stop_recording(self):
        """Stop recording and save video"""
        if not self.is_recording or not self.video_writer:
            return
            
        self.is_recording = False
        self.video_writer.release()
        self.video_writer = None
        
        duration = time.time() - self.recording_start_time
        print(f"📹 Stopped recording. Duration: {duration:.2f}s")
        
        # Update Redis with duration
        if self.current_session_id:
            r.hset(f"recording:{self.current_session_id}", 'duration', duration)
            r.hset(f"recording:{self.current_session_id}", 'end_time', time.time())
            r.expire(f"recording:{self.current_session_id}", 86400)  # Expire in 24h
        
        self.current_session_id = None
        self.recording_start_time = None
    
    def get_frames_after_trigger(self, after_seconds=4):
        """Get frames from after trigger point (for post-roll)"""
        if not self.frames:
            return []
        
        # We'll stop recording after this many seconds
        # The actual post-roll frames will be written in real-time
        pass

# ==================== MAIN DETECTION CLASS ====================
class PersonTracker:
    def __init__(self, source=0, buffer_seconds=4, post_roll_seconds=4):
        self.source = source
        self.buffer_seconds = buffer_seconds
        self.post_roll_seconds = post_roll_seconds
        self.entry_times = {}
        self.exit_times = {}
        self.recording_sessions = {}  # track_id -> session info
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default fallback
            
        # Initialize video buffer
        self.buffer = VideoBuffer(buffer_seconds=self.buffer_seconds, fps=self.fps)
        
        # Redis pub/sub for real-time events
        self.pubsub = r.pubsub()
        
    def publish_event(self, event_type, track_id, data=None):
        """Publish event to Redis for other services"""
        # Преобразуем все numpy типы в стандартные Python типы
        serializable_data = convert_to_serializable(data or {})
        track_id_int = int(track_id) if isinstance(track_id, (np.integer, np.floating)) else track_id
        
        event = {
            'type': event_type,
            'track_id': track_id_int,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'data': serializable_data
        }
        r.publish('detection:events', json.dumps(event))
        
        # Also store in Redis for history
        r.lpush('detection:history', json.dumps(event))
        r.ltrim('detection:history', 0, 999)
    
    def log_person_entry(self, track_id, bbox, confidence):
        """Handle person entering the frame"""
        entry_time = time.time()
        
        # Преобразуем numpy типы в стандартные Python типы
        track_id_int = int(track_id) if isinstance(track_id, (np.integer, np.floating)) else track_id
        bbox_list = [int(x) for x in bbox]  # Преобразуем все значения в int
        confidence_val = float(confidence) if isinstance(confidence, (np.floating, np.integer)) else float(confidence)
        
        self.entry_times[track_id_int] = entry_time
        
        # Store in Redis
        person_key = f"person:{track_id_int}"
        r.hset(person_key, mapping={
            'first_seen': entry_time,
            'last_seen': entry_time,
            'bbox': str(bbox_list),  # Сохраняем как строку
            'confidence': confidence_val,
            'status': 'active'
        })
        r.expire(person_key, 3600)  # Expire after 1 hour
        
        # Add to active set
        r.sadd('active:people', track_id_int)
        
        # Start recording
        self.buffer.start_recording(session_id=str(track_id_int))
        self.recording_sessions[track_id_int] = {
            'start_time': entry_time,
            'active': True
        }
        
        # Publish event
        self.publish_event('person_entered', track_id_int, {
            'bbox': bbox_list,
            'confidence': confidence_val
        })
        
        print(f"👤 Person {track_id_int} entered at {datetime.fromtimestamp(entry_time)}")
    
    def log_person_exit(self, track_id):
        """Handle person leaving the frame"""
        # Преобразуем track_id в int если нужно
        track_id_int = int(track_id) if isinstance(track_id, (np.integer, np.floating)) else track_id
        
        if track_id_int not in self.entry_times:
            return
            
        entry_time = self.entry_times.pop(track_id_int)
        exit_time = time.time()
        duration = exit_time - entry_time
        
        # Log to CSV
        with open("tracking_log.csv", "a") as f:
            f.write(f"{datetime.fromtimestamp(entry_time)}, {datetime.fromtimestamp(exit_time)}, {duration:.2f}s, ID:{track_id_int}\n")
        
        # Update Redis
        person_key = f"person:{track_id_int}"
        r.hset(person_key, 'status', 'exited')
        r.hset(person_key, 'exit_time', exit_time)
        r.hset(person_key, 'duration', duration)
        r.srem('active:people', track_id_int)
        
        # Stop recording after post-roll period
        if track_id_int in self.recording_sessions:
            # Schedule stop after post_roll_seconds
            def delayed_stop():
                time.sleep(self.post_roll_seconds)
                if self.buffer.is_recording:
                    self.buffer.stop_recording()
                    self.recording_sessions[track_id_int]['active'] = False
                    print(f"🛑 Recording stopped for person {track_id_int}")
            
            threading.Thread(target=delayed_stop, daemon=True).start()
        
        # Publish event
        self.publish_event('person_exited', track_id_int, {
            'duration': duration
        })
        
        print(f"🚶 Person {track_id_int} exited. Duration: {duration:.2f}s")
    
    def run(self):
        """Main detection loop"""
        print("🎯 Starting person tracking with video recording...")
        print(f"📹 Buffer: {self.buffer_seconds}s pre-roll, {self.post_roll_seconds}s post-roll")
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            # Add frame to buffer (always, even if no detection)
            self.buffer.add_frame(frame)
            
            # Run detection and tracking
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                      classes=[0], verbose=False)
            
            current_ids = set()
            
            if results[0].boxes.id is not None:
                # Get detection data
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                current_ids = set(track_ids)
                
                # Update last seen time for each detected person
                for i, track_id in enumerate(track_ids):
                    # Преобразуем в int для использования как ключ
                    track_id_int = int(track_id)
                    self.entry_times[track_id_int] = time.time()  # Update last seen
                    
                    # Update Redis with current position
                    bbox_list = [int(x) for x in boxes[i].tolist()]
                    r.hset(f"person:{track_id_int}", 'last_seen', time.time())
                    r.hset(f"person:{track_id_int}", 'bbox', str(bbox_list))
                    
                    # Draw bounding boxes
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id_int}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # New person detected
                    if track_id_int not in self.recording_sessions:
                        self.log_person_entry(track_id_int, boxes[i], confidences[i])
            
            # Check for people who have exited
            for track_id in list(self.recording_sessions.keys()):
                if track_id not in current_ids and self.recording_sessions[track_id].get('active', False):
                    # Person is no longer in frame
                    self.log_person_exit(track_id)
            
            # Update display
            cv2.imshow("YOLOv8 + ByteTrack", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Stop any remaining recordings
        if self.buffer.is_recording:
            self.buffer.stop_recording()
        
        print("👋 Application stopped")

# ==================== REDIS MONITOR (Optional) ====================
def monitor_detections():
    """Separate thread to monitor Redis events"""
    pubsub = r.pubsub()
    pubsub.subscribe('detection:events')
    
    print("👀 Monitoring Redis events...")
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                event = json.loads(message['data'])
                if event['type'] == 'person_entered':
                    print(f"🔔 ALERT: Person {event['track_id']} entered at {event['datetime']}")
                elif event['type'] == 'person_exited':
                    print(f"🔔 ALERT: Person {event['track_id']} exited after {event['data']['duration']:.2f}s")
            except:
                pass

# ==================== MAIN ====================
if __name__ == "__main__":
    # Start Redis monitor in background
    monitor_thread = threading.Thread(target=monitor_detections, daemon=True)
    monitor_thread.start()
    
    # Run tracker
    tracker = PersonTracker(
        source=0,  # Use 0 for webcam, or 'video.mp4' for file
        buffer_seconds=4,  # 4 seconds before detection
        post_roll_seconds=4  # 4 seconds after person disappears
    )
    tracker.run()