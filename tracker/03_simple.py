import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import os
import threading
from collections import deque
import numpy as np

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
        self.recording_sessions = {}  # track_id -> session info
        
    def add_frame(self, frame, timestamp=None):
        """Add frame to buffer with timestamp"""
        if timestamp is None:
            timestamp = time.time()
        self.frames.append(frame.copy())
        self.timestamps.append(timestamp)
        
        # If recording, also write to video file
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
    
    def start_recording(self, track_id=None):
        """Start recording video (will include pre-roll frames)"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.current_session_id = track_id or f"unknown_{int(time.time())}"
        self.recording_start_time = time.time()
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}_ID{self.current_session_id}.mp4"
        
        # Ensure recordings directory exists
        os.makedirs("recordings", exist_ok=True)
        filepath = os.path.join("recordings", filename)
        
        # Get frame dimensions from buffer or use default
        if self.frames:
            h, w = self.frames[0].shape[:2]
        else:
            h, w = 480, 640
            
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))
        
        # Write pre-roll frames from buffer
        for frame in self.frames:
            self.video_writer.write(frame)
            
        print(f"📹 Started recording: {filename}")
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
        
        # Log to file
        with open("recording_log.csv", "a") as f:
            f.write(f"{datetime.fromtimestamp(self.recording_start_time)}, ")
            f.write(f"{datetime.now()}, {duration:.2f}s, ")
            f.write(f"ID:{self.current_session_id}\n")
        
        self.current_session_id = None
        self.recording_start_time = None

# ==================== MAIN DETECTION CLASS ====================
class PersonTracker:
    def __init__(self, source=0, buffer_seconds=4, post_roll_seconds=4):
        self.source = source
        self.buffer_seconds = buffer_seconds
        self.post_roll_seconds = post_roll_seconds
        self.entry_times = {}
        self.active_recordings = {}  # track_id -> True/False
        self.post_roll_timers = {}  # track_id -> timer thread
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
            
        # Initialize video buffer
        self.buffer = VideoBuffer(buffer_seconds=self.buffer_seconds, fps=self.fps)
        
        # Create recordings directory
        os.makedirs("recordings", exist_ok=True)
        
        # Initialize log file
        with open("tracking_log.csv", "w") as f:
            f.write("Entry Time, Exit Time, Duration (s), Track ID\n")
    
    def log_person_entry(self, track_id, bbox, confidence):
        """Handle person entering the frame"""
        entry_time = time.time()
        self.entry_times[track_id] = entry_time
        
        # Start recording with pre-roll
        self.buffer.start_recording(track_id=track_id)
        self.active_recordings[track_id] = True
        
        # Cancel any pending stop timer
        if track_id in self.post_roll_timers:
            self.post_roll_timers[track_id].cancel()
            del self.post_roll_timers[track_id]
        
        print(f"👤 Person {track_id} entered at {datetime.fromtimestamp(entry_time)}")
        print(f"   📹 Recording started (with {self.buffer_seconds}s pre-roll)")
    
    def log_person_exit(self, track_id):
        """Handle person leaving the frame"""
        if track_id not in self.entry_times:
            return
            
        entry_time = self.entry_times.pop(track_id)
        exit_time = time.time()
        duration = exit_time - entry_time
        
        # Log to CSV
        with open("tracking_log.csv", "a") as f:
            f.write(f"{datetime.fromtimestamp(entry_time)}, ")
            f.write(f"{datetime.fromtimestamp(exit_time)}, ")
            f.write(f"{duration:.2f}s, ID:{track_id}\n")
        
        # Mark as inactive but don't stop recording yet (post-roll)
        self.active_recordings[track_id] = False
        
        # Schedule recording stop after post_roll_seconds
        def stop_after_delay():
            time.sleep(self.post_roll_seconds)
            if self.buffer.is_recording and not any(self.active_recordings.values()):
                # Only stop if no other people are being recorded
                self.buffer.stop_recording()
            elif self.buffer.is_recording:
                # Check if this was the last person
                active_count = sum(1 for v in self.active_recordings.values() if v)
                if active_count == 0:
                    self.buffer.stop_recording()
            
            # Clean up
            if track_id in self.post_roll_timers:
                del self.post_roll_timers[track_id]
        
        timer = threading.Timer(self.post_roll_seconds, stop_after_delay)
        timer.daemon = True
        timer.start()
        self.post_roll_timers[track_id] = timer
        
        print(f"🚶 Person {track_id} exited. Duration: {duration:.2f}s")
        print(f"   📹 Continuing recording for {self.post_roll_seconds}s (post-roll)")
    
    def run(self):
        """Main detection loop"""
        print("🎯 Starting person tracking with video recording...")
        print(f"📹 Buffer: {self.buffer_seconds}s pre-roll, {self.post_roll_seconds}s post-roll")
        print("Press 'q' to quit\n")
        
        frame_count = 0
        start_time = time.time()
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            # Add frame to buffer (always, even if no detection)
            self.buffer.add_frame(frame)
            frame_count += 1
            
            # Run detection every other frame for speed
            if frame_count % 2 == 0:
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
                    
                    # Update tracking for detected people
                    for i, track_id in enumerate(track_ids):
                        # Update last seen time
                        self.entry_times[track_id] = time.time()
                        
                        # Draw bounding boxes
                        x1, y1, x2, y2 = boxes[i]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"{confidences[i]:.2f}", (x1, y1 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # New person detected
                        if track_id not in self.active_recordings or not self.active_recordings[track_id]:
                            self.log_person_entry(track_id, boxes[i], confidences[i])
                
                # Check for people who have exited
                for track_id in list(self.active_recordings.keys()):
                    if track_id not in current_ids and self.active_recordings[track_id]:
                        self.log_person_exit(track_id)
            
            # Add recording status to display
            status_text = f"Recording: {'🟢 ACTIVE' if self.buffer.is_recording else '⏸️ IDLE'}"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.buffer.is_recording else (255, 255, 0), 2)
            
            # Show active count
            active_count = sum(1 for v in self.active_recordings.values() if v)
            cv2.putText(frame, f"People: {active_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("YOLOv8 + ByteTrack", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Stop any remaining recordings
        if self.buffer.is_recording:
            print("📹 Stopping final recording...")
            self.buffer.stop_recording()
        
        print(f"\n👋 Application stopped")
        print(f"📊 Check 'tracking_log.csv' for person tracking data")
        print(f"🎬 Recordings saved in 'recordings/' folder")

# ==================== MAIN ====================
if __name__ == "__main__":
    # Choose source: 0 for webcam, or 'video.mp4' for file
    SOURCE = 0  # Change to 'video.mp4' for file input
    
    # Run tracker with 4-second pre and post roll
    tracker = PersonTracker(
        source=SOURCE,
        buffer_seconds=4,   # 4 seconds before detection
        post_roll_seconds=4 # 4 seconds after person disappears
    )
    tracker.run()