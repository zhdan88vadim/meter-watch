import cv2
from fps import FPS
from ultralytics import YOLO
import time
from datetime import datetime
import face_recognition
import numpy as np
import os
import pickle
import sys

# ==================== SUPPRESS FFMPEG WARNINGS ====================
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # Suppress FFMPEG logs

# ==================== RTSP STREAM WITH RECONNECTION ====================
class RTSPStream:
    def __init__(self, rtsp_url, target_width=640, target_height=480, max_retries=50):
        self.rtsp_url = rtsp_url
        self.target_width = target_width
        self.target_height = target_height
        self.max_retries = max_retries
        self.cap = None
        self.retry_count = 0
        
    def connect(self):
        """Connect to RTSP stream with retries"""
        print(f"📡 Connecting to RTSP: {self.rtsp_url}")
        
        # Try different backends
        backends = [
            cv2.CAP_FFMPEG,
            cv2.CAP_ANY
        ]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.rtsp_url, backend)
                
                if self.cap.isOpened():
                    # Optimize for RTSP
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    
                    # Reduce resolution if possible
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                    
                    # Lower timeout for faster reconnection
                    # self.cap.set(cv2.CAP_PROP_TIMEOUT_MS, 1000)
                    
                    # Test read
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"✅ Connected to RTSP stream (backend: {backend})")
                        return True
                    else:
                        self.cap.release()
                        self.cap = None
                        
            except Exception as e:
                print(f"⚠️ Backend {backend} failed: {e}")
                continue
        
        print("❌ Failed to connect to RTSP stream")
        return False
    
    def read(self):
        """Read frame with reconnection on error"""
        if self.cap is None:
            if not self.connect():
                return False, None
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            print("⚠️ Lost frame, attempting reconnect...")
            self.retry_count += 1
            
            if self.retry_count > self.max_retries:
                print("❌ Max retries exceeded")
                return False, None
            
            # Reconnect
            if self.cap:
                self.cap.release()
            time.sleep(1)
            
            if self.connect():
                self.retry_count = 0
                return self.read()
            return False, None
        
        # Reset retry count on successful read
        self.retry_count = 0
        
        # Resize frame if needed
        if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        return True, frame
    
    def release(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
            self.cap = None

# ==================== FACE RECOGNITION SETUP ====================
class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", encodings_file="face_encodings.pkl"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.encodings_file = encodings_file
        self.known_faces_dir = known_faces_dir
        self.recognition_cache = {}
        self.cache_timeout = 60
        
        if os.path.exists(encodings_file):
            self.load_encodings()
        else:
            self.train_known_faces()
    
    def train_known_faces(self):
        """Train face encodings from images in known_faces directory"""
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"📁 Created {self.known_faces_dir} directory")
            print("📸 Add face images named as: person_name.jpg")
            return
        
        print("🔄 Training face encodings...")
        
        for filename in os.listdir(self.known_faces_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(self.known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
                if face_encodings:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(name)
                    print(f"   ✅ Trained: {name}")
                else:
                    print(f"   ⚠️ Could not encode face in: {filename}")
            else:
                print(f"   ⚠️ No face found in: {filename}")
        
        if self.known_face_encodings:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            print(f"💾 Saved {len(self.known_face_encodings)} face encodings")
    
    def load_encodings(self):
        """Load pre-trained face encodings"""
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"✅ Loaded {len(self.known_face_encodings)} face encodings")
        except Exception as e:
            print(f"⚠️ Could not load encodings: {e}")
            self.train_known_faces()
    
    def recognize_face(self, face_image, track_id=None):
        """Recognize a face with caching"""
        try:
            if not self.known_face_encodings:
                return "Unknown", 0.0
            
            if face_image is None or face_image.size == 0:
                return "Unknown", 0.0
            
            if len(face_image.shape) != 3:
                return "Unknown", 0.0
            
            h, w = face_image.shape[:2]
            if h < 20 or w < 20:
                return "Unknown", 0.0
            
            try:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            except:
                face_rgb = face_image
            
            if face_rgb.shape[0] > 100 or face_rgb.shape[1] > 100:
                face_rgb = cv2.resize(face_rgb, (100, 100))
            
            face_locations = face_recognition.face_locations(face_rgb, model="hog")
            
            if not face_locations:
                return "Unknown", 0.0
            
            face_encodings = face_recognition.face_encodings(
                face_rgb, 
                known_face_locations=face_locations,
                num_jitters=0
            )
            
            if not face_encodings:
                return "Unknown", 0.0
            
            face_encoding = face_encodings[0]
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                confidence = 1 - distances[best_match_index]
                
                if confidence > 0.3:
                    name = self.known_face_names[best_match_index]
                    return name, confidence
            
            return "Unknown", 0.0
            
        except Exception as e:
            print(f"   ⚠️ Face recognition error: {e}")
            return "Unknown", 0.0

# ==================== MAIN TRACKING CLASS ====================
class PersonTrackerWithFaces:
    def __init__(self, source=0):
        self.source = source
        self.model = YOLO('yolov8n.pt')
        
        # Use RTSPStream wrapper
        if isinstance(source, str) and source.startswith('rtsp://'):
            self.stream = RTSPStream(source, target_width=640, target_height=480)
            if not self.stream.connect():
                print("❌ Failed to connect to RTSP stream")
                sys.exit(1)
            self.cap = None
            self.use_stream = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.use_stream = False
        
        # Frame skipping settings
        self.skip_frames = 1
        self.recognition_interval = 10
        self.target_width = 640
        self.target_height = 480
        
        self.entry_times = {}
        self.person_names = {}
        self.face_recognizer = FaceRecognizer()
        self.fps_calculator = FPS(avg_window=30)
        
        # Settings
        self.confidence_threshold = 0.3
        self.face_padding = 20
        self.face_upper_ratio = 0.4
        self.recognition_attempts = {}
        
        # Create log file
        with open("tracking_log.csv", "w") as f:
            f.write("Entry Time, Exit Time, Duration (s), Track ID, Person Name\n")
    
    def log_person_exit(self, track_id, entry_time):
        exit_time = time.time()
        duration = exit_time - entry_time
        name = self.person_names.get(track_id, "Unknown")
        
        with open("tracking_log.csv", "a") as f:
            f.write(f"{datetime.fromtimestamp(entry_time)}, ")
            f.write(f"{datetime.fromtimestamp(exit_time)}, ")
            f.write(f"{duration:.2f}s, ID:{track_id}, {name}\n")
        
        print(f"🚶 {name} (ID:{track_id}) left. Duration: {duration:.2f}s")
        
        # Cleanup
        if track_id in self.person_names:
            del self.person_names[track_id]
        if track_id in self.recognition_attempts:
            del self.recognition_attempts[track_id]
    
    def extract_face_region(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        
        face_y1 = max(0, y1 - self.face_padding)
        face_y2 = min(frame.shape[0], y1 + int((y2 - y1) * self.face_upper_ratio))
        face_x1 = max(0, x1 - self.face_padding)
        face_x2 = min(frame.shape[1], x2 + self.face_padding)
        
        return frame[face_y1:face_y2, face_x1:face_x2]
    
    def identify_person(self, frame, bbox, track_id):
        face_region = self.extract_face_region(frame, bbox)
        
        if face_region.size == 0:
            return "Unknown", 0.0
        
        name, confidence = self.face_recognizer.recognize_face(face_region, track_id)
        return name, confidence
    
    def draw_person_info(self, frame, bbox, track_id, confidence, name):
        x1, y1, x2, y2 = bbox
        is_known = name != "Unknown"
        color = (0, 255, 0) if is_known else (0, 0, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{name} (ID:{track_id})"
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, f"det: {confidence:.2f}", (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show attempts for unknown
        if name == "Unknown" and track_id in self.recognition_attempts:
            attempts = self.recognition_attempts[track_id]
            cv2.putText(frame, f"attempts: {attempts}", (x1, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def process_tracked_people(self, frame, boxes, track_ids, confidences, processed_count):
        current_ids = set(track_ids)
        
        # Check exits
        for track_id in list(self.entry_times.keys()):
            if track_id not in current_ids:
                self.log_person_exit(track_id, self.entry_times.pop(track_id))
        
        # Process each person
        for i, track_id in enumerate(track_ids):
            bbox = boxes[i]
            
            # New person
            if track_id not in self.entry_times:
                self.entry_times[track_id] = time.time()
                self.person_names[track_id] = "Unknown"
                self.recognition_attempts[track_id] = 0
                print(f"🆕 New person. ID: {track_id}")
                
                # Try to recognize immediately
                name, face_conf = self.identify_person(frame, bbox, track_id)
                self.recognition_attempts[track_id] += 1
                
                if name != "Unknown":
                    self.person_names[track_id] = name
                    print(f"   ✅ Recognized as: {name} (confidence: {face_conf:.2f})")
                else:
                    print(f"   ❌ Face not recognized (will retry)")
            
            # Periodic recognition for Unknown
            elif self.person_names.get(track_id) == "Unknown":
                if processed_count % self.recognition_interval == 0:
                    name, face_conf = self.identify_person(frame, bbox, track_id)
                    self.recognition_attempts[track_id] += 1
                    
                    if name != "Unknown":
                        self.person_names[track_id] = name
                        print(f"   ✅ RECOGNIZED! ID:{track_id} as: {name} (confidence: {face_conf:.2f}) after {self.recognition_attempts[track_id]} attempts")
                    elif self.recognition_attempts[track_id] % 10 == 0:
                        print(f"   ⏳ ID:{track_id} still unknown after {self.recognition_attempts[track_id]} attempts")
            
            # Draw
            name = self.person_names.get(track_id, "Unknown")
            self.draw_person_info(frame, bbox, track_id, confidences[i], name)
    
    def draw_info_panel(self, frame, current_fps, real_time_fps, frame_count, processed_count, skipped):
        y_pos = 30
        line_height = 30
        
        unknown_count = sum(1 for name in self.person_names.values() if name == "Unknown")
        
        info = [
            (f"FPS (process): {current_fps:.1f}", (0, 255, 255)),
            (f"FPS (display): {real_time_fps:.1f}", (100, 255, 100)),
            (f"People: {len(self.entry_times)}", (255, 255, 255)),
            (f"Unknown: {unknown_count}", (0, 0, 255)),
            (f"Known faces: {len(self.face_recognizer.known_face_names)}", (255, 255, 255)),
            (f"Frame: {frame_count} (proc: {processed_count})", (200, 200, 200)),
        ]
        
        for text, color in info:
            cv2.putText(frame, text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += line_height
    
    def read_frame(self):
        """Read frame from either RTSP stream or webcam"""
        if self.use_stream:
            return self.stream.read()
        else:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Resize if needed
                if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
            return ret, frame
    
    def run(self):
        print("🎯 Starting tracking with face recognition...")
        print(f"📊 Known faces: {len(self.face_recognizer.known_face_names)}")
        print(f"⚡ Frame skip: every {self.skip_frames + 1}-th frame")
        print(f"⚡ Recognition: every {self.recognition_interval} frames")
        print("📸 Press 'q' to exit")
        
        frame_count = 0
        processed_count = 0
        skipped_count = 0
        
        real_time_start = time.time()
        real_time_frames = 0
        
        while True:
            # Read frame
            success, frame = self.read_frame()
            if not success:
                print("⚠️ No frame received")
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # Frame skipping
            if frame_count % (self.skip_frames + 1) != 0:
                skipped_count += 1
                cv2.putText(frame, f"SKIP {frame_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("YOLOv8 + ByteTrack + Face Recognition", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            
            processed_count += 1
            
            # Real-time FPS
            real_time_frames += 1
            if time.time() - real_time_start >= 1.0:
                real_time_fps = real_time_frames / (time.time() - real_time_start)
                real_time_frames = 0
                real_time_start = time.time()
            else:
                real_time_fps = 0
            
            current_fps = self.fps_calculator.update()
            
            # YOLO tracking
            try:
                results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                          classes=[0], verbose=False)
            except Exception as e:
                print(f"⚠️ YOLO error: {e}")
                continue
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                self.process_tracked_people(frame, boxes, track_ids, confidences, processed_count)
            
            self.draw_info_panel(frame, current_fps, real_time_fps, 
                               frame_count, processed_count, skipped_count)
            
            cv2.imshow("YOLOv8 + ByteTrack + Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Cleanup
        if self.use_stream:
            self.stream.release()
        else:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Log remaining
        for track_id, entry_time in list(self.entry_times.items()):
            self.log_person_exit(track_id, entry_time)
        
        print("\n👋 Program finished")
        print(f"📊 Total frames: {frame_count}")
        print(f"📊 Processed: {processed_count}")
        print(f"📊 Skipped: {skipped_count}")
        print("📊 Results saved to tracking_log.csv")

# ==================== MAIN ====================
if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
        print("📁 Created 'known_faces' folder")
        print("📸 Add photos of known people")
        print("   Format: person_name.jpg")
    
    # Use RTSP stream
    rtsp_url = "rtsp://192.168.0.102:8554/hikvision_room?mp4"
    # Or webcam: rtsp_url = 0
    
    tracker = PersonTrackerWithFaces(source=rtsp_url)
    tracker.run()