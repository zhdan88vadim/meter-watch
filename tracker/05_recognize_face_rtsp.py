import cv2
from fps import FPS
from ultralytics import YOLO
import time
from datetime import datetime
import face_recognition
import numpy as np
import os
import pickle
import warnings

# ==================== SUPPRESS WARNINGS ====================
warnings.filterwarnings("ignore")
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

# ==================== VIDEO RECORDER ====================
class VideoRecorder:
    def __init__(self, output_dir="recordings", record_on_any_person=True):
        self.output_dir = output_dir
        self.recording = False
        self.raw_writer = None
        self.annotated_writer = None
        self.record_start_time = None
        self.last_person_time = time.time()
        self.no_person_timeout = 4.0
        self.session_id = None
        self.current_track_ids = set()
        self.record_on_any_person = record_on_any_person  # Режим записи
        self.frame_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Recordings will be saved to: {output_dir}")
    
    def start_recording(self, frame, track_ids):
        """Start recording if not already recording"""
        if not self.recording:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_start_time = time.time()
            self.current_track_ids = set(track_ids)
            self.frame_count = 0
            
            h, w = frame.shape[:2]
            
            raw_path = os.path.join(self.output_dir, f"raw_{self.session_id}.mp4")
            self.raw_writer = cv2.VideoWriter(
                raw_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                25.0,
                (w, h)
            )
            
            annotated_path = os.path.join(self.output_dir, f"annotated_{self.session_id}.mp4")
            self.annotated_writer = cv2.VideoWriter(
                annotated_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                25.0,
                (w, h)
            )
            
            self.recording = True
            self.last_person_time = time.time()
            print(f"🎥 Recording started: {self.session_id}")
            print(f"   Raw: {raw_path}")
            print(f"   Annotated: {annotated_path}")
            return True
        return False
    
    def update(self, frame, annotated_frame, track_ids):
        """Update recording state"""
        current_time = time.time()
        
        # ✅ START RECORDING if there are people and not recording
        if not self.recording and track_ids:
            self.start_recording(frame, track_ids)
            # Write first frame immediately
            if self.recording:
                self.raw_writer.write(frame)
                self.annotated_writer.write(annotated_frame)
                return
        
        # Update last person time if there are people
        if track_ids:
            self.last_person_time = current_time
            self.current_track_ids = set(track_ids)
        
        # Check if we should stop recording
        if self.recording and (current_time - self.last_person_time > self.no_person_timeout):
            self.stop_recording()
            return
        
        # Write frames if recording
        if self.recording:
            self.raw_writer.write(frame)
            self.annotated_writer.write(annotated_frame)
            self.frame_count += 1
            
            # Update info on annotated frame
            duration = current_time - self.record_start_time
            cv2.putText(annotated_frame, f"REC {duration:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Frames: {self.frame_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def stop_recording(self):
        """Stop recording and release resources"""
        if self.recording:
            if self.raw_writer:
                self.raw_writer.release()
                self.raw_writer = None
            if self.annotated_writer:
                self.annotated_writer.release()
                self.annotated_writer = None
            
            duration = time.time() - self.record_start_time
            print(f"⏹️ Recording stopped: {self.session_id} (duration: {duration:.1f}s, frames: {self.frame_count})")
            self.recording = False
            self.session_id = None
            self.frame_count = 0
    
    def release(self):
        """Release all resources"""
        self.stop_recording()

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
            
            # if face_rgb.shape[0] > 100 or face_rgb.shape[1] > 100:
            #     face_rgb = cv2.resize(face_rgb, (100, 100))
            
            # face_locations = face_recognition.face_locations(face_rgb, model="hog")
            face_locations = face_recognition.face_locations(face_rgb)

            print("face_recognition")
            
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

# ==================== RTSP STREAM ====================
class RTSPStream:
    def __init__(self, rtsp_url, target_width=640, target_height=480, max_retries=5):
        self.rtsp_url = rtsp_url
        self.target_width = target_width
        self.target_height = target_height
        self.max_retries = max_retries
        self.cap = None
        self.retry_count = 0
        
    def connect(self):
        """Connect to RTSP stream"""
        print(f"📡 Connecting to RTSP: {self.rtsp_url}")
        
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if self.cap.isOpened():
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except:
                    pass
                
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                except:
                    pass
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"✅ Connected to RTSP stream")
                    return True
                else:
                    self.cap.release()
                    self.cap = None
            else:
                print("❌ Could not open RTSP stream")
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self.cap = None
        
        return False
    
    def read(self):
        """Read frame with reconnection"""
        if self.cap is None:
            if not self.connect():
                return False, None
        
        try:
            ret, frame = self.cap.read()
        except:
            ret = False
            frame = None
        
        if not ret or frame is None:
            self.retry_count += 1
            if self.retry_count > self.max_retries:
                return False, None
            
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            
            time.sleep(1)
            if self.connect():
                self.retry_count = 0
                return self.read()
            return False, None
        
        self.retry_count = 0
        
        if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        return True, frame
    
    def release(self):
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

# ==================== MAIN TRACKING CLASS ====================
class PersonTrackerWithFaces:
    def __init__(self, source=0):
        self.source = source
        
        # Use RTSPStream wrapper for RTSP
        if isinstance(source, str) and source.startswith('rtsp://'):
            self.stream = RTSPStream(source, target_width=800, target_height=600)
            if not self.stream.connect():
                print("❌ Failed to connect to RTSP stream")
                sys.exit(1)
            self.cap = None
            self.use_stream = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.use_stream = False
        
        self.model = YOLO('yolov8n.pt')
        
        # Video recorder
        self.recorder = VideoRecorder(output_dir="recordings")
        
        # Settings
        self.skip_frames = 3
        self.recognition_interval = 5
        self.target_width = 800
        self.target_height = 600
        
        self.entry_times = {}
        self.person_names = {}
        self.face_recognizer = FaceRecognizer()
        self.fps_calculator = FPS(avg_window=30)
        
        self.confidence_threshold = 0.3
        self.face_padding = 20
        self.face_upper_ratio = 0.9 # 0.4
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

    def save_face_debug(self, face_region, track_id):
            """Save face region to logs folder for debugging"""
            # Create logs directory if it doesn't exist
            log_dir = "logs/faces"
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate filename with timestamp and track_id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"track_{track_id}_{timestamp}.jpg"
            filepath = os.path.join(log_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, face_region)
            
            # Optional: Also save with bounding box info in a text file
            info_file = os.path.join(log_dir, "face_info.txt")
            with open(info_file, "a") as f:
                f.write(f"{filename}: track_id={track_id}, shape={face_region.shape}\n")


    def identify_person(self, frame, bbox, track_id):
        face_region = self.extract_face_region(frame, bbox)
        
        self.save_face_debug(face_region, track_id)

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
                
                name, face_conf = self.identify_person(frame, bbox, track_id)
                self.recognition_attempts[track_id] += 1
                
                if name != "Unknown":
                    self.person_names[track_id] = name
                    print(f"   ✅ Recognized as: {name} (confidence: {face_conf:.2f})")
                else:
                    print(f"   ❌ Face not recognized (will retry)")
            
            # Periodic recognition
            elif self.person_names.get(track_id) == "Unknown":
                if processed_count % self.recognition_interval == 0:
                    name, face_conf = self.identify_person(frame, bbox, track_id)
                    self.recognition_attempts[track_id] += 1
                    
                    if name != "Unknown":
                        self.person_names[track_id] = name
                        print(f"   ✅ RECOGNIZED! ID:{track_id} as: {name}")
            
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
        if self.use_stream:
            return self.stream.read()
        else:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
            return ret, frame
    
    def run(self):
        print("🎯 Starting tracking with face recognition...")
        print(f"📊 Known faces: {len(self.face_recognizer.known_face_names)}")
        print(f"⚡ Frame skip: every {self.skip_frames + 1}-th frame")
        print("📸 Press 'q' to exit")
        
        frame_count = 0
        processed_count = 0
        skipped_count = 0
        
        real_time_start = time.time()
        real_time_frames = 0
        
        while True:
            success, frame = self.read_frame()
            if not success:
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
            
            # Create annotated frame
            annotated_frame = frame.copy()
            track_ids = set()
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = set(results[0].boxes.id.cpu().numpy().astype(int))
                confidences = results[0].boxes.conf.cpu().numpy()
                
                self.process_tracked_people(annotated_frame, boxes, list(track_ids), confidences, processed_count)
            
            # Update info panel on annotated frame
            self.draw_info_panel(annotated_frame, current_fps, real_time_fps, 
                               frame_count, processed_count, skipped_count)
            
            # ===== VIDEO RECORDING =====
            self.recorder.update(frame, annotated_frame, track_ids)
            
            # Show video
            cv2.imshow("YOLOv8 + ByteTrack + Face Recognition", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Cleanup
        self.recorder.release()
        
        if self.use_stream:
            self.stream.release()
        else:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Log remaining
        for track_id, entry_time in list(self.entry_times.items()):
            self.log_person_exit(track_id, entry_time)
        
        print("\n👋 Program finished")

# ==================== MAIN ====================
if __name__ == "__main__":
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
        print("📁 Created 'known_faces' folder")
        print("📸 Add photos of known people")
        print("   Format: person_name.jpg")
    
    # RTSP or webcam
    # rtsp_url = 0  # Webcam
    # rtsp_url = "rtsp://192.168.0.102:8554/hikvision_room?mp4"
    rtsp_url = "rtsp://192.168.0.102:8554/balcony_camera_hero_4mp_wifi_h264?mp4"
    
    tracker = PersonTrackerWithFaces(source=rtsp_url)
    tracker.run()