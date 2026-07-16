import cv2
from fps import FPS
from ultralytics import YOLO
import time
from datetime import datetime
import face_recognition
import numpy as np
import os
import pickle

# ==================== FACE RECOGNITION SETUP ====================
class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", encodings_file="face_encodings.pkl"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.encodings_file = encodings_file
        self.known_faces_dir = known_faces_dir
        self.recognition_cache = {}
        self.cache_timeout = 5  # Cache timeout in seconds
        
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
        """Recognize a face with caching for speed"""
        try:
            # Check cache first
            # if track_id and track_id in self.recognition_cache:
            #     cached_name, cached_conf, cached_time = self.recognition_cache[track_id]
            #     if time.time() - cached_time < self.cache_timeout:
            #         return cached_name, cached_conf
            
            if not self.known_face_encodings:
                return "Unknown", 0.0
            
            # Validate input
            if face_image is None or face_image.size == 0:
                return "Unknown", 0.0
            
            if len(face_image.shape) != 3:
                return "Unknown", 0.0
            
            h, w = face_image.shape[:2]
            if h < 20 or w < 20:
                return "Unknown", 0.0
            
            # Convert BGR to RGB (face_recognition expects RGB)
            try:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            except:
                face_rgb = face_image
            
            # Resize for faster processing
            # if face_rgb.shape[0] > 100 or face_rgb.shape[1] > 100:
            #     face_rgb = cv2.resize(face_rgb, (100, 100))
            
            # Detect face locations using HOG (faster than CNN)
            face_locations = face_recognition.face_locations(face_rgb, model="hog")
            
            if not face_locations:
                return "Unknown", 0.0
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(
                face_rgb, 
                known_face_locations=face_locations,
                num_jitters=0  # Faster
            )
            
            if not face_encodings:
                return "Unknown", 0.0
            
            # Compare with known faces
            face_encoding = face_encodings[0]
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                confidence = 1 - distances[best_match_index]
                
                if confidence > 0.4:
                    name = self.known_face_names[best_match_index]
                    # Cache the result
                    if track_id:
                        self.recognition_cache[track_id] = (name, confidence, time.time())
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
        self.cap = cv2.VideoCapture(source)
        
        # RTSP optimizations
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Frame skipping settings for better FPS
        self.skip_frames = 1  # Process every 2nd frame (0 = process all)
        self.recognition_interval = 30  # Recognize every 10th PROCESSED frame
        self.target_width = 640
        self.target_height = 480
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.entry_times = {}
        self.person_names = {}
        self.face_recognizer = FaceRecognizer()
        self.fps_calculator = FPS(avg_window=30)
        
        # Recognition settings
        self.confidence_threshold = 0.4
        self.face_padding = 20
        self.face_upper_ratio = 0.4
        
        # Create log file
        with open("tracking_log.csv", "w") as f:
            f.write("Entry Time, Exit Time, Duration (s), Track ID, Person Name, Confidence\n")
    
    def log_person_exit(self, track_id, entry_time):
        """Log person exit"""
        exit_time = time.time()
        duration = exit_time - entry_time
        name = self.person_names.get(track_id, "Unknown")
        
        with open("tracking_log.csv", "a") as f:
            f.write(f"{datetime.fromtimestamp(entry_time)}, ")
            f.write(f"{datetime.fromtimestamp(exit_time)}, ")
            f.write(f"{duration:.2f}s, ID:{track_id}, {name}, 0.00\n")
        
        print(f"🚶 {name} (ID:{track_id}) left frame. Duration: {duration:.2f}s")
    
    def extract_face_region(self, frame, bbox):
        """Extract face region from bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Use upper part of the body for face detection
        face_y1 = max(0, y1 - self.face_padding)
        face_y2 = min(frame.shape[0], y1 + int((y2 - y1) * self.face_upper_ratio))
        face_x1 = max(0, x1 - self.face_padding)
        face_x2 = min(frame.shape[1], x2 + self.face_padding)
        
        return frame[face_y1:face_y2, face_x1:face_x2]
    
    def identify_person(self, frame, bbox, track_id):
        """Identify person by face recognition"""
        face_region = self.extract_face_region(frame, bbox)
        
        self.save_face_debug(face_region, track_id)

        if face_region.size == 0:
            return "Unknown", 0.0
        
        name, confidence = self.face_recognizer.recognize_face(face_region, track_id)
        return name, confidence
    
    def draw_person_info(self, frame, bbox, track_id, confidence, name):
        """Draw person information on frame"""
        x1, y1, x2, y2 = bbox
        is_known = name != "Unknown"
        color = (0, 255, 0) if is_known else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw name and ID
        label = f"{name} (ID:{track_id})"
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw detection confidence
        cv2.putText(frame, f"det: {confidence:.2f}", (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_tracked_people(self, frame, boxes, track_ids, confidences, processed_count, last_recognition_time):
        """Process all tracked people"""
        current_ids = set(track_ids)
        
        # Check for people who left
        for track_id in list(self.entry_times.keys()):
            if track_id not in current_ids:
                self.log_person_exit(track_id, self.entry_times.pop(track_id))
                self.person_names.pop(track_id, None)
                last_recognition_time.pop(track_id, None)
        
        # Process each person
        for i, track_id in enumerate(track_ids):
            bbox = boxes[i]
            
            # New person - always try to recognize
            if track_id not in self.entry_times:
                self._register_new_person(frame, bbox, track_id, last_recognition_time)
            else:
                # Periodic recognition for unknown people
                self._periodic_recognition(frame, bbox, track_id, processed_count, last_recognition_time)
            
            # Draw person info
            name = self.person_names.get(track_id, "Unknown")
            self.draw_person_info(frame, bbox, track_id, confidences[i], name)
    
    def _register_new_person(self, frame, bbox, track_id, last_recognition_time):
        """Register a new person and try to recognize"""
        self.entry_times[track_id] = time.time()
        self.person_names[track_id] = "Unknown"
        last_recognition_time[track_id] = 0
        print(f"🆕 New person registered. ID: {track_id}")
        
        # Try to recognize immediately
        name, face_conf = self.identify_person(frame, bbox, track_id)
        if name != "Unknown":
            self.person_names[track_id] = name
            print(f"   ✅ Recognized as: {name} (confidence: {face_conf:.2f})")
        else:
            print("   ❌ Face not recognized (will retry)")

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

    def _periodic_recognition(self, frame, bbox, track_id, processed_count, last_recognition_time):
        """Periodic recognition for unknown people"""

        # Only try to recognize unknown people
        # if self.person_names.get(track_id) != "Unknown":
        #     print(0)
        #     return
        
        # Check if we should recognize this frame
        if processed_count % self.recognition_interval != 0:
            print(1, processed_count, processed_count % self.recognition_interval)
            return
        
        # Check time since last attempt
        current_time = time.time()
        if current_time - last_recognition_time.get(track_id, 0) <= 1.0:
            print(2)
            return
        
        # Try to recognize
        name, face_conf = self.identify_person(frame, bbox, track_id)
        last_recognition_time[track_id] = current_time
        
        if name != "Unknown":
            self.person_names[track_id] = name
            print(f"   🔄 Re-recognized as: {name} (confidence: {face_conf:.2f})")
    
    def draw_info_panel(self, frame, current_fps, real_time_fps, frame_count, processed_count, skipped):
        """Draw information panel on frame"""
        y_pos = 30
        line_height = 35
        
        info = [
            (f"FPS (process): {current_fps:.1f}", (0, 255, 255)),
            (f"FPS (display): {real_time_fps:.1f}", (100, 255, 100)),
            (f"People: {len(self.entry_times)}", (255, 255, 255)),
            (f"Known faces: {len(self.face_recognizer.known_face_names)}", (255, 255, 255)),
            (f"Frame: {frame_count} (proc: {processed_count})", (200, 200, 200)),
            (f"Skipped: {skipped}", (200, 200, 200)),
        ]
        
        for text, color in info:
            cv2.putText(frame, text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += line_height
    
    def run(self):
        """Main program loop with frame skipping"""
        print("🎯 Starting tracking with face recognition...")
        print(f"📊 Known faces in database: {len(self.face_recognizer.known_face_names)}")
        print(f"⚡ Frame skip: every {self.skip_frames + 1}-th frame")
        print("📸 Press 'q' to exit")
        
        frame_count = 0
        processed_count = 0
        skipped_count = 0
        last_recognition_time = {}
        
        # Real-time FPS tracking
        real_time_start = time.time()
        real_time_frames = 0
        display_fps = 0
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # ===== FRAME SKIPPING =====
            # Skip frames to increase FPS
            if frame_count % (self.skip_frames + 1) != 0:
                skipped_count += 1
                # Show frame but don't process (for visual display)
                display_frame = cv2.resize(frame, (self.target_width, self.target_height))
                cv2.putText(display_frame, f"SKIPPED {frame_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("YOLOv8 + ByteTrack + Face Recognition", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            
            # ===== PROCESS FRAME =====
            processed_count += 1
            
            # Resize for faster processing
            frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # Calculate real-time FPS
            real_time_frames += 1
            if time.time() - real_time_start >= 1.0:
                real_time_fps = real_time_frames / (time.time() - real_time_start)
                real_time_frames = 0
                real_time_start = time.time()
            else:
                real_time_fps = 0
            
            current_fps = self.fps_calculator.update()
            
            # Run YOLO detection and tracking
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                      classes=[0], verbose=False)
            
            # Process results
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Process tracked people with processed_count (not frame_count!)
                self.process_tracked_people(frame, boxes, track_ids, confidences, 
                                           processed_count, last_recognition_time)
            
            # Draw info panel
            self.draw_info_panel(frame, current_fps, real_time_fps, 
                               frame_count, processed_count, skipped_count)
            
            cv2.imshow("YOLOv8 + ByteTrack + Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Log remaining people
        for track_id, entry_time in list(self.entry_times.items()):
            self.log_person_exit(track_id, entry_time)
        
        print("\n👋 Program finished")
        print(f"📊 Total frames: {frame_count}")
        print(f"📊 Processed: {processed_count}")
        print(f"📊 Skipped: {skipped_count}")
        print(f"📊 Skip rate: {(skipped_count/frame_count*100):.1f}%")
        print("📊 Results saved to tracking_log.csv")

# ==================== MAIN ====================
if __name__ == "__main__":
    # Create known faces directory if it doesn't exist
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
        print("📁 Created 'known_faces' folder")
        print("📸 Add photos of known people")
        print("   Format: person_name.jpg")
    
    # Use RTSP stream or webcam
    rtsp_url = "rtsp://192.168.0.102:8554/hikvision_room?mp4"
    # rtsp_url = 0  # Webcam
    # rtsp_url = "rtsp://192.168.0.102:8554/balcony_camera_hero_4mp_wifi_h264"
    
    tracker = PersonTrackerWithFaces(source=rtsp_url)
    tracker.run()