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
        
        # Load or train known faces
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
        trained_count = 0
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.known_faces_dir, filename)
                
                # Load image
                image = face_recognition.load_image_file(image_path)
                
                # Detect faces
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    # Get encoding - use the first face found
                    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        trained_count += 1
                        print(f"   ✅ Trained: {name}")
                    else:
                        print(f"   ⚠️ Could not encode face in: {filename}")
                else:
                    print(f"   ⚠️ No face found in: {filename}")
        
        # Save encodings
        if self.known_face_encodings:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            print(f"💾 Saved {len(self.known_face_encodings)} face encodings")
        else:
            print(f"⚠️ No faces were trained! Please add clear face photos to {self.known_faces_dir}")
    
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
            print("Training from scratch...")
            self.train_known_faces()
    
    def recognize_face(self, face_image):
        """Recognize a face in the image with robust error handling"""
        try:
            if not self.known_face_encodings:
                return "Unknown", 0.0
            
            # Validate input
            if face_image is None or face_image.size == 0:
                return "Unknown", 0.0
            
            # Check image dimensions
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
            
            # Detect face locations
            face_locations = face_recognition.face_locations(face_rgb)
            
            if not face_locations:
                return "Unknown", 0.0
            
            # Get face encodings - FIXED: pass face_locations correctly
            face_encodings = face_recognition.face_encodings(face_rgb, known_face_locations=face_locations)
            
            if not face_encodings:
                return "Unknown", 0.0
            
            # Compare with known faces
            face_encoding = face_encodings[0]
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                confidence = 1 - distances[best_match_index]
                
                # Lower threshold for testing
                if confidence > 0.4:
                    return self.known_face_names[best_match_index], confidence
            
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
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.entry_times = {}
        self.person_names = {}  # track_id -> name
        self.face_recognizer = FaceRecognizer()
        self.fps_calculator = FPS(avg_window=30)
        
        # Create log file with headers
        with open("tracking_log.csv", "w") as f:
            f.write("Entry Time, Exit Time, Duration (s), Track ID, Person Name, Confidence\n")
    
    def log_person_exit(self, track_id, entry_time):
        exit_time = time.time()
        duration = exit_time - entry_time
        name = self.person_names.get(track_id, "Unknown")
        confidence = 0.0
        
        with open("tracking_log.csv", "a") as f:
            f.write(f"{datetime.fromtimestamp(entry_time)}, ")
            f.write(f"{datetime.fromtimestamp(exit_time)}, ")
            f.write(f"{duration:.2f}s, ID:{track_id}, {name}, {confidence:.2f}\n")
        
        print(f"🚶 {name} (ID:{track_id}) покинул кадр. Время: {duration:.2f}с")
        
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
        """Extract face from bbox and identify person"""
        x1, y1, x2, y2 = bbox
        padding = 20
        
        # Extract face region (upper part of the body)
        face_y1 = max(0, y1 - padding)
        face_y2 = min(frame.shape[0], y1 + int((y2 - y1) * 0.4))  # Upper 40% of body
        face_x1 = max(0, x1 - padding)
        face_x2 = min(frame.shape[1], x2 + padding)
        
        face_region = frame[face_y1:face_y2, face_x1:face_x2]
        
        # self.save_face_debug(face_region, track_id)

        if face_region.size == 0:
            return "Unknown", 0.0
        
        # Recognize face
        name, confidence = self.face_recognizer.recognize_face(face_region)
        
        if name and confidence > 0.5:
            return name, confidence
        else:
            return "Unknown", 0.0
    
    def run(self):
        print("🎯 Запуск трекинга с распознаванием лиц...")
        print(f"📊 Известных лиц в базе: {len(self.face_recognizer.known_face_names)}")
        print("📸 Нажмите 'q' для выхода")
        
        frame_count = 0
        last_recognition_time = {}
          
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            frame_count += 1
            current_fps = self.fps_calculator.update()
            
            # Запуск детекции и трекинга
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                      classes=[0], verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                current_ids = set(track_ids)
                
                # Проверяем, кто вышел
                for track_id in list(self.entry_times.keys()):
                    if track_id not in current_ids:
                        self.log_person_exit(track_id, self.entry_times.pop(track_id))
                        if track_id in self.person_names:
                            del self.person_names[track_id]


                # Обрабатываем каждого человека
                for i, track_id in enumerate(track_ids):
                    # Регистрируем нового человека
                    if track_id not in self.entry_times:
                        self.entry_times[track_id] = time.time()
                        self.person_names[track_id] = "Unknown"  # Initialize with Unknown
                        last_recognition_time[track_id] = 0  # Initialize with 0
                        print(f"🆕 Зарегистрирован новый человек. ID: {track_id}")
                        
                        # Попытка распознать лицо при появлении
                        name, face_conf = self.identify_person(frame, boxes[i], track_id)
                        if name and name != "Unknown":
                            self.person_names[track_id] = name
                            print(f"   ✅ Распознан как: {name} (уверенность: {face_conf:.2f})")
                        else:
                            print("   ❌ Лицо не распознано (будет повторная попытка)")
                    
                    # Периодическое распознавание (каждые 30 кадров)
                    elif frame_count % 30 == 0:
                        # Check if enough time has passed since last recognition
                        current_time = time.time()
                        if current_time - last_recognition_time.get(track_id, 0) > 1.0:  # At least 1 second between attempts
                            name, face_conf = self.identify_person(frame, boxes[i], track_id)
                            last_recognition_time[track_id] = current_time
                            
                            if name and name != "Unknown":
                                self.person_names[track_id] = name
                                print(f"   🔄 Перераспознан как: {name} (уверенность: {face_conf:.2f})")
                            else:
                                print("   ❌ Лицо не распознано (будет повторная попытка)")                                
                        
                    # Рисуем рамку и информацию
                    x1, y1, x2, y2 = boxes[i]
                    # Цвет рамки: зеленый для известных, красный для неизвестных
                    color = (0, 255, 0) if self.person_names.get(track_id, "Unknown") != "Unknown" else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Имя и ID
                    name = self.person_names.get(track_id, "Unknown")
                    label = f"{name} (ID:{track_id})"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Уверенность детекции
                    cv2.putText(frame, f"det: {confidences[i]:.2f}", (x1, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Добавляем информацию о базе лиц
            cv2.putText(frame, f"Known faces: {len(self.face_recognizer.known_face_names)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Показываем FPS
            cv2.putText(frame, f"People: {len(self.entry_times)}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)            
            
            cv2.imshow("YOLOv8 + ByteTrack + Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Закрытие
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Логируем оставшихся людей
        for track_id, entry_time in list(self.entry_times.items()):
            self.log_person_exit(track_id, entry_time)
        
        print("\n👋 Программа завершена")
        print("📊 Результаты сохранены в tracking_log.csv")

# ==================== MAIN ====================
if __name__ == "__main__":
    # Создаем папку для известных лиц, если её нет
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
        print("📁 Создана папка 'known_faces'")
        print("📸 Добавьте в нее фотографии известных людей")
        print("   Формат: имя_человека.jpg")
    
    tracker = PersonTrackerWithFaces(source=0)  # или 'video.mp4'
    tracker.run()