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
        self.cache_timeout = 30
        
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
        try:
            if track_id and track_id in self.recognition_cache:
                cached_name, cached_conf, cached_time = self.recognition_cache[track_id]
                if time.time() - cached_time < self.cache_timeout:
                    return cached_name, cached_conf
            
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
                
                if confidence > 0.4:
                    name = self.known_face_names[best_match_index]
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
        
        # ⚡ RTSP оптимизации
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # ⚡ Настройки пропуска кадров
        self.skip_frames = 1  # Обрабатываем каждый 3-й кадр
        self.recognition_interval = 10  # Распознаем каждый 10-й кадр (из обработанных)
        self.target_width = 640
        self.target_height = 480
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.entry_times = {}
        self.person_names = {}
        self.face_recognizer = FaceRecognizer()
        self.fps_calculator = FPS(avg_window=30)
        
        # ⚡ Для отслеживания реального времени
        self.frame_timestamp = 0
        self.last_processed_frame_time = 0
        self.video_start_time = None
        self.processing_start_time = None
        self.time_multiplier = 1.0
        
        # Настройки
        # self.recognition_interval = 30
        self.confidence_threshold = 0.4
        self.face_padding = 20
        self.face_upper_ratio = 0.4
        
        with open("tracking_log.csv", "w") as f:
            f.write("Entry Time, Exit Time, Duration (s), Track ID, Person Name, Confidence\n")
    
    def log_person_exit(self, track_id, entry_time):
        exit_time = time.time()
        duration = exit_time - entry_time
        name = self.person_names.get(track_id, "Unknown")
        
        with open("tracking_log.csv", "a") as f:
            f.write(f"{datetime.fromtimestamp(entry_time)}, ")
            f.write(f"{datetime.fromtimestamp(exit_time)}, ")
            f.write(f"{duration:.2f}s, ID:{track_id}, {name}, 0.00\n")
        
        print(f"🚶 {name} (ID:{track_id}) покинул кадр. Время: {duration:.2f}с")
    
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"det: {confidence:.2f}", (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_tracked_people(self, frame, boxes, track_ids, confidences, frame_count, last_recognition_time):
        current_ids = set(track_ids)
        
        for track_id in list(self.entry_times.keys()):
            if track_id not in current_ids:
                self.log_person_exit(track_id, self.entry_times.pop(track_id))
                self.person_names.pop(track_id, None)
                last_recognition_time.pop(track_id, None)
        
        for i, track_id in enumerate(track_ids):
            bbox = boxes[i]
            
            if track_id not in self.entry_times:
                self._register_new_person(frame, bbox, track_id, last_recognition_time)
            else:
                self._periodic_recognition(frame, bbox, track_id, frame_count, last_recognition_time)
            
            name = self.person_names.get(track_id, "Unknown")
            self.draw_person_info(frame, bbox, track_id, confidences[i], name)
    
    def _register_new_person(self, frame, bbox, track_id, last_recognition_time):
        self.entry_times[track_id] = time.time()
        self.person_names[track_id] = "Unknown"
        last_recognition_time[track_id] = 0
        print(f"🆕 Зарегистрирован новый человек. ID: {track_id}")
        
        name, face_conf = self.identify_person(frame, bbox, track_id)
        if name != "Unknown":
            self.person_names[track_id] = name
            print(f"   ✅ Распознан как: {name} (уверенность: {face_conf:.2f})")
        else:
            print("   ❌ Лицо не распознано (будет повторная попытка)")
    
    def _periodic_recognition(self, frame, bbox, track_id, frame_count, last_recognition_time):
        if frame_count % self.recognition_interval != 0:
            return
        
        current_time = time.time()
        if current_time - last_recognition_time.get(track_id, 0) <= 1.0:
            return
        
        name, face_conf = self.identify_person(frame, bbox, track_id)
        last_recognition_time[track_id] = current_time
        
        if name != "Unknown" and self.person_names.get(track_id) == "Unknown":
            self.person_names[track_id] = name
            print(f"   🔄 Перераспознан как: {name} (уверенность: {face_conf:.2f})")
    
    def draw_info_panel(self, frame, current_fps, real_time_fps):
        """Отрисовка информационной панели с реальным временем"""
        y_pos = 30
        line_height = 35
        
        info = [
            (f"FPS: {current_fps:.1f}", (0, 255, 255)),
            (f"Real-time FPS: {real_time_fps:.1f}", (100, 255, 100)),
            (f"People: {len(self.entry_times)}", (255, 255, 255)),
            (f"Known faces: {len(self.face_recognizer.known_face_names)}", (255, 255, 255)),
        ]
        
        for text, color in info:
            cv2.putText(frame, text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += line_height
    
    def run(self):
        """Основной цикл программы с пропуском кадров"""
        print("🎯 Запуск трекинга с распознаванием лиц...")
        print(f"📊 Известных лиц в базе: {len(self.face_recognizer.known_face_names)}")
        print(f"⚡ Пропуск кадров: каждый {self.skip_frames + 1}-й кадр")
        print("📸 Нажмите 'q' для выхода")
        
        frame_count = 0
        processed_count = 0
        last_recognition_time = {}
        
        # ⚡ Для отслеживания реального времени
        real_time_start = time.time()
        real_time_frames = 0
        display_fps = 0
        last_display_update = time.time()
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # ⚡ ПРОПУСК КАДРОВ: обрабатываем только каждый N-й кадр
            if frame_count % (self.skip_frames + 1) != 0:
                # Показываем кадр, но не обрабатываем (для визуального отображения)
                # Ресайз для отображения
                display_frame = cv2.resize(frame, (self.target_width, self.target_height))
                cv2.putText(display_frame, f"SKIP {frame_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("YOLOv8 + ByteTrack + Face Recognition", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            
            # ⚡ ОБРАБОТКА КАДРА
            processed_count += 1
            
            # Ресайз для обработки
            frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # ⚡ Расчет реального FPS
            real_time_frames += 1
            if time.time() - real_time_start >= 1.0:
                real_time_fps = real_time_frames / (time.time() - real_time_start)
                real_time_frames = 0
                real_time_start = time.time()
            else:
                real_time_fps = 0
            
            current_fps = self.fps_calculator.update()
            
            # Запуск детекции и трекинга
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                      classes=[0], verbose=False)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                self.process_tracked_people(frame, boxes, track_ids, confidences, 
                                           processed_count, last_recognition_time)
            
            # ⚡ Добавляем информацию о пропущенных кадрах
            self.draw_info_panel(frame, current_fps, real_time_fps)
            cv2.putText(frame, f"Frame: {frame_count} (processed: {processed_count})", 
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Skip: {self.skip_frames}", 
                        (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("YOLOv8 + ByteTrack + Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        for track_id, entry_time in list(self.entry_times.items()):
            self.log_person_exit(track_id, entry_time)
        
        print("\n👋 Программа завершена")
        print(f"📊 Всего кадров: {frame_count}")
        print(f"📊 Обработано: {processed_count}")
        print(f"📊 Пропущено: {frame_count - processed_count}")
        print("📊 Результаты сохранены в tracking_log.csv")

# ==================== MAIN ====================
if __name__ == "__main__":
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
        print("📁 Создана папка 'known_faces'")
        print("📸 Добавьте в нее фотографии известных людей")
        print("   Формат: имя_человека.jpg")

    rtsp_url = "rtsp://192.168.0.102:8554/hikvision_room?mp4"    

    rtsp_url = 0
    tracker = PersonTrackerWithFaces(source=rtsp_url) 
    tracker.run()