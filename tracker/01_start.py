#  redisObject.setex(key, time=timeout, value=value)

import cv2
from ultralytics import YOLO
import time
from datetime import datetime

# 1. Загрузка модели. Встроенный трекер ByteTrack в YOLOv8 
#    активируется параметром tracker="bytetrack.yaml" [citation:4]
model = YOLO('yolov8n.pt') 

# Словарь для хранения времени входа каждого трека
# Ключ: track_id, Значение: время первого появления
entry_times = {} 

# Функция для записи лога
def log_person_exit(track_id, entry_time):
    exit_time = time.time()
    duration = exit_time - entry_time
    with open("tracking_log.csv", "a") as f:
        f.write(f"{datetime.fromtimestamp(entry_time)}, {datetime.fromtimestamp(exit_time)}, {duration:.2f}s, ID:{track_id}\n")
    print(f"Человек {track_id} покинул кадр. Время: {duration:.2f} секунд.")

# Источник: 0 для веб-камеры или путь к файлу
source = 0  # или 'video.mp4'
cap = cv2.VideoCapture(source)

# Для видеофайла можно получить FPS для корректного времени
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. Запуск детекции и трекинга. classes=[0] для поиска только людей [citation:4]
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

    if results[0].boxes.id is not None:
        # Получаем ID и координаты боксов
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        current_ids = set(track_ids)
        previous_ids = set(entry_times.keys())

        # 3. Логика логирования: проверяем, кто из tracked_ids вышел
        for track_id in list(entry_times.keys()):
            if track_id not in current_ids:
                log_person_exit(track_id, entry_times.pop(track_id))

        # 4. Регистрируем новых людей, появившихся в этом кадре
        for i, track_id in enumerate(track_ids):
            if track_id not in entry_times:
                entry_times[track_id] = time.time()
                print(f"Зарегистрирован новый человек. ID: {track_id}")

            # Рисуем рамку и ID
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 + ByteTrack", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Закрываем все окна и освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()