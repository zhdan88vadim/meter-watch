from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import time
import json
from datetime import datetime
from app.config import config
from app.redis_manager import RedisManager
import logging
import os

logger = logging.getLogger(__name__)

app = Flask(__name__, 
    template_folder='../templates',
    static_folder='../static'
)

# Глобальная ссылка на поток видео
video_capture = None

@app.route('/')
def dashboard():
    """Главная страница дашборда"""
    return render_template('dashboard.html')

@app.route('/api/stream_status')
def stream_status():
    """Статус видеопотока"""
    return jsonify({
        'active': video_capture is not None,
        'timestamp': datetime.now().isoformat()
    })

def generate_frames():
    """Генерация видеопотока"""
    global video_capture
    
    if video_capture is None or not video_capture.isOpened():
        return
    
    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        # Пропуск кадров для веб-интерфейса
        frame_count += 1
        if frame_count % config.FRAME_SKIP != 0:
            continue
        
        # Добавляем информацию на кадр
        gas_status = RedisManager.get_key(config.REDIS_KEYS['gas_flow'])
        last_seen = RedisManager.get_key(config.REDIS_KEYS['human_last_seen'])
        
        # Оверлей
        if gas_status == '1':
            cv2.putText(frame, "🔥 GAS FLOWING", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if last_seen:
            time_ago = time.time() - float(last_seen)
            if time_ago < 60:
                cv2.putText(frame, f"👤 Person detected {int(time_ago)}s ago", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"👤 No person {int(time_ago/60)}min", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Кодируем в JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Видеопоток для веб-интерфейса"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/realtime_stats')
def realtime_stats():
    """Реалтайм статистика для веб-интерфейса"""
    gas_status = RedisManager.get_key(config.REDIS_KEYS['gas_flow'])
    last_seen = RedisManager.get_key(config.REDIS_KEYS['human_last_seen'])
    alert_active = RedisManager.key_exists(config.REDIS_KEYS['alert_triggered'])
    
    stats = {
        'timestamp': time.time(),
        'gas_flowing': gas_status == '1',
        'person_present': last_seen and (time.time() - float(last_seen) < 60),
        'last_seen': float(last_seen) if last_seen else None,
        'alert_active': alert_active,
        'recording_active': False  # Будет обновляться из внешнего состояния
    }
    
    return jsonify(stats)

@app.route('/api/recordings/list')
def list_recordings():
    """Список записей"""
    keys = RedisManager.get_connection().keys(f"{config.REDIS_KEYS['recording_prefix']}*")
    recordings = []
    
    for key in keys:
        data = RedisManager.hgetall(key)
        if data:
            recordings.append({
                'id': key.replace(config.REDIS_KEYS['recording_prefix'], ''),
                'filename': data.get('filename', ''),
                'start_time': float(data.get('start_time', 0)),
                'duration': float(data.get('duration', 0))
            })
    
    recordings.sort(key=lambda x: x['start_time'], reverse=True)
    return jsonify(recordings[:10])  # Последние 10 записей

def start_web_dashboard(capture=None):
    """Запускает веб-дашборд"""
    global video_capture
    video_capture = capture
    
    app.run(
        host=config.WEB_HOST,
        port=config.WEB_PORT + 1,  # Используем другой порт для дашборда
        debug=config.WEB_DEBUG,
        threaded=True
    )