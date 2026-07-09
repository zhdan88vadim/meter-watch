from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import time
from datetime import datetime
from typing import Dict, Any
from app.config import config
from app.redis_manager import RedisManager
from app.telegram_bot import telegram_bot
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Состояние системы
system_status = {
    'recording': False,
    'tracking': True,
    'last_updated': time.time()
}

def require_auth():
    """Проверка API ключа"""
    api_key = request.headers.get('X-API-Key')
    if api_key != config.API_SECRET_KEY:
        return False
    return True

@app.route('/api/status', methods=['GET'])
def get_status():
    """Получение статуса системы"""
    # if not require_auth():
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    gas_status = RedisManager.get_key(config.REDIS_KEYS['gas_flow'])
    last_seen = RedisManager.get_key(config.REDIS_KEYS['human_last_seen'])
    alert_active = RedisManager.key_exists(config.REDIS_KEYS['alert_triggered'])
    startup_mode = RedisManager.key_exists(config.REDIS_KEYS['startup'])
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'recording': system_status['recording'],
            'tracking': system_status['tracking'],
            'startup_mode': startup_mode
        },
        'gas': {
            'flowing': gas_status == '1',
            'status': gas_status
        },
        'person': {
            'last_seen': float(last_seen) if last_seen else None,
            'last_seen_str': datetime.fromtimestamp(float(last_seen)).strftime('%H:%M:%S') if last_seen else None,
            'is_present': last_seen and (time.time() - float(last_seen) < 300)
        },
        'alert': {
            'active': alert_active,
            'cooldown': RedisManager.key_exists(config.REDIS_KEYS['alert_cooldown'])
        }
    }
    
    return jsonify(status)

@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    """Начать запись видео"""
    # if not require_auth():
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    system_status['recording'] = True
    system_status['last_updated'] = time.time()
    
    # Публикуем событие в Redis
    RedisManager.publish('system:commands', json.dumps({
        'command': 'start_recording',
        'timestamp': time.time()
    }))
    
    return jsonify({
        'success': True,
        'message': 'Recording started',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    """Остановить запись видео"""
    # if not require_auth():
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    system_status['recording'] = False
    system_status['last_updated'] = time.time()
    
    RedisManager.publish('system:commands', json.dumps({
        'command': 'stop_recording',
        'timestamp': time.time()
    }))
    
    return jsonify({
        'success': True,
        'message': 'Recording stopped',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/alert/reset', methods=['POST'])
def reset_alert():
    """Сбросить тревогу"""
    # if not require_auth():
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    RedisManager.delete_key(config.REDIS_KEYS['alert_triggered'])
    RedisManager.delete_key(config.REDIS_KEYS['alert_cooldown'])
    RedisManager.set_key(config.REDIS_KEYS['human_last_seen'], str(time.time()))
    
    return jsonify({
        'success': True,
        'message': 'Alert reset',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/recordings', methods=['GET'])
def get_recordings():
    """Получить список записей"""
    # if not require_auth():
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    keys = RedisManager.get_connection().keys(f"{config.REDIS_KEYS['recording_prefix']}*")
    recordings = []
    
    for key in keys:
        data = RedisManager.hgetall(key)
        if data:
            recordings.append({
                'id': key.replace(config.REDIS_KEYS['recording_prefix'], ''),
                'filename': data.get('filename', ''),
                'start_time': float(data.get('start_time', 0)),
                'end_time': float(data.get('end_time', 0)),
                'duration': float(data.get('duration', 0)),
                'person_id': data.get('person_id', '')
            })
    
    # Сортируем по времени начала
    recordings.sort(key=lambda x: x['start_time'], reverse=True)
    
    return jsonify({
        'recordings': recordings,
        'total': len(recordings)
    })

@app.route('/api/recordings/<recording_id>', methods=['GET'])
def get_recording(recording_id):
    """Получить информацию о записи"""
    # if not require_auth():
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    key = f"{config.REDIS_KEYS['recording_prefix']}{recording_id}"
    data = RedisManager.hgetall(key)
    
    if not data:
        return jsonify({'error': 'Recording not found'}), 404
    
    return jsonify({
        'id': recording_id,
        'filename': data.get('filename', ''),
        'start_time': float(data.get('start_time', 0)),
        'end_time': float(data.get('end_time', 0)),
        'duration': float(data.get('duration', 0)),
        'person_id': data.get('person_id', '')
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Получить статистику"""
    # if not require_auth():
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    # Получаем историю
    history = RedisManager.get_connection().lrange(
        config.REDIS_KEYS['detection_history'], 0, 99
    )
    
    entries = 0
    exits = 0
    total_duration = 0
    
    for item in history:
        try:
            event = json.loads(item)
            if event['type'] == 'person_entered':
                entries += 1
            elif event['type'] == 'person_exited':
                exits += 1
                total_duration += event['data'].get('duration', 0)
        except:
            pass
    
    return jsonify({
        'total_entries': entries,
        'total_exits': exits,
        'avg_duration': total_duration / exits if exits > 0 else 0,
        'recordings_count': len(RedisManager.get_connection().keys(f"{config.REDIS_KEYS['recording_prefix']}*"))
    })

@app.route('/api/system/control', methods=['POST'])
def system_control():
    """Управление системой"""
    # if not require_auth():
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    action = data.get('action')
    
    if action == 'restart':
        # Перезапуск системы
        RedisManager.delete_key(config.REDIS_KEYS['startup'])
        RedisManager.set_timestamp_key(config.REDIS_KEYS['startup'], config.STARTUP_DURATION)
        
        return jsonify({
            'success': True,
            'message': 'System restarted',
            'startup_mode': True
        })
    
    elif action == 'silence':
        # Отключить звук
        RedisManager.set_key(config.REDIS_KEYS['alert_cooldown'], '1', config.ALERT_COOLDOWN)
        
        return jsonify({
            'success': True,
            'message': 'Alert silenced for 10 minutes'
        })
    
    return jsonify({'error': 'Invalid action'}), 400

def start_api():
    """Запускает API сервер"""
    app.run(
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        debug=config.WEB_DEBUG,
        threaded=True
    )