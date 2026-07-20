from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from datetime import datetime
from meter_watch_shared.config import config
from meter_watch_shared.redis_manager import RedisManager
from app.state_manager import StateManager
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

state_manager = StateManager()

def require_auth():
    """Проверка API ключа"""
    api_key = request.headers.get('X-API-Key')
    if api_key != config.API_SECRET_KEY:
        return False
    return True

@app.route('/api/status', methods=['GET'])
def get_status():
    """Получение статуса системы"""   
    gas_status = RedisManager.get_key(config.REDIS_KEYS['gas_flow'])
    last_seen = RedisManager.get_key(config.REDIS_KEYS['human_last_seen'])
    alert_active = RedisManager.key_exists(config.REDIS_KEYS['alert_triggered'])
    startup_mode = RedisManager.key_exists(config.REDIS_KEYS['startup'])
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'startup_mode': startup_mode
        },
        'gas': {
            'flowing': gas_status == '1',
        },
        'person': {
            'last_seen': float(last_seen) if last_seen else None,
            'last_seen_str': datetime.fromtimestamp(float(last_seen)).strftime('%H:%M:%S') if last_seen else None,
            # 'is_present': last_seen and (time.time() - float(last_seen) < config.PERSON_ABSENCE_THRESHOLD),
            'is_active': last_seen and (time.time() - float(last_seen) < config.PERSON_IS_ACTIVE_THRESHOLD)
        },
        'alert': {
            'active': alert_active,
            'cooldown': RedisManager.key_exists(config.REDIS_KEYS['alert_cooldown'])
        }
    }
    
    return jsonify(status)

@app.route('/api/alert/reset', methods=['POST'])
def reset_alert():
    """Сбросить тревогу"""
    state_manager.reset_alert()
    
    return jsonify({
        'success': True,
        'message': 'Alert reset',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/system/control', methods=['POST'])
def system_control():
    """Управление системой"""
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