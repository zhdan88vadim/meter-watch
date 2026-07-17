import time
import threading
from utils.number_utils import list_to_number
from configuration import Config
from utils.api_utils import fetch_image, timestamp_ms
from services.recognition import recognize_image
from utils.log_data import save_test_image
from services.config import ConfigKeys, config
from meter_watch_shared.config import config as meter_watch_shared_config
from meter_watch_shared.redis_manager import RedisManager

history = []
last_recognized_digits = []
last_image = None
last_update_value = None
history_lock = threading.Lock()
last_nearly_activity_data = None
last_nearly_activity_counter = 0

def check_history_and_save_if_needed(new_digits, img):
    """Check if digits should be saved"""
    global last_recognized_digits, history, last_nearly_activity_counter, last_image

    if len(last_recognized_digits) > 0:
        last_number = list_to_number(last_recognized_digits)
        new_number = list_to_number(new_digits)
        difference = abs(new_number - last_number)
        
        if difference > 10:
            save_test_image(img, new_digits, f"big_diff_{difference}")
            if last_image is not None:
                save_test_image(last_image, new_digits, f"big_diff_{difference}")

        if new_number >= last_number:
            print(f"{new_number} is greater than {last_number}")
        else:
            print(f"{new_number} is not greater than {last_number}")
            save_test_image(img, new_digits, "less")            
            if last_image is not None:          
                save_test_image(last_image, new_digits, "less")
        
        last_image = img            

def monitor_loop():
    """
    Every minute, it requests an image from the camera, processes it, and saves it to history if any changes occur.
    """
    global last_recognized_digits, history, last_nearly_activity_data, last_nearly_activity_counter, last_update_value
    
    consecutive_failures = 0
    
    while True:
        # try:
        print("Запрос изображения с камеры...")
        
        img = fetch_image(config.get(ConfigKeys.CAMERA_URL) + str(timestamp_ms()))
        
        if img is None:
            print("Не удалось получить изображение, пропуск цикла.")
            consecutive_failures += 1
        else:
            consecutive_failures = 0

            result, min_conf = recognize_image(img)

            new_digits = list(result['full_number'])

            if min_conf < config.get(ConfigKeys.SAVE_THRESHOLD):
                save_test_image(img, result['full_number'], "low_conf")

            current_time = time.time()
            time_str = time.strftime("%H:%M %d:%m:%Y", time.localtime(current_time))
            is_need_add_to_history = True

            if last_update_value is None:
                last_update_value = {
                    "time": time_str,
                    "digits": new_digits,
                    "timestamp": current_time
                }

            # Фильтрация дубликатов
            if len(history) >= 2:
                last_item = list_to_number(history[-1]["digits"])
                current_number = list_to_number(new_digits)

                if current_number == last_item:
                    is_need_add_to_history = False
                else:
                    check_history_and_save_if_needed(new_digits, img)
                    last_update_value = {
                        "time": time_str,
                        "digits": new_digits,
                        "timestamp": current_time
                    }

            with history_lock:
                if is_need_add_to_history and new_digits:
                    last_recognized_digits = new_digits
                    history.append({
                        "time": time_str,
                        "digits": new_digits,
                        "timestamp": current_time
                    })
                    while len(history) > Config.MAX_HISTORY_SIZE:
                        history.pop(0)
                    print("✅ Обнаружено изменение; новые цифры:", new_digits)

                    save_test_image(img, result['full_number'], "next", Config.VALIDATION_DIR)

                    RedisManager.set_key(meter_watch_shared_config.REDIS_KEYS['gas_flow'], "1")
                    RedisManager.set_key(meter_watch_shared_config.REDIS_KEYS['gas_number'], str(result['full_number']))
                    RedisManager.set_key(meter_watch_shared_config.REDIS_KEYS['gas_last_activity'], time_str)
                    
                    last_nearly_activity_data = {"time": time_str, "digits": new_digits}
                    last_nearly_activity_counter = 0
                else:
                    print("⏺️ Изменений не обнаружено. Текущие цифры:", new_digits)
                    RedisManager.set_key(meter_watch_shared_config.REDIS_KEYS['gas_flow'], "0")
                    
                    # if -1 not in new_digits:                            
                    #     last_nearly_activity_counter += 1
                    #     if last_nearly_activity_counter > 5:
                    #         last_nearly_activity_data = None

        # except Exception as ex:
        #     print("❌ Ошибка в monitor_loop:", ex)
        #     consecutive_failures += 1
            
        time.sleep(Config.POLL_INTERVAL_SECONDS)

def start_monitoring():
    """Start the monitoring thread"""
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    return monitor_thread

def get_history():
    """Get recognition history"""
    with history_lock:
        return history.copy()

def get_last_activity():
    """Get last activity data"""
    if last_nearly_activity_data and history:        
        if len(history) >= 4:
            return history[-4:], last_update_value
        else:
            return history.copy(), last_update_value
    
    return [], last_update_value