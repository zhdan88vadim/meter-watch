import time
import threading
from typing import List, Optional, Tuple, Any

from services.database import save_meter_data_to_database
from models.error_models import ImageFetchError, RecognitionError
from configuration import Config
from utils.api_utils import fetch_image, timestamp_ms
from utils.log_data import save_test_image
from services.config import ConfigKeys, config
from models.monitoring_models import MeterState, RecognitionResult


class MeterMonitor:
    """Монитор состояния счетчика"""
    
    def __init__(self):
        self.history: List[MeterState] = []
        self.anomaly_history: List[MeterState] = []
        self.last_state: Optional[MeterState] = None
        self.last_update_value: Optional[MeterState] = None
        self.last_image: Optional[Any] = None
        self.last_nearly_activity_data: Optional[MeterState] = None
        self.last_nearly_activity_counter: int = 0
        
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def _should_process(self, result: RecognitionResult) -> bool:
        """Проверить, нужно ли обрабатывать результат"""
        if not result.digits:
            return False
            
        # Проверка на дубликат
        if self.history and self.history[-1].number == result.number:
            return False
            
        return True
    
    def _handle_low_confidence(self, result: RecognitionResult) -> bool:
        """Обработка низкой уверенности"""
        if result.min_conf < config.get(ConfigKeys.SAVE_THRESHOLD):
            save_test_image(
                result.image, 
                result.digits, 
                "low_conf"
            )
            return True
        return False
    
    def _handle_big_difference(self, result: RecognitionResult) -> bool:
        """Обработка большого скачка показаний"""
        with self._lock:
            if not self.last_state:
                return False
                
            difference = abs(result.number - self.last_state.number)
            if difference > 10:
                save_test_image(result.image, result.number, f"big_diff_{difference}")
                if self.last_image is not None:
                    save_test_image(self.last_image, result.number, f"big_diff_{difference}")
                return True
            return False
    
    def _handle_decrease(self, result: RecognitionResult) -> bool:
        """Обработка уменьшения показаний"""
        with self._lock:
            if self.last_state and result.number < self.last_state.number:
                save_test_image(result.image, result.number, "less")
                if self.last_image is not None:
                    save_test_image(self.last_image, result.number, "less")
            return True
        return False                    
        
    def _add_to_history(self, result: RecognitionResult) -> None:
        """Добавить результат в историю"""
        state = MeterState(
            digits=result.digits,
            timestamp=result.timestamp,
            time_str=result.time_str
        )
        
        with self._lock:
            self.history.append(state)
            self.last_state = state
            self.last_update_value = state
            self.last_image = result.image
            
            # Ограничение размера истории
            if len(self.history) > Config.MAX_HISTORY_SIZE:
                self.history = self.history[-Config.MAX_HISTORY_SIZE:]
            
            # Обновление активности
            self.last_nearly_activity_data = state
            self.last_nearly_activity_counter = 0
    
        
    def _add_to_anomaly_history(self, result: RecognitionResult) -> None:
        """Добавить результат в историю"""
        state = MeterState(
            digits=result.digits,
            timestamp=result.timestamp,
            time_str=result.time_str
        )
        
        with self._lock:
            self.anomaly_history.append(state)

            # Ограничение размера истории
            if len(self.anomaly_history) > Config.MAX_ANOMALY_HISTORY_SIZE:
                self.anomaly_history = self.anomaly_history[-Config.MAX_ANOMALY_HISTORY_SIZE:]

    def _check_anomaly_sequence_validity(self) -> bool:
        """
        Проверить, идут ли числа в anomaly_history по порядку (+1)
        Returns: True если все числа последовательные (1,2,3...)
        """
        with self._lock:
            if len(self.anomaly_history) < 2:
                return False
            
            # Проверяем, что каждое следующее число = предыдущее + 1
            for i in range(1, len(self.anomaly_history)):
                # Сравниваем числа (number) а не объекты MeterState
                if self.anomaly_history[i].number - self.anomaly_history[i-1].number != 1:
                    return False
            
            return True

    def _update_redis(self, result: RecognitionResult) -> None:
        """Обновить Redis (закомментировано, но оставлено для ясности)"""
        # RedisManager.set_key(
        #     meter_watch_shared_config.REDIS_KEYS['gas_flow'], 
        #     "1"
        # )
        # RedisManager.set_key(
        #     meter_watch_shared_config.REDIS_KEYS['gas_number'], 
        #     str(result.digits)
        # )
        # RedisManager.set_key(
        #     meter_watch_shared_config.REDIS_KEYS['gas_last_activity'], 
        #     result.time_str
        # )
        pass
    
    def _handle_no_change(self, result: RecognitionResult) -> None:
        """Обработка отсутствия изменений"""
        print("⏺️ Изменений не обнаружено. Текущие цифры:", result.digits)
        # RedisManager.set_key(
        #     meter_watch_shared_config.REDIS_KEYS['gas_flow'], 
        #     "0"
        # )
        
        # Обновление счетчика бездействия
        # if result.is_valid:
        #     self.last_nearly_activity_counter += 1
        #     if self.last_nearly_activity_counter > 5:
        #         self.last_nearly_activity_data = None
    
    def process_result(self, result: RecognitionResult) -> None:
        """Обработка результата распознавания"""
        with self._lock:
            is_low_confidence = self._handle_low_confidence(result)

            if not self._should_process(result):
                self._handle_no_change(result)
                return

            is_big_difference = self._handle_big_difference(result)
            is_decrease = self._handle_decrease(result)
            is_anomaly = is_low_confidence or is_big_difference or is_decrease

            is_anomaly_correct = self._check_anomaly_sequence_validity()

            if is_anomaly and not is_anomaly_correct:
                self._add_to_anomaly_history(result)
            else:
                self._add_to_history(result)
                self._update_redis(result)

                save_test_image(
                    result.image, 
                    result.digits, 
                    "next", 
                    Config.VALIDATION_DIR
                )            
            
            save_meter_data_to_database(result, is_anomaly)

            print(f"✅ Обнаружено изменение; новые цифры: {result.digits}, is_anomaly: {is_anomaly}")
    
    def run_cycle(self) -> None:
        """Один цикл мониторинга"""
        try:
            print("📷 Запрос изображения с камеры...")
            
            # Получение изображения
            image = fetch_image(
                config.get(ConfigKeys.CAMERA_URL) + str(timestamp_ms())
            )
            
            if image is None:
                print("❌ Не удалось получить изображение")
                raise ImageFetchError("Не удалось получить изображение")
            
            # Распознавание
            result = RecognitionResult.from_image(image)
            if result is None:
                raise RecognitionError("Не удалось распознать изображение")
            
            # Обработка результата
            self.process_result(result)
            
        except (ImageFetchError, RecognitionError) as e:
            raise
        except Exception as e:
            raise
    
    def run_forever(self) -> None:
        """Бесконечный цикл мониторинга"""
        
        consecutive_failures = 0
        max_failures = 10
        
        while self._running:
            try:
                self.run_cycle()
                consecutive_failures = 0  # Сброс при успехе
            except (ImageFetchError, RecognitionError) as e:
                consecutive_failures += 1
                print(f"❌ Временная ошибка ({consecutive_failures}/{max_failures}): {e}")
                
                if consecutive_failures >= max_failures:
                    print("⚠️ Критическое количество ошибок, перезапуск...")
                    consecutive_failures = 0
            except Exception as e:
                print(f"❌ Критическая ошибка: {e}")
                consecutive_failures += 1
        
            time.sleep(Config.POLL_INTERVAL_SECONDS)
    
    def start(self) -> threading.Thread:
        """Запуск мониторинга в отдельном потоке"""
        self._running = True
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()
        return self._thread
    
    def stop(self) -> None:
        """Остановка мониторинга"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
    
    def get_history(self, limit: Optional[int] = None) -> List[MeterState]:
        """Получить историю"""
        with self._lock:
            if limit:
                return self.history[-limit:]
            return self.history.copy()
    
    def get_last_activity(self) -> Tuple[List[MeterState], Optional[MeterState]]:
        """Получить последнюю активность"""
        with self._lock:
            if self.last_nearly_activity_data and self.history:
                # Возвращаем последние 4 записи или всю историю
                recent = self.history[-4:] if len(self.history) >= 4 else self.history.copy()
                return recent, self.last_update_value
            
            return [], self.last_update_value
    
    @property
    def current_state(self) -> Optional[MeterState]:
        """Текущее состояние"""
        with self._lock:
            return self.history[-1] if self.history else None


monitor = MeterMonitor()

def start_monitoring():
    """Start the monitoring thread"""
    return monitor.start()

def get_history():
    """Get recognition history"""
    return monitor.get_history()

def get_last_activity():
    """Get last activity data"""
    return monitor.get_last_activity()