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
    """Meter state monitor"""

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
        """Check if the result should be processed"""
        if not result.digits:
            return False
            
        # Check for duplicates
        return not (self.history and self.history[-1].number == result.number)
    
    def _handle_low_confidence(self, result: RecognitionResult) -> bool:
        """Handle low confidence results"""
        if result.min_conf < config.get(ConfigKeys.SAVE_THRESHOLD):
            save_test_image(
                result.image, 
                result.digits, 
                "low_conf"
            )
            return True
        return False
    
    def _handle_big_difference(self, result: RecognitionResult) -> bool:
        """Handle large jumps in readings"""
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
        """Handle decrease in readings"""
        if self.last_state and result.number < self.last_state.number:
            save_test_image(result.image, result.number, "less")
            if self.last_image is not None:
                save_test_image(self.last_image, result.number, "less")
            return True
        return False                    
        
    def _add_to_history(self, result: RecognitionResult) -> None:
        """Add result to history"""
        state = MeterState(
            digits=result.digits,
            timestamp=result.timestamp,
            time_str=result.time_str
        )

        self.history.append(state)
        self.last_state = state
        self.last_update_value = state
        self.last_image = result.image
        
        # Limit history size
        if len(self.history) > Config.MAX_HISTORY_SIZE:
            self.history = self.history[-Config.MAX_HISTORY_SIZE:]
        
        # Update activity
        self.last_nearly_activity_data = state
        self.last_nearly_activity_counter = 0
    
        
    def _add_to_anomaly_history(self, result: RecognitionResult) -> None:
        """Add result to anomaly history"""
        state = MeterState(
            digits=result.digits,
            timestamp=result.timestamp,
            time_str=result.time_str
        )

        print(f"🔍 [{self.id}] Adding anomaly: {result.number}")
    
        # Check if we can add to the current sequence
        if self.anomaly_history:
            last = self.anomaly_history[-1]
            diff = result.number - last.number
            
            # If difference > 4 or decrease - start a new sequence
            if diff < 0 or diff > 4:
                print(f"🔄 New sequence: {last.number} -> {result.number} (diff={diff})")
                self.anomaly_history = []
                            
        self.anomaly_history.append(state)

        # Limit history size
        if len(self.anomaly_history) > Config.MAX_ANOMALY_HISTORY_SIZE:
            self.anomaly_history = self.anomaly_history[-Config.MAX_ANOMALY_HISTORY_SIZE:]

    def _check_anomaly_sequence_validity(self) -> bool:
        """
        Check if anomalies are consecutive readings
        """
        if len(self.anomaly_history) <= 2:
            return False
        
        # Check that each next number is greater than the previous by no more than 3
        for i in range(1, len(self.anomaly_history)):
            diff = self.anomaly_history[i].number - self.anomaly_history[i-1].number
            
            # Should increase (diff > 0) and not more than 4
            if diff < 0 or diff > 4:
                print(f"❌ Invalid sequence: {self.anomaly_history[i-1].number} -> {self.anomaly_history[i].number} (diff={diff})")
                return False
        
        print(f"✅ Valid sequence of {len(self.anomaly_history)} anomalies")
        return True

    def _update_redis(self, result: RecognitionResult) -> None:
        """Update Redis"""
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
        """Handle no change detection"""
        print("⏺️ No change detected. Current digits:", result.digits)
        # RedisManager.set_key(
        #     meter_watch_shared_config.REDIS_KEYS['gas_flow'], 
        #     "0"
        # )
        
        # Update inactivity counter
        # if result.is_valid:
        #     self.last_nearly_activity_counter += 1
        #     if self.last_nearly_activity_counter > 5:
        #         self.last_nearly_activity_data = None
        
    def process_result(self, result: RecognitionResult) -> None:
        """Process recognition result"""
        with self._lock:
            if not self._should_process(result):
                self._handle_no_change(result)
                return

            is_low_confidence = self._handle_low_confidence(result)
            is_big_difference = self._handle_big_difference(result)
            is_decrease = self._handle_decrease(result)
            is_anomaly = is_low_confidence or is_big_difference or is_decrease

            # Always add to anomaly history if it's an anomaly
            if is_anomaly:
                self._add_to_anomaly_history(result)
                print(f"⚠️ Anomaly #{len(self.anomaly_history)}: {result.number}")

                print([state.digits for state in self.anomaly_history])
                
                # Check the sequence
                if self._check_anomaly_sequence_validity():
                    # Confirm all anomalies
                    print(f"✅ Sequence of {len(self.anomaly_history)} anomalies confirmed!")
                    
                    # Add to main history
                    for anomaly_state in self.anomaly_history:
                        self.history.append(anomaly_state)
                        self.last_state = anomaly_state

                    print("clean 1")
                    # Clear anomaly history
                    self.anomaly_history = []
                    
                    # Save as NOT an anomaly
                    save_meter_data_to_database(result, is_anomaly=False)
                    print(f"✅ Saved as real reading: {result.number}")
                    return
                return               
            
            # If this is NOT an anomaly
            if not is_anomaly:
                print("clean 2")
                self.anomaly_history = []
                
                # Add normal reading
                self._add_to_history(result)
                self._update_redis(result)
                save_meter_data_to_database(result, is_anomaly=False)
                print(f"✅ Normal reading: {result.number}")

                
    def run_cycle(self) -> None:
        """One monitoring cycle"""
        try:
            print("📷 Requesting image from camera...")
            
            image = fetch_image(
                config.get(ConfigKeys.CAMERA_URL) + str(time.time() * 1000)
            )
            
            if image is None:
                print("❌ Failed to get image", config.get(ConfigKeys.CAMERA_URL))
                raise ImageFetchError("Failed to get image")
            
            # Recognition
            result = RecognitionResult.from_image(image)
            if result is None:
                raise RecognitionError("Failed to recognize image")
            
            self.process_result(result)
            
        except (ImageFetchError, RecognitionError) as e:
            raise
        except Exception as e:
            raise
    
    def run_forever(self) -> None:
        """Infinite monitoring loop"""
        
        consecutive_failures = 0
        max_failures = 10
        
        while self._running:
            try:
                self.run_cycle()
                consecutive_failures = 0  # Reset on success
            except (ImageFetchError, RecognitionError) as e:
                consecutive_failures += 1
                print(f"❌ Temporary error ({consecutive_failures}/{max_failures}): {e}")
                
                if consecutive_failures >= max_failures:
                    print("⚠️ Critical number of errors, restarting...")
                    consecutive_failures = 0
            except Exception as e:
                print(f"❌ Critical error: {e}")
                consecutive_failures += 1
        
            time.sleep(Config.POLL_INTERVAL_SECONDS)
    
    def start(self) -> threading.Thread:
        """Start monitoring in a separate thread"""
        self._running = True
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()
        return self._thread
    
    def stop(self) -> None:
        """Stop monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
    
    def get_history(self, limit: Optional[int] = None) -> List[MeterState]:
        """Get history"""
        with self._lock:
            if limit:
                return self.history[-limit:]
            return self.history.copy()
    
    def get_last_activity(self) -> Tuple[List[MeterState], Optional[MeterState]]:
        """Get last activity"""
        with self._lock:
            if self.last_nearly_activity_data and self.history:
                # Return last 4 records or entire history
                recent = self.history[-4:] if len(self.history) >= 4 else self.history.copy()
                return recent, self.last_update_value
            
            return [], self.last_update_value
    
    @property
    def current_state(self) -> Optional[MeterState]:
        """Current state"""
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