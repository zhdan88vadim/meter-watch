import time
from collections import deque

class FPS:
    def __init__(self, avg_window=30):
        self.start_time = None
        self.frame_count = 0
        self.fps = 0
        self.avg_window = avg_window
        self.frame_times = deque(maxlen=avg_window)
    
    def update(self):
        """Call this method every frame to update FPS"""
        current_time = time.time()
        self.frame_count += 1
        self.frame_times.append(current_time)
        
        if self.start_time is None:
            self.start_time = current_time
            return 0
        
        # Calculate FPS based on time window
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = (len(self.frame_times) - 1) / time_diff
        
        return self.fps
    
    def get_fps(self):
        """Return current FPS"""
        return self.fps
    
    def reset(self):
        """Reset FPS counter"""
        self.start_time = None
        self.frame_count = 0
        self.fps = 0
        self.frame_times.clear()