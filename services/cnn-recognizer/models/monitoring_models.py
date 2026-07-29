import time
from typing import List, Optional, Any
from dataclasses import dataclass

from utils.number_utils import list_to_number
from services.recognition import recognize_image


@dataclass
class MeterState:
    """Meter state at a specific point in time"""
    digits: List[int]
    timestamp: float
    time_str: str
    
    @property
    def number(self) -> int:
        return list_to_number(self.digits)
    
    @property
    def is_valid(self) -> bool:
        return -1 not in self.digits
    
    def __repr__(self):
        return f"MeterState(number={self.number}, digits={self.digits}, time={self.time_str})"
    
    def __str__(self):
        return f"Reading: {self.number} ({self.digits}) at {self.time_str}"    


@dataclass
class RecognitionResult:
    """Recognition result"""
    digits: List[int]
    number: int
    min_conf: float
    image: Any
    timestamp: float
    time_str: str
    
    @classmethod
    def from_image(cls, image: Any) -> Optional['RecognitionResult']:
        """Create result from image"""
        try:
            result, min_conf = recognize_image(image)
            digits = list(result['full_number'])
            
            return cls(
                digits=digits,
                number=list_to_number(digits),
                min_conf=min_conf,
                image=image,
                timestamp=time.time(),
                time_str=time.strftime("%H:%M %d:%m:%Y", time.localtime())
            )
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return None