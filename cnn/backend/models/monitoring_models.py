import time
from typing import List, Optional, Any
from dataclasses import dataclass

from utils.number_utils import list_to_number
from services.recognition import recognize_image


@dataclass
class MeterState:
    """Состояние счетчика в один момент времени"""
    digits: List[int]
    timestamp: float
    time_str: str
    
    @property
    def number(self) -> int:
        return list_to_number(self.digits)
    
    @property
    def is_valid(self) -> bool:
        return -1 not in self.digits


@dataclass
class RecognitionResult:
    """Результат распознавания"""
    digits: List[int]
    number: int
    min_conf: float
    image: Any
    timestamp: float
    time_str: str
    
    @classmethod
    def from_image(cls, image: Any) -> Optional['RecognitionResult']:
        """Создать результат из изображения"""
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
            print(f"❌ Ошибка распознавания: {e}")
            return None

