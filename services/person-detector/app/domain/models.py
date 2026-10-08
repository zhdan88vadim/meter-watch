from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

Frame = np.ndarray


@dataclass(frozen=True)
class Detection:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass(frozen=True)
class Thresholds:
    person_is_active_threshold: int
    person_absence_threshold: int
    startup_person_timeout: int
    alert_cooldown: int
