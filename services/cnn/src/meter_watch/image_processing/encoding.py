"""Base64 encoding helpers for images."""
from __future__ import annotations

import base64

import cv2
import numpy as np


def image_to_base64(img: np.ndarray, ext: str = ".png") -> str:
    """Encode a numpy image as a base64 string."""
    ok, buffer = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")
