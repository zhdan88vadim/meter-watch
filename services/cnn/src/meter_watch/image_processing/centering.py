"""Center a digit ROI within a fixed-size canvas."""
from __future__ import annotations

import cv2
import numpy as np


def center_digit(image: np.ndarray, padding: int = 10) -> np.ndarray:
    """Crop tightly around the digit and re-center it in a blank canvas."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image

    x_min, y_min = float("inf"), float("inf")
    x_max, y_max = 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(image.shape[1], int(x_max) + padding)
    y_max = min(image.shape[0], int(y_max) + padding)

    digits_area = image[y_min:y_max, x_min:x_max]
    h, w = digits_area.shape[:2]

    centered = np.ones_like(image)
    start_y = (image.shape[0] - h) // 2
    start_x = (image.shape[1] - w) // 2
    centered[start_y : start_y + h, start_x : start_x + w] = digits_area
    return centered
