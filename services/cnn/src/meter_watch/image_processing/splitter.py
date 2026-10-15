"""Split a binarized meter image into individual digit ROIs."""
from __future__ import annotations

import cv2
import numpy as np


def split_digits(
    thresh: np.ndarray,
    original: np.ndarray,
    *,
    min_height: int = 5,
    min_width: int = 5,
    min_area: int = 50,
    max_aspect_ratio: float = 10.0,
    y_gap_threshold: int = 15,
    final_min_area: int = 300,
    padding_crop: int = 5,
) -> list[np.ndarray]:
    """Split a binarized image into digit ROIs, sorted left-to-right.

    Returns an empty list if no digits are found.
    """
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered = _filter_contours(
        contours, min_height, min_width, min_area, max_aspect_ratio
    )
    if not filtered:
        return []

    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, filtered, -1, 255, -1)

    merged = _merge_nearby_digits(mask, y_gap_threshold)

    new_contours, _ = cv2.findContours(
        merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    new_contours = [c for c in new_contours if cv2.contourArea(c) > final_min_area]
    new_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

    height, width = thresh.shape[:2]
    digits: list[np.ndarray] = []

    for cnt in new_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(0, x - padding_crop)
        y1 = max(0, y - padding_crop)
        x2 = min(width, x + w + padding_crop)
        y2 = min(height, y + h + padding_crop)

        digit = thresh[y1:y2, x1:x2]
        if digit.size == 0:
            digit = np.zeros((h + 2 * padding_crop, w + 2 * padding_crop), np.uint8)
        digits.append(digit)

    return digits


# ---------- internals ----------


def _filter_contours(
    contours: list,
    min_height: int,
    min_width: int,
    min_area: int,
    max_aspect_ratio: float,
) -> list:
    filtered = []
    for c in contours:
        _, _, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect = w / h if h > 0 else 0
        if (
            h > min_height
            and w > min_width
            and area > min_area
            and aspect < max_aspect_ratio
        ):
            filtered.append(c)
    return filtered


def _merge_nearby_digits(mask: np.ndarray, y_gap_threshold: int) -> np.ndarray:
    """Morphologically close gaps between vertically adjacent digit parts."""
    padding = 70
    padded = cv2.copyMakeBorder(
        mask.copy(), padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=0,
    )
    kernel_height = y_gap_threshold * 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kernel_height))
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    return closed[padding:-padding, padding:-padding]
