"""Image preprocessing for digit recognition."""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_PARAMS = {
    "blur_ksize": 7,
    "blur_sigma": 5,
    "adaptive_block_size": 57,
    "adaptive_c": 5,
    "morph_kernel": 2,
    "morph_iter": 1,
}


def preprocess_image(image: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Convert an image into a binarized, inverted, cleaned-up mask."""
    params = params or DEFAULT_PARAMS

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(
        gray,
        (params["blur_ksize"], params["blur_ksize"]),
        params["blur_sigma"],
    )
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        params["adaptive_block_size"],
        params["adaptive_c"],
    )
    kernel = np.ones(
        (params["morph_kernel"], params["morph_kernel"]), dtype="uint8"
    )
    opened = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=params["morph_iter"],
    )
    return opened


def prepare_for_model(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center-pad a ROI to a square, resize to 28x28, normalize to [-1, 1]."""
    h, w = roi.shape[:2]
    size = max(h, w)

    square = np.zeros((size, size), dtype=roi.dtype)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = roi

    resized = cv2.resize(square, (28, 28))
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    return normalized, resized
