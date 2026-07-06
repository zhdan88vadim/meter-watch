
import cv2
import os
import numpy as np
from datetime import datetime
import base64

def image_to_base64(img):
    """Конвертирует numpy массив в base64 строку"""
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def compare_images_simple(img1, img2, threshold=0.05, save_path=None):
    """
    Простое сравнение двух изображений
    """
    if img1 is None or img2 is None:
        return True, 1.0
    
    if img1.shape != img2.shape:
        return True, 1.0
    
    # Вычисляем разницу
    diff = cv2.absdiff(img1, img2)
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    normalized_diff = mse / (255 ** 2)
    
    # Если разница больше порога
    if normalized_diff > threshold:
        if save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{save_path}/changed_{timestamp}.png", img2)
        return True, normalized_diff
    
    return False, normalized_diff
