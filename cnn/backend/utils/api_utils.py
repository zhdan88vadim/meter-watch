import requests
import cv2
import numpy as np
import time
from configuration import Config

def get_api(url, timeout=5):
    """Выполняет GET-запрос к API (например, для включения LED)"""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            return response.status_code
        else:
            print("Ошибка API, статус:", response.status_code)
            return None
    except Exception as e:
        print("Ошибка в get_api:", e)
        return None

def fetch_image(url, timeout=5, save_copy=True):
    """Скачивает изображение по URL и возвращает его как изображение OpenCV"""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            data = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            if save_copy:
                Config.create_directories()
                cv2.imwrite(f'{Config.OUTPUT_DIR}/camera.png', image)
            
            return image
        else:
            print("Ошибка загрузки изображения, статус:", response.status_code)
            return None
    except Exception as e:
        print("Ошибка в fetch_image:", e)
        return None

def timestamp_ms():
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)