
import cv2
import base64

def image_to_base64(img):
    """Конвертирует numpy массив в base64 строку"""
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64