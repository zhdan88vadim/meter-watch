
import os
import cv2
import time
from configuration import Config

def save_test_image(img, digits, comment: str):
    """Save test image with comment"""
    filename = os.path.join(Config.WRONG_PREDICTIONS_DIR, f"{comment}__{digits}__{int(time.time())}.png")
    cv2.imwrite(filename, img)
    print(f"image saved to: {filename}")
    return filename
