
import os
import cv2
import time
from configuration import Config

def save_test_image(img, digits, comment: str, dir: str = Config.WRONG_PREDICTIONS_DIR):
    """Save test image with comment"""
    filename = os.path.join(dir, f"{comment}__{digits}__{int(time.time())}.png")
    cv2.imwrite(filename, img)
    print(f"image saved to: {filename}")
    return filename
