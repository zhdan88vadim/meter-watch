import cv2
import numpy as np
from models.pytorch_model import predict_digit
from utils.image_utils import image_to_base64
from utils.preprocessing import preprocess_image
from utils.splitter import split_number, center_digits

def recognize_image(image):
    min_conf = 1
    thresh = preprocess_image(image)

    digits, original_digits = split_number(thresh, image)

    result = {
        'digits': [],
        'full_number': ''
    }
    
    for i, digit in enumerate(digits):
        centered = center_digits(digit)

        pred, conf, cam, saliency, resized_square, prepared = predict_digit(centered)
        if conf < min_conf:
            min_conf = conf
        
        # Converting heatmaps to base64
        cam_vis = (cam * 255).astype(np.uint8)
        saliency_vis = (saliency * 255).astype(np.uint8)
        
        result['digits'].append({
            'position': i + 1,
            'prediction': pred,
            'confidence': float(conf),
            'heatmap_gradcam': image_to_base64(cv2.applyColorMap(cam_vis, cv2.COLORMAP_JET)),
            'heatmap_saliency': image_to_base64(cv2.applyColorMap(saliency_vis, cv2.COLORMAP_HOT)),
            'digit_image': image_to_base64(centered),
            'raw_image': resized_square,
            'prepared_model_image': prepared,
            'source_image': original_digits[i],
        })
        result['full_number'] += str(pred)
    
    return result, min_conf