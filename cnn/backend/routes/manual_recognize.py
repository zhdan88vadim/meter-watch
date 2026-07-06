import cv2
import numpy as np
from flask import Blueprint, request, jsonify
import base64
import io
import os
from PIL import Image
from datetime import datetime
from pathlib import Path
import uuid
import re
from services.recognition import recognize_image
from configuration import Config


manual_recognize_bp = Blueprint('manual_recognize', __name__)

@manual_recognize_bp.route('/recognize', methods=['POST'])
def recognize():
    """
    Ожидает JSON с base64 изображением
    Пример: {"image": "base64_string"}
    """
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode the image
        img_data = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_data))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        result, _ = recognize_image(img)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@manual_recognize_bp.route('/test', methods=['GET'])
def test():
    """
    GET метод для тестирования распознавания
    Принимает параметр filename - имя файла в датасете
    Пример: GET /test?filename=image.jpg&dataset=test
    """
    try:
        filename = request.args.get('filename')
        dataset = request.args.get('dataset')
        
        if not filename:
            return jsonify({
                'error': 'No filename provided',
                'usage': 'GET /test?filename=your_image.jpg&dataset=test'
            }), 400
        
        base_path = Config.OUTPUT_DIR
        dataset_path = os.path.join(base_path, dataset)
        file_path = os.path.join(dataset_path, filename)
        
        print(f"Looking for file: {file_path}")  # Для отладки
        
        # Проверяем существование файла
        if not os.path.exists(file_path):
            return jsonify({
                'error': f'File not found: {filename} in dataset {dataset}',
                'path': file_path
            }), 404
        
        # Читаем изображение
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Could not read image file'}), 400
        
        result, _ = recognize_image(img)
        result['filename'] = filename
            
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@manual_recognize_bp.route('/delete-file', methods=['POST'])
def delete_file():
    """
    POST method for removing a file from a dataset
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        filename = data.get('filename')
        dataset = data.get('dataset')
        
        if not filename:
            return jsonify({"error": "Filename is required"}), 400
        
        # Forming a full path to the file
        base_path = Config.OUTPUT_DIR
        dataset_path = os.path.join(base_path, dataset)
        file_path = os.path.join(dataset_path, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                "error": f"File not found: {filename}",
                "path": file_path
            }), 404
        
        os.remove(file_path)
        
        return jsonify({
            "success": True,
            "message": f"File deleted successfully: {filename}",
            "filename": filename,
            "dataset": dataset
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@manual_recognize_bp.route('/test/list', methods=['GET'])
def test_list():
    """
    GET method to get a list of available images in the current directory
    """
    try:
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        base_path = Config.OUTPUT_DIR
        dataset = request.args.get('dataset')
        dataset_path = os.path.join(base_path, dataset)

        files = []
        
        for f in os.listdir(dataset_path):
            full_path = os.path.join(dataset_path, f)
            if os.path.isfile(full_path) and any(f.lower().endswith(ext) for ext in valid_extensions):
                files.append(f)
        
        return jsonify({
            'files': files,
            'count': len(files)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@manual_recognize_bp.route('/save-digit', methods=['POST'])
def save_digit():
    """
    Saves a base64 image to a folder with the specified number.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        digit = data.get('digit')
        image_base64 = data.get('image_base64')
        filename = data.get('filename')
        
        if digit is None:
            return jsonify({"error": "Digit is required"}), 400
        
        try:
            digit = int(digit)
        except ValueError:
            return jsonify({"error": "Digit must be a number"}), 400
        
        if digit < 0 or digit > 9:
            return jsonify({"error": "Digit must be between 0 and 9"}), 400
        
        if not image_base64:
            return jsonify({"error": "Image data is required"}), 400
        
        digit_folder = Path(Config.TRAINING_DATA_DIR) / str(digit)
        digit_folder.mkdir(parents=True, exist_ok=True)
        
        if filename:
            if not filename.endswith('.png'):
                filename += '.png'
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{digit}_{timestamp}_{unique_id}_org.png"
        
        filepath = digit_folder / filename
        
        # Remove the data:image/png;base64 prefix, if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Remove whitespace
        image_base64 = re.sub(r'\s+', '', image_base64)
        
        # Decoding base64 into bytes
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 data: {str(e)}"}), 400
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
            print(f"File saved to {filepath}")

        return jsonify({
            "success": True,
            "message": "Image saved successfully",
            "digit": digit,
            "filename": filename,
            "filepath": str(filepath),
            "folder": str(digit_folder)
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500
