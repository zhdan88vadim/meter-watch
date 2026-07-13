import os
import uuid
import shutil
from pathlib import Path
import cv2
from typing import Dict, List, Any
import json
import numpy as np

from models.pytorch_model import load_pytorch_model
from services.recognition import recognize_image

def save_rename_history(history: List[Dict], log_file: str, log_format: str = "json"):
    """Сохраняет историю переименований в файл"""
    
    if log_format in ["json", "both"]:
        json_file = log_file if log_file.endswith('.json') else f"{log_file}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON лог сохранен: {json_file}")

def denormalize_image(normalized):
    """
    Преобразует нормализованное изображение обратно в обычный формат для отображения
    
    Args:
        normalized: нормализованное изображение (значения от -1 до 1)
    
    Returns:
        изображение в формате uint8 (0-255)
    """
    # Денормализация: (x * 0.5) + 0.5
    denormalized = (normalized * 0.5) + 0.5
    
    # Обрезаем значения вне диапазона [0, 1]
    denormalized = np.clip(denormalized, 0, 1)
    
    # Конвертируем в uint8 (0-255)
    denormalized = (denormalized * 255).astype(np.uint8)
    
    return denormalized

def process_and_rename_images(input_dir, output_dir, recognize_image_func):
    """
    Обрабатывает все изображения из директории, распознает их и переименовывает
    
    Args:
        input_dir: путь к директории с исходными изображениями
        output_dir: путь к директории для сохранения обработанных изображений
        recognize_image_func: функция распознавания, возвращающая (result, min_conf)
    """
    
    # Создаем выходную директорию, если её не существует
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Поддерживаемые форматы изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    # Получаем список всех файлов изображений
    image_files = [f for f in os.listdir(input_dir) 
                   if Path(f).suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"В директории {input_dir} не найдено изображений")
        return
    
    print(f"Найдено {len(image_files)} изображений")
    
    processed_count = 0
    failed_count = 0
    rename_history = []
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        
        try:
            
            image = cv2.imread(img_path)

            result, min_conf = recognize_image(image)


            for digit in result['digits']:

                # if digit['confidence'] > 0.5:
                unique_id = str(uuid.uuid4())[:8]
                dir = f"../../dataset_val_test_raw/{digit['prediction']}/"
                filepath = os.path.join(dir, f"{digit['prediction']}__{digit['confidence']:.2f}_{unique_id}.jpg")
                filepath_original = os.path.join(dir, f"orgnl__{digit['prediction']}__{digit['confidence']:.2f}_{unique_id}.jpg")
                filepath_model = os.path.join(dir, f"model__{digit['prediction']}__{digit['confidence']:.2f}_{unique_id}.jpg")
                print(filepath)
                Path(dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(filepath, digit['raw_image'])
                cv2.imwrite(filepath_original, digit['source_image'])
                # cv2.imwrite(filepath_model, denormalize_image(digit['prepared_model_image']))
        
            # Получаем распознанное число
            new_digits = list(result['full_number'])
            recognized_number = result['full_number']
            
            # Если число не распознано или пустое
            if not recognized_number:
                print(f"⚠️  Не удалось распознать число на {img_file}, пропускаем")
                failed_count += 1
                continue
            
            # Генерируем уникальный ID (первые 8 символов для краткости)
            unique_id = str(uuid.uuid4())[:8]
            
            # Формируем новое имя файла
            original_extension = Path(img_file).suffix
            new_filename = f"{recognized_number}_{unique_id}{original_extension}"
            new_path = os.path.join(output_dir, new_filename)
            
            # Копируем и переименовываем файл
            shutil.copy2(img_path, new_path)
            
            print(f"✓ {img_file} -> {new_filename} (уверенность: {min_conf:.2f})")
            processed_count += 1

            # Сохраняем информацию о переименовании
            history_entry = {
                'old_name': img_file,
                'new_name': new_filename,
                'recognized_number': recognized_number,
                'confidence': min_conf,
            }
            rename_history.append(history_entry)
            
        except Exception as e:
            print(f"✗ Ошибка при обработке {img_file}: {str(e)}")
            failed_count += 1
    

    save_rename_history(rename_history, "rename_history.json", "json")

    # Выводим статистику
    print("\n" + "="*50)
    print(f"Обработка завершена!")
    print(f"✅ Успешно обработано: {processed_count}")
    print(f"❌ Ошибок/пропущено: {failed_count}")
    print(f"📁 Результаты сохранены в: {output_dir}")
    print("="*50)


# Пример использования с псевдо-функцией распознавания
# (замените на вашу реальную функцию)

def example_recognize_image(img_path):
    """
    Пример функции распознавания.
    Замените на вашу реальную реализацию.
    """
    # Ваша реальная логика распознавания здесь
    # Должна возвращать (result, min_conf)
    
    # Пример для тестирования:
    class Result:
        pass
    
    result = Result()
    # Предполагаем, что результат содержит поле 'full_number'
    result['full_number'] = "12345"  # пример распознанного числа
    
    min_conf = 0.95  # пример уверенности
    
    return result, min_conf


if __name__ == "__main__":
    load_pytorch_model()
    
    # INPUT_DIRECTORY = "/media/vadim/1TB_SSD/my_github/meter-watch/hard"
    INPUT_DIRECTORY = "/media/vadim/1TB_SSD/my_github/meter-watch/raw_images"
    OUTPUT_DIRECTORY = "/media/vadim/1TB_SSD/my_github/meter-watch/hard_images_output"
    
    path = "/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_test_raw/"

    # Get only directories
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for dir_name in dirs:
        dir_path = os.path.join(path, dir_name)
        shutil.rmtree(dir_path)
        print(f"Removed directory: {dir_name}")

    # Замените example_recognize_image на вашу реальную функцию
    process_and_rename_images(INPUT_DIRECTORY, OUTPUT_DIRECTORY, example_recognize_image)