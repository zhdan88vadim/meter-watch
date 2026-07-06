import cv2
import numpy as np
from datetime import datetime
import os

class ChangeDetector:
    def __init__(self, change_percent=4, threshold=0.05):
        self.threshold = threshold  # порог изменения (0-1)
        self.change_percent_level = change_percent  # порог изменения (0-1)
        self.prev_image = None
        self.frame_count = 0
    
    def check_and_save(self, current, save_path):
        """
        Проверяет изменения и сохраняет если есть
        Возвращает: (changed, diff_image)
        """
        changed = False
        diff_image = None
        
        if self.prev_image is None:
            # Первое изображение - сохраняем
            changed = True
            reason = "first_frame"
        else:
            # Проверяем размеры
            if current.shape != self.prev_image.shape:
                changed = True
                reason = "size_changed"
            else:
                # Вычисляем разницу
                diff = cv2.absdiff(current, self.prev_image)

                print('diff', diff)
                
                # Считаем процент измененных пикселей
                change_mask = diff > 30
                change_percent = (np.sum(change_mask) / change_mask.size) * 100
                
                # MSE разница
                mse = np.mean((current.astype(float) - self.prev_image.astype(float)) ** 2)
                normalized_diff = mse / (255 ** 2)
                
                # Если изменения превышают порог
                if normalized_diff > self.threshold or change_percent > self.change_percent_level:
                    changed = True
                    reason = f"change_{change_percent:.1f}%"
                    
                    # Создаем изображение с рамкой вокруг изменений
                    diff_image = self.draw_changes(current, self.prev_image, diff)
        
        if changed:
            # Сохраняем изображение
            # self.save_image(current, save_path, reason)
            # Обновляем предыдущее изображение
            self.prev_image = current.copy()
            self.frame_count += 1
        
        return changed, diff_image
    
    def draw_changes(self, current, prev, diff):
        """
        Рисует рамки вокруг измененных областей
        """
        # Копируем текущее изображение
        result = current.copy()
        
        # Находим контуры изменений
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        # Находим контуры
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Рисуем рамки вокруг изменений
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Игнорируем маленькие изменения
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(result, f"change", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Добавляем общую информацию
        cv2.putText(result, f"Changes detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
    
    def save_image(self, image, save_path, reason):
        """
        Сохраняет изображение
        """
        # Создаем папку
        os.makedirs(save_path, exist_ok=True)
        
        # Генерируем имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{self.frame_count:04d}_{timestamp}_{reason}.png"
        full_path = os.path.join(save_path, filename)
        
        # Сохраняем
        cv2.imwrite(full_path, image)
        print(f"Saved: {full_path}")
        
        return full_path

