import cv2
import numpy as np


def split_number(thresh):
    # 3. Поиск контуров и фильтрация
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect_ratio = w / h if h > 0 else 0
        
        if h > 5 and w > 5 and area > 50 and aspect_ratio < 10:
            filtered_contours.append(c)
    
    if not filtered_contours:
        return None
    
    # 4. Создание маски
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, filtered_contours, -1, 255, -1)
    
    # 5. Добавление рамки и морфологическое закрытие для объединения цифр
    # для того чтобы при morphologyEx ядро не достало до края картинки и не расширило картинку белой рамкой, 
    # которая потом сольется с цифрами

    padding = 70
    mask_padded = cv2.copyMakeBorder(
        mask.copy(),
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=0
    )
    
    y_gap_threshold = 15
    kernel_height = y_gap_threshold * 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kernel_height))
    closed = cv2.morphologyEx(mask_padded, cv2.MORPH_CLOSE, kernel)
    
    # Удаление рамки
    closed = closed[padding:-padding, padding:-padding]

    # 6. Поиск объединенных контуров
    new_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_new_contours = []
    for c in new_contours:
        area = cv2.contourArea(c)
        if area > 300:
            filtered_new_contours.append(c)

    if not filtered_new_contours:
        return None

    # 8. Сортировка контуров слева направо (по X координате)
    filtered_new_contours = sorted(filtered_new_contours, 
                                    key=lambda c: cv2.boundingRect(c)[0])

    # 9. Вырезаем каждый контур как отдельное изображение
    digits = []
    height, width = thresh.shape[:2]

    for i, cnt in enumerate(filtered_new_contours):
        # Получаем bounding box контура
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Добавляем небольшой отступ вокруг цифры (опционально)
        padding_crop = 5
        x1 = max(0, x - padding_crop)
        y1 = max(0, y - padding_crop)
        x2 = min(width, x + w + padding_crop)
        y2 = min(height, y + h + padding_crop)
        
        # Вырезаем область из оригинального изображения
        digit = thresh[y1:y2, x1:x2]
        
        # Если вырезанная область пуста, создаем пустое изображение
        if digit.size == 0:
            digit = np.zeros((h + 2*padding_crop, w + 2*padding_crop, 3), dtype=np.uint8)
        
        digits.append(digit)
        
        # Для отладки: можно нарисовать bounding box на оригинальном изображении
        # cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return digits

def split_number_simple(img, start_x=8, digit_width=60):
    height, width = img.shape[:2]
    
    max_needed_x = start_x + 5 * digit_width
    if max_needed_x > width:
        digit_width = (width - start_x) // 5
    
    digits = []
    for i in range(5):
        x1 = start_x + i * digit_width
        x2 = min(x1 + digit_width, width)
        digit = img[:, x1:x2]
        
        if digit.size == 0:
            digit = np.zeros((height, digit_width, 3), dtype=np.uint8)
        
        digits.append(digit)
    
    return digits

def center_digits(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # if np.sum(binary == 255) > np.sum(binary == 0):
    #     binary = cv2.bitwise_not(binary)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)
    
    digits_area = image[y_min:y_max, x_min:x_max]
    h, w = digits_area.shape[:2]
    
    centered = np.ones_like(image) if len(image.shape) == 2 else np.ones_like(image)
    
    start_y = (image.shape[0] - h) // 2
    start_x = (image.shape[1] - w) // 2
    
    centered[start_y:start_y+h, start_x:start_x+w] = digits_area
    
    return centered