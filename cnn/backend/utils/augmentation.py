
import cv2
import torch
from torchvision import transforms
import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import random


class AdaptiveAugmentationBuilder:
    """Адаптивные аугментации с кэшированием параметров"""
    
    def __init__(self, base_size=64):
        self.base_size = base_size
        self.size_cache = {}
    
    def get_adaptive_params(self, current_size):
        """Вычисляет параметры аугментаций на основе размера"""
        if current_size in self.size_cache:
            return self.size_cache[current_size]
        
        scale = current_size[0] / self.base_size
        
        params = {
            'blob_size': (max(1, int(2 * scale)), max(1, int(4 * scale))),
            'spot_size': (max(1, int(2 * scale)), max(1, int(4 * scale))),
            'cut_size': (max(1, int(2 * scale)), max(1, int(4 * scale))),
            'blur_radius': (0.5 * scale, 1.2 * scale),
            'stroke_width': (-max(1, int(1 * scale)), max(1, int(2 * scale))),
            'translate': (0.1 * (scale**0.5), 0.2 * (scale**0.5)),
            'shear': 10 * scale,
            'degrees': 10 * min(1.0, scale)
        }
        
        self.size_cache[current_size] = params
        return params
    
    def build_train_transform(self, image_size):
        params = self.get_adaptive_params(image_size)

        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            ExtractLetterWithMargin(margin=10, fill_white=False),
            SquarePad(fill_white=False),
            transforms.Resize(image_size),
            # Binarize(threshold=20, fill_white=True),
            # transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            # AddRandomBlobs(p=0.5, num_blobs=(3, 5), 
            #               blob_size=params['blob_size'], intensity=(250, 255)),
            # AddRandomBlobs(p=0.5, num_blobs=(3, 5),
            #               blob_size=params['blob_size'], intensity=(0, 5)),
            # AddRandomBlackSpots(p=0.5, num_spots=(2, 5),
            #                    spot_size=params['spot_size']),
            # RandomStrokeWidth(p=0.5, thickness_range=params['stroke_width']),
            # RandomBleed(p=0.5, blur_radius=params['blur_radius']),
            # RandomMissingPart(p=0.5, cut_size=params['cut_size']),
            transforms.RandomAffine(
                degrees=params['degrees'],
                translate=params['translate'],
                shear=params['shear']
            ),
            transforms.ToTensor(),         
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def build_val_transform(self, image_size):
        
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # Invert(),
            ExtractLetterWithMargin(margin=10, fill_white=False),
            SquarePad(fill_white=False),
            transforms.Resize(image_size),
            # Binarize(threshold=20, fill_white=True),
            # Invert(),
            # SimpleThinOrThicken(p=1, strength='medium', min_thickness=1),
            # Invert(),
            # transforms.Lambda(lambda x: 255 - np.array(x) if isinstance(x, Image.Image) else 255 - x),
            # transforms.ToPILImage(),  # обратно в PIL        
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

class Binarize:
    """Преобразует PIL Image в бинарное (черно-белое) с порогом"""
    def __init__(self, threshold=200, fill_white=True):
        self.threshold = threshold
        self.fill_white = fill_white
    
    def __call__(self, image):
        # image - PIL Image
        import numpy as np
        img_np = np.array(image)
        
        if self.fill_white:
            # Значения выше порога становятся белыми (255)
            # Остальные остаются как есть
            result = np.where(img_np > self.threshold, 255, img_np).astype(np.uint8)
        else:
            # Полностью бинарное
            result = np.where(img_np > self.threshold, 255, 0).astype(np.uint8)
        
        return Image.fromarray(result)

# class Binarize:
#     """Преобразует изображение в бинарное (черно-белое) с порогом"""
#     def __init__(self, threshold=200, fill_white=True):
#         """
#         threshold: значение порога (0-255)
#         fill_white: если True, значения выше порога становятся белыми (1.0),
#                    если False - черными (0.0)
#         """
#         self.threshold = threshold
#         self.fill_white = fill_white
    
#     def __call__(self, image):
#         # image - тензор [C, H, W] в диапазоне [0, 1]
#         threshold_norm = self.threshold / 255.0
        
#         if self.fill_white:
#             # Серые -> белые, черные остаются черными
#             return torch.where(image > threshold_norm, torch.tensor(1.0), image)
#         else:
#             # Полностью бинарное: выше порога - белые, ниже - черные
#             return torch.where(image > threshold_norm, torch.tensor(1.0), torch.tensor(0.0))
        

class SquarePad:
    """
    Adds padding to the image to make it square.
    Size is determined by the longer side.
    """
    def __init__(self, fill_white=False):
        """
        Args:
            fill_value: fill value (0-255) if fill_white=False
            fill_white: if True - white padding (255), if False - black (fill_value)
        """        
        self.fill_white = fill_white
    
    def __call__(self, img):
        # Get image dimensions
        width, height = img.size
        
        # Determine square size (larger side)
        max_side = max(width, height)
        
        # Calculate required padding
        pad_left = (max_side - width) // 2
        pad_top = (max_side - height) // 2
        pad_right = max_side - width - pad_left
        pad_bottom = max_side - height - pad_top
        
        # Determine padding color
        if self.fill_white:
            fill_color = 255
        else:
            fill_color = 0
        
        # Add padding
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        img_padded = ImageOps.expand(img, padding, fill=fill_color)
        
        return img_padded


class ExtractLetterWithMargin:
    """Вырезает букву по контуру с добавлением отступа"""
    
    def __init__(self, margin=10, fill_white=True):
        self.margin = margin
        self.fill_white = fill_white
    
    def __call__(self, img):
        # Конвертируем PIL в numpy (если нужно)
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # Если изображение цветное, конвертируем в оттенки серого для поиска контура
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Бинаризация изображения
        # _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Поиск контуров
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
        
        # Объединяем все контуры в один bounding box
        all_contours = np.vstack([contour.reshape(-1, 2) for contour in contours])
        x, y, w, h = cv2.boundingRect(all_contours)
        
        # Добавляем отступ
        x1 = max(0, x - self.margin)
        y1 = max(0, y - self.margin)
        x2 = min(img_np.shape[1], x + w + self.margin)
        y2 = min(img_np.shape[0], y + h + self.margin)
        
        # Вырезаем область с отступом
        cropped = img_np[y1:y2, x1:x2]
        
        # Если нужно заполнить недостающие пиксели белым
        if self.fill_white:
            # Получаем целевую ширину и высоту (исходный размер + отступы)
            target_h = h + 2 * self.margin
            target_w = w + 2 * self.margin
            
            # Проверяем, нужно ли расширять изображение
            if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
                # Создаем белый холст нужного размера
                if len(img_np.shape) == 3:
                    canvas = np.ones((target_h, target_w, img_np.shape[2]), dtype=np.uint8) * 255
                else:
                    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
                
                # Вычисляем позицию для вставки (центрируем)
                y_offset = (target_h - cropped.shape[0]) // 2
                x_offset = (target_w - cropped.shape[1]) // 2
                
                # Вставляем вырезанную область
                canvas[y_offset:y_offset+cropped.shape[0], 
                       x_offset:x_offset+cropped.shape[1]] = cropped
                cropped = canvas
        
        # Конвертируем обратно в PIL
        return Image.fromarray(cropped)

class SimpleThinOrThicken:
    """Только утоньшение букв (делает их тонкими) - упрощенная версия"""
    
    def __init__(self, p=0.9, strength='strong', min_thickness=1):
        """
        Args:
            p: вероятность применения (0-1)
            strength: 'light', 'medium', 'strong' или число итераций
            min_thickness: минимальная толщина линии в пикселях (1-10)
        """
        self.p = p
        self.min_thickness = min_thickness
        
        if strength == 'light':
            self.iterations = 1
        elif strength == 'medium':
            self.iterations = 2
        elif strength == 'strong':
            self.iterations = 3
        else:
            self.iterations = int(strength)
    
    def __call__(self, img):
        if np.random.random() > self.p:
            return img
        
        # Конвертируем в numpy
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        kernel = np.ones((3,3), np.uint8)
        
        # Для оттенков серого
        if len(img_np.shape) == 2:
            # Просто применяем эрозию нужное количество раз
            result = cv2.erode(img_np, kernel, iterations=self.iterations)
            return Image.fromarray(result)
        
        # Для цветных
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            result_gray = cv2.erode(gray, kernel, iterations=self.iterations)
            result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(result)


class Invert:
    """Инвертирование изображения"""
    def __call__(self, img):
        return Image.fromarray(255 - np.array(img))

class AddGaussianNoise:
    """Гауссовский шум для тензоров"""
    def __init__(self, std_range=(0.1, 0.8), p=1):
        self.std_range = std_range
        self.p = p
    
    def __call__(self, tensor):
        if np.random.random() > self.p:
            return tensor
        
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0, 1)

class RandomMissingPart(object):
    """Симулирует отсутствующую часть буквы (вырезает случайный прямоугольник)"""
    def __init__(self, p=0.3, cut_size=(5, 15)):
        self.p = p
        self.cut_size = cut_size
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        cut_h = random.randint(self.cut_size[0], min(self.cut_size[1], h//3))
        cut_w = random.randint(self.cut_size[0], min(self.cut_size[1], w//3))
        
        x = random.randint(0, w - cut_w)
        y = random.randint(0, h - cut_h)
        
        # Заполняем белым (фоном)
        if len(img_np.shape) == 3:
            img_np[y:y+cut_h, x:x+cut_w, :] = 255
        else:
            img_np[y:y+cut_h, x:x+cut_w] = 255
        
        return Image.fromarray(img_np)

class RandomBleed(object):
    """Симулирует растекание чернил (размытие краев)"""
    def __init__(self, p=0.3, blur_radius=(0.5, 1.5)):
        self.p = p
        self.blur_radius = blur_radius
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        radius = random.uniform(self.blur_radius[0], self.blur_radius[1])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class AddRandomBlobs(object):
    """Добавляет случайные крупные пятна (блюбы) размером 4-5 пикселей"""
    def __init__(self, p=0.5, num_blobs=(2, 5), blob_size=(4, 5), intensity=(200, 255)):
        """
        p: вероятность применения
        num_blobs: диапазон количества пятен (min, max)
        blob_size: диапазон размера пятен (min, max)
        intensity: диапазон интенсивности (min, max) - для белого шума
        """
        self.p = p
        self.num_blobs = num_blobs
        self.blob_size = blob_size
        self.intensity = intensity
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        # Конвертируем в numpy для работы
        if isinstance(img, torch.Tensor):
            # Если тензор, конвертируем в PIL
            img = transforms.ToPILImage()(img)
        
        # Создаем копию для рисования
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        width, height = img_copy.size
        
        # Добавляем случайные пятна
        num_blobs = random.randint(self.num_blobs[0], self.num_blobs[1])
        
        for _ in range(num_blobs):
            # Случайный размер пятна
            blob_w = random.randint(self.blob_size[0], self.blob_size[1])
            blob_h = random.randint(self.blob_size[0], self.blob_size[1])
            
            # Случайная позиция
            x = random.randint(0, width - blob_w)
            y = random.randint(0, height - blob_h)
            
            # Случайная интенсивность
            intensity_val = random.randint(self.intensity[0], self.intensity[1])
            
            # Рисуем залитый эллипс или прямоугольник
            if random.choice([True, False]):
                # Прямоугольник
                draw.rectangle([x, y, x + blob_w, y + blob_h], fill=intensity_val)
            else:
                # Эллипс (круглое пятно)
                draw.ellipse([x, y, x + blob_w, y + blob_h], fill=intensity_val)
        
        return img_copy

class RandomStrokeWidth(object):
    """Случайно изменяет толщину линий (утолщает или истончает)"""
    def __init__(self, p=0.5, thickness_range=(-1, 2)):
        """
        thickness_range: диапазон изменения толщины (отрицательные - истончение, положительные - утолщение)
        """
        self.p = p
        self.thickness_range = thickness_range
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        # Конвертируем в numpy для морфологических операций
        img_np = np.array(img.convert('L'))
        
        thickness = random.randint(self.thickness_range[0], self.thickness_range[1])
        
        if thickness > 0:
            # Утолщение (дилатация)
            kernel = np.ones((thickness+1, thickness+1), np.uint8)
            img_np = cv2.dilate(img_np, kernel, iterations=1)
        elif thickness < 0:
            # Истончение (эрозия)
            kernel = np.ones((abs(thickness)+1, abs(thickness)+1), np.uint8)
            img_np = cv2.erode(img_np, kernel, iterations=1)
        
        return Image.fromarray(img_np)

class AddRandomBlackSpots(object):
    """Добавляет черные пятна (типа грязи) размером 4-5 пикселей"""
    def __init__(self, p=0.5, num_spots=(3, 6), spot_size=(3, 6)):
        self.p = p
        self.num_spots = num_spots
        self.spot_size = spot_size
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        width, height = img_copy.size
        num_spots = random.randint(self.num_spots[0], self.num_spots[1])
        
        for _ in range(num_spots):
            spot_w = random.randint(self.spot_size[0], self.spot_size[1])
            spot_h = random.randint(self.spot_size[0], self.spot_size[1])
            
            x = random.randint(0, width - spot_w)
            y = random.randint(0, height - spot_h)
            
            # Черные пятна
            draw.rectangle([x, y, x + spot_w, y + spot_h], fill=0)
        
        return img_copy
