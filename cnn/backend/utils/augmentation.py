
import cv2
import torch
from torchvision import transforms
import numpy as np
from typing import Union, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import random


class AdaptiveAugmentationBuilder:
    """Адаптивные аугментации с кэшированием параметров"""

    def __init__(self, base_size=64):
        self.base_size = base_size
        self.size_cache = {}

                
        self.adaptive_preprocess_params = {
            'blur_ksize': 7,           # Уменьшено с 7 до 3
            'blur_sigma': 5,           # Уменьшено с 5 до 1
            'adaptive_block_size': 57, # Уменьшено с 57 до 11 (должно быть > 1 и нечетное)
            'adaptive_c': 5,           # Уменьшено с 5 до 3
            'morph_kernel': 2,         # Уменьшено с 2 до 1
            'morph_iter': 1            # Оставлено 1
        }
    
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
        # params = self.get_adaptive_params(image_size)

        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # BinarizeNP(apply_prob=1, threshold=100),
            ExtractLetterWithMargin(margin=20, fill_white=None),
            SquarePadAdaptBackground(min_size=128),
            # BinarizeCV(),
            # Invert(),
            AdaptivePreprocess(apply_prob=1, params=self.adaptive_preprocess_params),
            # transforms.Resize((128, 128)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,              # Угол поворота в градусах (-180 до 180) или (min, max)
                translate=(0.1, 0.1),   # Сдвиг: (по_горизонтали_макс%, по_вертикали_макс%)
                scale=(0.8, 1.1),       # Масштабирование: (мин_коэф, макс_коэф)
                shear=4,                # Наклон в градусах или (min, max) или (x_min, x_max, y_min, y_max)
                interpolation=2,        # Метод интерполяции (NEAREST=0, BILINEAR=2, BICUBIC=3)
                fill=0,                 # Цвет заливки для новых пикселей
            ),
            transforms.CenterCrop((90, 90)),
            transforms.Resize((28, 28)),
            # OnlyBrighten(max_brightness=2.5),
            RemoveSmallObjects(min_area=5, apply_prob=0.5),
            # MorphologicalTransform(
            #     erosion=(0, 2),      # эрозия 0-2 итерации
            #     dilation=(0, 2),     # дилатация 0-2 итерации
            #     kernel_size=(1, 3),  # ядро 1-3
            #     prob=1            # 50% вероятность
            # ),            
            
            # Binarize(),
            # OnlyBrighten(max_brightness=2),
            # transforms.Resize(64),
            # AdaptivePreprocess(),
            # ContourFilter(
            #     min_height=2,      # минимальная высота
            #     min_width=2,       # минимальная ширина
            #     min_area=5,       # минимальная площадь
            #     max_aspect_ratio=5, # макс соотношение сторон
            #     apply_prob=1.0     # вероятность применения (1.0 = всегда)
            # ),            
            # SquarePadAdaptBackground(min_size=128),
            # ExtractLetterWithMargin(margin=10, fill_white=False),
            # transforms.Resize(image_size),
            # Binarize(threshold=20, fill_white=True),
            # transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            # transforms.RandomRotation(2),
            # AddRandomBlobs(p=0.5, num_blobs=(3, 5), 
            #               blob_size=params['blob_size'], intensity=(250, 255)),
            # AddRandomBlobs(p=0.5, num_blobs=(3, 5),
            #               blob_size=params['blob_size'], intensity=(0, 5)),
            # AddRandomBlackSpots(p=0.5, num_spots=(2, 5),
            #                    spot_size=params['spot_size']),
            # RandomStrokeWidth(p=0.5, thickness_range=params['stroke_width']),
            # RandomBleed(p=0.5, blur_radius=params['blur_radius']),
            # RandomMissingPart(p=0.5, cut_size=params['cut_size']),
            # transforms.RandomAffine(
            #     degrees=0,
            #     translate=(0.1, 0.2),
            #     shear=3
            # ),
            transforms.ToTensor(),         
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def build_val_transform(self, image_size):
            
        adaptive_preprocess_params_for_small_images = {
                'blur_ksize': 3,           # Уменьшено с 7 до 3
                'blur_sigma': 1,           # Уменьшено с 5 до 1
                'adaptive_block_size': 11, # Уменьшено с 57 до 11 (должно быть > 1 и нечетное)
                'adaptive_c': 3,           # Уменьшено с 5 до 3
                'morph_kernel': 2,         # Уменьшено с 2 до 1
                'morph_iter': 1            # Оставлено 1
            }

        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # BinarizeCV(),
            # Invert(),            
            # SquarePadAdaptBackground(min_size=128),
            # AdaptivePreprocess(apply_prob=1, params=adaptive_preprocess_params_for_small_images),   
            # ExtractLetterWithMargin(margin=20, fill_white=None),            
            # transforms.CenterCrop((90, 90)),
            # transforms.Resize((28, 28)),

            # Invert(),
            # ExtractLetterWithMargin(margin=10, fill_white=None),
            # SquarePadAdaptBackground(),
            # transforms.Resize(image_size),
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


class RemoveSmallObjects:
    """Удаляет маленькие белые объекты, не связанные с большими"""
    
    def __init__(self, min_area=50, apply_prob=1.0, debug=False):
        """
        Args:
            min_area: минимальная площадь объекта (количество пикселей)
            apply_prob: вероятность применения
            debug: если True - показывает отладку
        """
        self.min_area = min_area
        self.apply_prob = apply_prob
        self.debug = debug
    
    def __call__(self, image):
        import cv2
        import numpy as np
        import random
        from PIL import Image
        
        if random.random() > self.apply_prob:
            return image
        
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('L'))
        else:
            image_np = image
        
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np
        
        # Бинаризация (белые объекты на черном фоне)
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # НАХОДИМ ВСЕ СВЯЗАННЫЕ КОМПОНЕНТЫ (области)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )
        
        # Создаем маску для больших объектов
        mask = np.zeros_like(thresh)
        
        # stats содержит: [x, y, w, h, area]
        for i in range(1, num_labels):  # i=0 это фон
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Если объект достаточно большой - сохраняем
            if area >= self.min_area:
                mask[labels == i] = 255
        
        # Применяем маску к оригинальному изображению
        result = cv2.bitwise_and(gray, gray, mask=mask)
       
        return Image.fromarray(result)

class MorphologicalTransform:
    """
    PyTorch трансформация для применения эрозии и дилатации
    
    Пример:
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            MorphologicalTransform(
                erosion=(0, 2),
                dilation=(0, 2),
                kernel_size=(1, 3)
            ),
            transforms.ToTensor(),
        ])
    """
    
    def __init__(
        self,
        erosion: tuple = (0, 2),          # (мин, макс) итераций эрозии
        dilation: tuple = (0, 2),         # (мин, макс) итераций дилатации
        kernel_size: tuple = (1, 3),      # (мин, макс) размер ядра
        kernel_type: str = 'ellipse',     # 'rect', 'ellipse', 'cross'
        prob: float = 0.5                 # вероятность применения
    ):
        self.erosion = erosion
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        self.prob = prob
    
    def __call__(self, img):
        # Проверка вероятности
        if random.random() > self.prob:
            return img
        
        # Конвертируем PIL в numpy
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # Сохраняем информацию о цвете
        is_color = len(img_np.shape) == 3
        
        # В оттенки серого
        if is_color:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Случайные параметры
        erosion_iter = random.randint(self.erosion[0], self.erosion[1])
        dilation_iter = random.randint(self.dilation[0], self.dilation[1])
        
        # Размер ядра (нечетный)
        ksize = random.randint(self.kernel_size[0], self.kernel_size[1])
        if ksize % 2 == 0:
            ksize += 1
        
        # Если ничего не делаем
        if erosion_iter == 0 and dilation_iter == 0:
            return img
        
        # Адаптивный порог
        block_size = random.randint(11, 51)
        if block_size % 2 == 0:
            block_size += 1
        
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            random.randint(2, 10)
        )
        
        # Создаем ядро
        if self.kernel_type == 'rect':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        elif self.kernel_type == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        else:  # cross
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
        
        # Применяем
        if erosion_iter > 0:
            processed = cv2.erode(processed, kernel, iterations=erosion_iter)
        if dilation_iter > 0:
            processed = cv2.dilate(processed, kernel, iterations=dilation_iter)
        
        # Обратно в PIL
        if is_color:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(processed)

class BinarizeCV:
    def __init__(self, apply_prob=1.0):
        self.apply_prob = apply_prob
    
    def __call__(self, image):
        import cv2
        import numpy as np
        import random
        
        if random.random() > self.apply_prob:
            return image
        
        # Конвертируем в numpy
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('L'))
        else:
            image_np = image
        
        # Адаптивная бинаризация
        # Используем адаптивный порог для разных фонов
        binary = cv2.adaptiveThreshold(
            image_np, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Или Otsu (проще, но менее адаптивно)
        # _, binary = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # # Инвертируем если цифры белые на черном
        # if np.mean(binary) > 127:  # если больше белого
        #     binary = 255 - binary
        
        from PIL import Image
        return Image.fromarray(binary)

class ContourFilter:
    """Фильтр мелких контуров"""
    
    def __init__(self, min_height=5, min_width=5, min_area=50, max_aspect_ratio=10, apply_prob=1.0):
        """
        Args:
            min_height: минимальная высота контура
            min_width: минимальная ширина контура
            min_area: минимальная площадь контура
            max_aspect_ratio: максимальное соотношение сторон
            apply_prob: вероятность применения (0.0 - 1.0)
        """
        self.min_height = min_height
        self.min_width = min_width
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.apply_prob = apply_prob
    
    def __call__(self, image):
        
        # Вероятность применения
        if random.random() > self.apply_prob:
            return image
        
        # # Конвертируем в numpy
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('L'))
        else:
            image_np = image
        
        # if len(image_np.shape) == 3:
        #     gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray = image_np
        
        # # Бинаризация
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Поиск контуров и фильтрация
        cnts, _ = cv2.findContours(image_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspect_ratio = w / h if h > 0 else 0
            
            if h > self.min_height and w > self.min_width and area > self.min_area and aspect_ratio < self.max_aspect_ratio:
                filtered_contours.append(c)
        
        if not filtered_contours:
            return image
        
        # 4. Создание маски
        mask = np.zeros_like(image_np)
        cv2.drawContours(mask, filtered_contours, -1, 255, -1)
        
        # 5. Добавление рамки и морфологическое закрытие
        padding = 70
        mask_padded = cv2.copyMakeBorder(
            mask.copy(),
            padding, padding, padding, padding,
            cv2.BORDER_CONSTANT,
            value=0
        )
        
        y_gap_threshold = 5
        kernel_height = y_gap_threshold * 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kernel_height))
        closed = cv2.morphologyEx(mask_padded, cv2.MORPH_CLOSE, kernel)
        
        # Удаление рамки
        closed = closed[padding:-padding, padding:-padding]
        
        # Применяем маску
        result = cv2.bitwise_and(image_np, image_np, mask=closed)

        return Image.fromarray(result)

class BinarizeNP:
    """Преобразует PIL Image в бинарное (черно-белое) с порогом"""
    def __init__(self, threshold=200, fill_white=True, apply_prob=1):
        self.threshold = threshold
        self.fill_white = fill_white
        self.apply_prob = apply_prob
    
    def __call__(self, image):
        # image - PIL Image
        if random.random() > self.apply_prob:
            return image   
                     
        img_np = np.array(image)
        
        if self.fill_white:
            # Значения выше порога становятся белыми (255)
            # Остальные остаются как есть
            result = np.where(img_np > self.threshold, 255, img_np).astype(np.uint8)
        else:
            # Полностью бинарное
            result = np.where(img_np > self.threshold, 255, 0).astype(np.uint8)
        
        return Image.fromarray(result)


class AdaptivePreprocess:
    """
    Адаптивная предобработка изображения с использованием OpenCV.
    Применяет CLAHE, адаптивный порог и морфологию.
    """
    def __init__(self, params=None, apply_prob=1):
        """
        Args:
            params: Словарь с параметрами предобработки
        """
        self.apply_prob = apply_prob
        
        if params is None:
            # self.params = {
            #     'blur_ksize': 7,
            #     'blur_sigma': 5,
            #     'adaptive_block_size': 57,
            #     'adaptive_c': 5,
            #     'morph_kernel': 2,
            #     'morph_iter': 1
            # }

            # Оптимизированные параметры для 28x28
            self.params = {
                'blur_ksize': 3,           # Уменьшено с 7 до 3
                'blur_sigma': 1,           # Уменьшено с 5 до 1
                'adaptive_block_size': 11, # Уменьшено с 57 до 11 (должно быть > 1 и нечетное)
                'adaptive_c': 3,           # Уменьшено с 5 до 3
                'morph_kernel': 1,         # Уменьшено с 2 до 1
                'morph_iter': 1            # Оставлено 1
            }            
        else:
            self.params = params
    
    def __call__(self, image):
        """
        Применяет предобработку к PIL Image.
        Возвращает PIL Image.
        """
        import random
        
        if random.random() > self.apply_prob:
            return image
        
        # Конвертируем PIL в numpy (RGB)
        img_np = np.array(image)
        
        # Если изображение в оттенках серого (1 канал), конвертируем в RGB
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        # Применяем предобработку
        processed = self._preprocess_image(img_np, self.params)
        
        # Конвертируем обратно в PIL
        return Image.fromarray(processed)
    
    def _preprocess_image(self, image, params):
        """
        Ваша оригинальная функция предобработки
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray, 
            (params['blur_ksize'], params['blur_ksize']), 
            params['blur_sigma']
        )
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            params['adaptive_block_size'],
            params['adaptive_c']
        )
        
        kernel = np.ones((params['morph_kernel'], params['morph_kernel']), dtype=np.uint8)
        opened = cv2.morphologyEx(
            thresh, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=params['morph_iter']
        )
        
        return opened
    
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
 
class OnlyBrighten:
    """Увеличивает яркость случайным образом, но не уменьшает."""
    
    def __init__(self, max_brightness=2):
        """
        Args:
            max_brightness: Максимальный коэффициент увеличения яркости (1.0 - без изменений)
        """
        self.max_brightness = max_brightness
    
    def __call__(self, img):
        # Случайный коэффициент от 1.0 до max_brightness
        brightness_factor = 1.0 + random.random() * (self.max_brightness - 1.0)
        return transforms.functional.adjust_brightness(img, brightness_factor)

class SquarePadAdaptBackground:
    """
    Дополняет изображение до квадрата или до минимальных размеров,
    заливая фон средним цветом краев.
    
    Args:
        border_size: Сколько пикселей брать с края для вычисления цвета фона
        min_size: Минимальный размер (ширина, высота) или одно число для обеих сторон.
                  Если None, то дополняет до квадрата по максимальной стороне.
                  Если задано, то доводит каждую сторону как минимум до этого значения.
    """
    def __init__(self, border_size: int = 2, min_size: Union[int, Tuple[int, int]] = None):
        self.border_size = border_size
        
        # Нормализуем min_size
        if min_size is None:
            self.min_size = None
        elif isinstance(min_size, int):
            self.min_size = (min_size, min_size)
        else:
            self.min_size = tuple(min_size)  # (width, height)

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Определяем целевые размеры
        if self.min_size is not None:
            target_w = max(w, self.min_size[0])
            target_h = max(h, self.min_size[1])
        else:
            # Старое поведение - квадрат по максимальной стороне
            target_w = target_h = max(w, h)
        
        # Если уже подходит по размерам, возвращаем как есть
        if w >= target_w and h >= target_h:
            return img
        
        # Вычисляем цвет фона
        fill_color = self._compute_fill_color(img_np)
        
        # Считаем отступы
        pad_left = (target_w - w) // 2
        pad_top = (target_h - h) // 2
        pad_right = target_w - w - pad_left
        pad_bottom = target_h - h - pad_top
        
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        img_padded = ImageOps.expand(img, padding, fill=fill_color)
        
        return img_padded
    
    def _compute_fill_color(self, img_np: np.ndarray):
        """Вычисляет средний цвет краев изображения."""
        b = self.border_size
        h, w = img_np.shape[:2]
        
        if len(img_np.shape) == 2:
            # Градации серого
            edges = np.concatenate([
                img_np[:b, :].ravel(),
                img_np[-b:, :].ravel(),
                img_np[:, :b].ravel(),
                img_np[:, -b:].ravel()
            ])
            median_val = np.median(edges).astype(np.uint8)
            
            if isinstance(median_val, np.ndarray):
                return int(median_val[0])
            return int(median_val)
        else:
            # Цветное (RGB)
            edges = np.concatenate([
                img_np[:b, :].reshape(-1, img_np.shape[2]),
                img_np[-b:, :].reshape(-1, img_np.shape[2]),
                img_np[:, :b].reshape(-1, img_np.shape[2]),
                img_np[:, -b:].reshape(-1, img_np.shape[2])
            ])
            median_val = np.median(edges, axis=0).astype(np.uint8)
            return tuple(map(int, median_val)) 

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
