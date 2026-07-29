import cv2
import torch
from torchvision import transforms
import numpy as np
from typing import Union, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import random


class AdaptiveAugmentationBuilder:
    """Adaptive augmentations with parameter caching"""

    def __init__(self, base_size=64):
        self.base_size = base_size
        self.size_cache = {}

                
        self.adaptive_preprocess_params = {
            'blur_ksize': 7,          
            'blur_sigma': 5,   
            'adaptive_block_size': 57, 
            'adaptive_c': 5,           
            'morph_kernel': 2,         
            'morph_iter': 1            
        }
    
    def get_adaptive_params(self, current_size):
        """Calculates augmentation parameters based on size"""
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

        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            ExtractLetterWithMargin(margin=20, fill_white=None),
            SquarePadAdaptBackground(min_size=128),
            AdaptivePreprocess(apply_prob=1, params=self.adaptive_preprocess_params),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,              # Rotation angle in degrees (-180 to 180) or (min, max)
                translate=(0.1, 0.1),   # Translation: (horizontal_max%, vertical_max%)
                scale=(0.7, 1.1),       # Scaling: (min_coef, max_coef)
                shear=4,                # Shear in degrees or (min, max) or (x_min, x_max, y_min, y_max)
                interpolation=2,        # Interpolation method (NEAREST=0, BILINEAR=2, BICUBIC=3)
                fill=0,                 # Fill color for new pixels
            ),
            transforms.CenterCrop((90, 90)),
            transforms.Resize((28, 28)),
            transforms.Pad(padding=4, fill=0),
            transforms.Resize((28, 28)),
            RemoveSmallObjects(min_area=5, apply_prob=0.5),
            transforms.ToTensor(),         
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def build_val_transform(self, image_size):

        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.Pad(padding=4, fill=0),
            transforms.Resize((28, 28)),    
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


class RemoveSmallObjects:
    """Removes small white objects not connected to larger ones"""
    
    def __init__(self, min_area=50, apply_prob=1.0, debug=False):
        """
        Args:
            min_area: minimum area of an object (number of pixels)
            apply_prob: probability of application
            debug: if True - shows debug info
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
        
        # Binarization (white objects on black background)
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # FIND ALL CONNECTED COMPONENTS (regions)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )
        
        # Create mask for large objects
        mask = np.zeros_like(thresh)
        
        # stats contains: [x, y, w, h, area]
        for i in range(1, num_labels):  # i=0 is the background
            area = stats[i, cv2.CC_STAT_AREA]
            
            # If object is large enough - keep it
            if area >= self.min_area:
                mask[labels == i] = 255
        
        # Apply mask to original image
        result = cv2.bitwise_and(gray, gray, mask=mask)
       
        return Image.fromarray(result)

class MorphologicalTransform:
    """
    PyTorch transformation for applying erosion and dilation
    
    Example:
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
        erosion: tuple = (0, 2),          # (min, max) erosion iterations
        dilation: tuple = (0, 2),         # (min, max) dilation iterations
        kernel_size: tuple = (1, 3),      # (min, max) kernel size
        kernel_type: str = 'ellipse',     # 'rect', 'ellipse', 'cross'
        prob: float = 0.5                 # probability of application
    ):
        self.erosion = erosion
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        self.prob = prob
    
    def __call__(self, img):
        # Check probability
        if random.random() > self.prob:
            return img
        
        # Convert PIL to numpy
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # Save color information
        is_color = len(img_np.shape) == 3
        
        # To grayscale
        if is_color:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Random parameters
        erosion_iter = random.randint(self.erosion[0], self.erosion[1])
        dilation_iter = random.randint(self.dilation[0], self.dilation[1])
        
        # Kernel size (odd)
        ksize = random.randint(self.kernel_size[0], self.kernel_size[1])
        if ksize % 2 == 0:
            ksize += 1
        
        # If nothing to do
        if erosion_iter == 0 and dilation_iter == 0:
            return img
        
        # Adaptive threshold
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
        
        # Create kernel
        if self.kernel_type == 'rect':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        elif self.kernel_type == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        else:  # cross
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
        
        # Apply
        if erosion_iter > 0:
            processed = cv2.erode(processed, kernel, iterations=erosion_iter)
        if dilation_iter > 0:
            processed = cv2.dilate(processed, kernel, iterations=dilation_iter)
        
        # Back to PIL
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
        
        # Convert to numpy
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('L'))
        else:
            image_np = image
        
        # Adaptive binarization
        # Use adaptive threshold for different backgrounds
        binary = cv2.adaptiveThreshold(
            image_np, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Or Otsu (simpler, but less adaptive)
        # _, binary = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # # Invert if digits are white on black
        # if np.mean(binary) > 127:  # if more white
        #     binary = 255 - binary
        
        from PIL import Image
        return Image.fromarray(binary)

class ContourFilter:
    """Filter for small contours"""
    
    def __init__(self, min_height=5, min_width=5, min_area=50, max_aspect_ratio=10, apply_prob=1.0):
        """
        Args:
            min_height: minimum contour height
            min_width: minimum contour width
            min_area: minimum contour area
            max_aspect_ratio: maximum aspect ratio
            apply_prob: probability of application (0.0 - 1.0)
        """
        self.min_height = min_height
        self.min_width = min_width
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.apply_prob = apply_prob
    
    def __call__(self, image):
        
        # Probability of application
        if random.random() > self.apply_prob:
            return image
        
        # # Convert to numpy
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('L'))
        else:
            image_np = image
        
        # if len(image_np.shape) == 3:
        #     gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray = image_np
        
        # # Binarization
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Find contours and filter
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
        
        # 4. Create mask
        mask = np.zeros_like(image_np)
        cv2.drawContours(mask, filtered_contours, -1, 255, -1)
        
        # 5. Add border and morphological closing
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
        
        # Remove border
        closed = closed[padding:-padding, padding:-padding]
        
        # Apply mask
        result = cv2.bitwise_and(image_np, image_np, mask=closed)

        return Image.fromarray(result)

class BinarizeNP:
    """Converts PIL Image to binary (black and white) with threshold"""
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
            # Values above threshold become white (255)
            # Others remain as is
            result = np.where(img_np > self.threshold, 255, img_np).astype(np.uint8)
        else:
            # Fully binary: above threshold - white, below - black
            result = np.where(img_np > self.threshold, 255, 0).astype(np.uint8)
        
        return Image.fromarray(result)


class AdaptivePreprocess:
    """
    Adaptive image preprocessing using OpenCV.
    Applies CLAHE, adaptive threshold, and morphology.
    """
    def __init__(self, params=None, apply_prob=1):
        """
        Args:
            params: Dictionary with preprocessing parameters
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

            # Optimized parameters for 28x28
            self.params = {
                'blur_ksize': 3,           # Reduced from 7 to 3
                'blur_sigma': 1,           # Reduced from 5 to 1
                'adaptive_block_size': 11, # Reduced from 57 to 11 (must be > 1 and odd)
                'adaptive_c': 3,           # Reduced from 5 to 3
                'morph_kernel': 1,         # Reduced from 2 to 1
                'morph_iter': 1            # Kept at 1
            }            
        else:
            self.params = params
    
    def __call__(self, image):
        """
        Applies preprocessing to PIL Image.
        Returns PIL Image.
        """
        import random
        
        if random.random() > self.apply_prob:
            return image
        
        # Convert PIL to numpy (RGB)
        img_np = np.array(image)
        
        # If image is grayscale (1 channel), convert to RGB
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        # Apply preprocessing
        processed = self._preprocess_image(img_np, self.params)
        
        # Convert back to PIL
        return Image.fromarray(processed)
    
    def _preprocess_image(self, image, params):
        """
        Your original preprocessing function
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

class OnlyBrighten:
    """Increases brightness randomly, but never decreases."""
    
    def __init__(self, max_brightness=2):
        """
        Args:
            max_brightness: Maximum brightness increase factor (1.0 - no change)
        """
        self.max_brightness = max_brightness
    
    def __call__(self, img):
        # Random factor from 1.0 to max_brightness
        brightness_factor = 1.0 + random.random() * (self.max_brightness - 1.0)
        return transforms.functional.adjust_brightness(img, brightness_factor)

class SquarePadAdaptBackground:
    """
    Pads image to square or to minimum dimensions,
    filling the background with the average color of the edges.
    
    Args:
        border_size: How many pixels to take from the edge to compute background color
        min_size: Minimum size (width, height) or a single number for both sides.
                  If None, pads to square based on the larger side.
                  If specified, brings each side to at least this value.
    """
    def __init__(self, border_size: int = 2, min_size: Union[int, Tuple[int, int]] = None):
        self.border_size = border_size
        
        # Normalize min_size
        if min_size is None:
            self.min_size = None
        elif isinstance(min_size, int):
            self.min_size = (min_size, min_size)
        else:
            self.min_size = tuple(min_size)  # (width, height)

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Determine target dimensions
        if self.min_size is not None:
            target_w = max(w, self.min_size[0])
            target_h = max(h, self.min_size[1])
        else:
            # Old behavior - square based on the larger side
            target_w = target_h = max(w, h)
        
        # If already fits, return as is
        if w >= target_w and h >= target_h:
            return img
        
        # Compute fill color
        fill_color = self._compute_fill_color(img_np)
        
        # Calculate padding
        pad_left = (target_w - w) // 2
        pad_top = (target_h - h) // 2
        pad_right = target_w - w - pad_left
        pad_bottom = target_h - h - pad_top
        
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        img_padded = ImageOps.expand(img, padding, fill=fill_color)
        
        return img_padded
    
    def _compute_fill_color(self, img_np: np.ndarray):
        """Computes the average color of the image edges."""
        b = self.border_size
        h, w = img_np.shape[:2]
        
        if len(img_np.shape) == 2:
            # Grayscale
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
            # Color (RGB)
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
    """Extracts a letter by contour with margin"""
    
    def __init__(self, margin=10, fill_white=True):
        self.margin = margin
        self.fill_white = fill_white
    
    def __call__(self, img):
        # Convert PIL to numpy (if needed)
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # If image is color, convert to grayscale for contour detection
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Binarization
        # _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
        
        # Merge all contours into one bounding box
        all_contours = np.vstack([contour.reshape(-1, 2) for contour in contours])
        x, y, w, h = cv2.boundingRect(all_contours)
        
        # Add margin
        x1 = max(0, x - self.margin)
        y1 = max(0, y - self.margin)
        x2 = min(img_np.shape[1], x + w + self.margin)
        y2 = min(img_np.shape[0], y + h + self.margin)
        
        # Crop with margin
        cropped = img_np[y1:y2, x1:x2]
        
        # If need to fill missing pixels with white
        if self.fill_white:
            # Get target width and height (original size + margins)
            target_h = h + 2 * self.margin
            target_w = w + 2 * self.margin
            
            # Check if image needs to be expanded
            if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
                # Create white canvas of desired size
                if len(img_np.shape) == 3:
                    canvas = np.ones((target_h, target_w, img_np.shape[2]), dtype=np.uint8) * 255
                else:
                    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
                
                # Calculate insertion position (center)
                y_offset = (target_h - cropped.shape[0]) // 2
                x_offset = (target_w - cropped.shape[1]) // 2
                
                # Insert cropped region
                canvas[y_offset:y_offset+cropped.shape[0], 
                       x_offset:x_offset+cropped.shape[1]] = cropped
                cropped = canvas
        
        # Convert back to PIL
        return Image.fromarray(cropped)

class SimpleThinOrThicken:
    """Only thins letters (makes them thin) - simplified version"""
    
    def __init__(self, p=0.9, strength='strong', min_thickness=1):
        """
        Args:
            p: probability of application (0-1)
            strength: 'light', 'medium', 'strong' or number of iterations
            min_thickness: minimum line thickness in pixels (1-10)
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
        
        # Convert to numpy
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        kernel = np.ones((3,3), np.uint8)
        
        # For grayscale
        if len(img_np.shape) == 2:
            # Simply apply erosion the required number of times
            result = cv2.erode(img_np, kernel, iterations=self.iterations)
            return Image.fromarray(result)
        
        # For color
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            result_gray = cv2.erode(gray, kernel, iterations=self.iterations)
            result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(result)


class Invert:
    """Invert image"""
    def __call__(self, img):
        return Image.fromarray(255 - np.array(img))

class AddGaussianNoise:
    """Gaussian noise for tensors"""
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
    """Simulates a missing part of a letter (removes a random rectangle)"""
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
        
        # Fill with white (background)
        if len(img_np.shape) == 3:
            img_np[y:y+cut_h, x:x+cut_w, :] = 255
        else:
            img_np[y:y+cut_h, x:x+cut_w] = 255
        
        return Image.fromarray(img_np)

class RandomBleed(object):
    """Simulates ink bleed (edge blur)"""
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
    """Adds random large blobs (size 4-5 pixels)"""
    def __init__(self, p=0.5, num_blobs=(2, 5), blob_size=(4, 5), intensity=(200, 255)):
        """
        p: probability of application
        num_blobs: range of number of blobs (min, max)
        blob_size: range of blob size (min, max)
        intensity: range of intensity (min, max) - for white noise
        """
        self.p = p
        self.num_blobs = num_blobs
        self.blob_size = blob_size
        self.intensity = intensity
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        # Convert to numpy for processing
        if isinstance(img, torch.Tensor):
            # If tensor, convert to PIL
            img = transforms.ToPILImage()(img)
        
        # Create a copy for drawing
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        width, height = img_copy.size
        
        # Add random blobs
        num_blobs = random.randint(self.num_blobs[0], self.num_blobs[1])
        
        for _ in range(num_blobs):
            # Random blob size
            blob_w = random.randint(self.blob_size[0], self.blob_size[1])
            blob_h = random.randint(self.blob_size[0], self.blob_size[1])
            
            # Random position
            x = random.randint(0, width - blob_w)
            y = random.randint(0, height - blob_h)
            
            # Random intensity
            intensity_val = random.randint(self.intensity[0], self.intensity[1])
            
            # Draw filled ellipse or rectangle
            if random.choice([True, False]):
                # Rectangle
                draw.rectangle([x, y, x + blob_w, y + blob_h], fill=intensity_val)
            else:
                # Ellipse (round blob)
                draw.ellipse([x, y, x + blob_w, y + blob_h], fill=intensity_val)
        
        return img_copy

class RandomStrokeWidth(object):
    """Randomly changes line thickness (thickens or thins)"""
    def __init__(self, p=0.5, thickness_range=(-1, 2)):
        """
        thickness_range: range of thickness change (negative - thinning, positive - thickening)
        """
        self.p = p
        self.thickness_range = thickness_range
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        # Convert to numpy for morphological operations
        img_np = np.array(img.convert('L'))
        
        thickness = random.randint(self.thickness_range[0], self.thickness_range[1])
        
        if thickness > 0:
            # Thickening (dilation)
            kernel = np.ones((thickness+1, thickness+1), np.uint8)
            img_np = cv2.dilate(img_np, kernel, iterations=1)
        elif thickness < 0:
            # Thinning (erosion)
            kernel = np.ones((abs(thickness)+1, abs(thickness)+1), np.uint8)
            img_np = cv2.erode(img_np, kernel, iterations=1)
        
        return Image.fromarray(img_np)

class AddRandomBlackSpots(object):
    """Adds black spots (like dirt) size 3-6 pixels"""
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
            
            # Black spots
            draw.rectangle([x, y, x + spot_w, y + spot_h], fill=0)
        
        return img_copy