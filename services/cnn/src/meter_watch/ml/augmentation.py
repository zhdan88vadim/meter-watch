"""Data augmentation pipeline for training.

Ported from the original utils/augmentation.py. Kept as-is except for
imports and docstrings.
"""
from __future__ import annotations

import random
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torchvision import transforms


class AdaptiveAugmentationBuilder:
    """Builds train/val transform pipelines with size-adaptive parameters."""

    def __init__(self, base_size: int = 64) -> None:
        self.base_size = base_size
        self.adaptive_preprocess_params: dict[str, int] = {
            "blur_ksize": 7,
            "blur_sigma": 5,
            "adaptive_block_size": 57,
            "adaptive_c": 5,
            "morph_kernel": 2,
            "morph_iter": 1,
        }

    def build_train_transform(
        self, image_size: Tuple[int, int]
    ) -> transforms.Compose:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            ExtractLetterWithMargin(margin=20, fill_white=None),
            SquarePadAdaptBackground(min_size=128),
            AdaptivePreprocess(
                apply_prob=1, params=self.adaptive_preprocess_params
            ),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.7, 1.1),
                shear=4,
                interpolation=2,
                fill=0,
            ),
            transforms.CenterCrop((90, 90)),
            transforms.Resize((28, 28)),
            transforms.Pad(padding=4, fill=0),
            transforms.Resize((28, 28)),
            RemoveSmallObjects(min_area=5, apply_prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def build_val_transform(
        self, image_size: Tuple[int, int]
    ) -> transforms.Compose:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.Pad(padding=4, fill=0),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


class RemoveSmallObjects:
    """Remove small white objects not connected to larger ones."""

    def __init__(
        self, min_area: int = 50, apply_prob: float = 1.0, debug: bool = False
    ) -> None:
        self.min_area = min_area
        self.apply_prob = apply_prob
        self.debug = debug

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.apply_prob:
            return image

        image_np = np.array(image.convert("L"))
        gray = (
            cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            if len(image_np.shape) == 3
            else image_np
        )

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )

        mask = np.zeros_like(thresh)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                mask[labels == i] = 255

        result = cv2.bitwise_and(gray, gray, mask=mask)
        return Image.fromarray(result)


class AdaptivePreprocess:
    """Adaptive image preprocessing using OpenCV."""

    def __init__(
        self,
        params: dict[str, int] | None = None,
        apply_prob: float = 1.0,
    ) -> None:
        self.apply_prob = apply_prob
        if params is None:
            self.params: dict[str, int] = {
                "blur_ksize": 3,
                "blur_sigma": 1,
                "adaptive_block_size": 11,
                "adaptive_c": 3,
                "morph_kernel": 1,
                "morph_iter": 1,
            }
        else:
            self.params = params

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.apply_prob:
            return image

        img_np = np.array(image)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        processed = self._preprocess_image(img_np, self.params)
        return Image.fromarray(processed)

    def _preprocess_image(
        self, image: np.ndarray, params: dict[str, int]
    ) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (params["blur_ksize"], params["blur_ksize"]),
            params["blur_sigma"],
        )
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            params["adaptive_block_size"],
            params["adaptive_c"],
        )
        kernel = np.ones(
            (params["morph_kernel"], params["morph_kernel"]), dtype=np.uint8
        )
        return cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=params["morph_iter"]
        )


class SquarePadAdaptBackground:
    """Pad image to square using the median edge color as background."""

    def __init__(
        self,
        border_size: int = 2,
        min_size: Union[int, Tuple[int, ...]] | None = None,
    ) -> None:
        self.border_size = border_size
        self.min_size: tuple[int, ...] | None
        if min_size is None:
            self.min_size = None
        elif isinstance(min_size, int):
            self.min_size = (min_size, min_size)
        else:
            self.min_size = tuple(min_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        if self.min_size is not None:
            target_w = max(w, self.min_size[0])
            target_h = max(h, self.min_size[1])
        else:
            target_w = target_h = max(w, h)

        if w >= target_w and h >= target_h:
            return img

        fill_color = self._compute_fill_color(img_np)
        pad_left = (target_w - w) // 2
        pad_top = (target_h - h) // 2
        pad_right = target_w - w - pad_left
        pad_bottom = target_h - h - pad_top
        return ImageOps.expand(
            img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill_color
        )

    def _compute_fill_color(self, img_np: np.ndarray) -> int | tuple[int, ...]:
        b = self.border_size
        if len(img_np.shape) == 2:
            edges = np.concatenate([
                img_np[:b, :].ravel(),
                img_np[-b:, :].ravel(),
                img_np[:, :b].ravel(),
                img_np[:, -b:].ravel(),
            ])
            val = np.median(edges).astype(np.uint8)
            return int(val) if not isinstance(val, np.ndarray) else int(val[0])
        edges = np.concatenate([
            img_np[:b, :].reshape(-1, img_np.shape[2]),
            img_np[-b:, :].reshape(-1, img_np.shape[2]),
            img_np[:, :b].reshape(-1, img_np.shape[2]),
            img_np[:, -b:].reshape(-1, img_np.shape[2]),
        ])
        return tuple(map(int, np.median(edges, axis=0).astype(np.uint8)))


class ExtractLetterWithMargin:
    """Extract a letter by contour with margin."""

    def __init__(self, margin: int = 10, fill_white: bool | None = True) -> None:
        self.margin = margin
        self.fill_white = fill_white

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        gray = (
            cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            if len(img_np.shape) == 3
            else img_np
        )
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return img

        all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
        x, y, w, h = cv2.boundingRect(all_pts)

        x1 = max(0, x - self.margin)
        y1 = max(0, y - self.margin)
        x2 = min(img_np.shape[1], x + w + self.margin)
        y2 = min(img_np.shape[0], y + h + self.margin)
        cropped: np.ndarray = img_np[y1:y2, x1:x2]

        if self.fill_white:
            target_h = h + 2 * self.margin
            target_w = w + 2 * self.margin
            if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
                if len(img_np.shape) == 3:
                    canvas: np.ndarray = np.ones(
                        (target_h, target_w, img_np.shape[2]), dtype=np.uint8
                    ) * 255
                else:
                    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
                y_off = (target_h - cropped.shape[0]) // 2
                x_off = (target_w - cropped.shape[1]) // 2
                canvas[
                    y_off : y_off + cropped.shape[0],
                    x_off : x_off + cropped.shape[1],
                ] = cropped
                cropped = canvas

        return Image.fromarray(cropped)


# --- Additional augmentations (ported as-is) ---


class MorphologicalTransform:
    """Erosion / dilation transform."""

    def __init__(
        self,
        erosion: tuple[int, int] = (0, 2),
        dilation: tuple[int, int] = (0, 2),
        kernel_size: tuple[int, int] = (1, 3),
        kernel_type: str = "ellipse",
        prob: float = 0.5,
    ) -> None:
        self.erosion = erosion
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img

        img_np = np.array(img)
        is_color = len(img_np.shape) == 3
        gray = (
            cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if is_color else img_np
        )

        erosion_iter = random.randint(*self.erosion)
        dilation_iter = random.randint(*self.dilation)
        ksize = random.randint(*self.kernel_size)
        if ksize % 2 == 0:
            ksize += 1

        if erosion_iter == 0 and dilation_iter == 0:
            return img

        block_size = random.randint(11, 51)
        if block_size % 2 == 0:
            block_size += 1

        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            random.randint(2, 10),
        )

        if self.kernel_type == "rect":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        elif self.kernel_type == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))

        if erosion_iter > 0:
            processed = cv2.erode(processed, kernel, iterations=erosion_iter)
        if dilation_iter > 0:
            processed = cv2.dilate(processed, kernel, iterations=dilation_iter)

        if is_color:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(processed)


class BinarizeCV:
    """Adaptive binarization with OpenCV."""

    def __init__(self, apply_prob: float = 1.0) -> None:
        self.apply_prob = apply_prob

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.apply_prob:
            return image
        image_np = np.array(image.convert("L"))
        binary = cv2.adaptiveThreshold(
            image_np,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        return Image.fromarray(binary)


class ContourFilter:
    """Filter small contours and reapply as a mask."""

    def __init__(
        self,
        min_height: int = 5,
        min_width: int = 5,
        min_area: int = 50,
        max_aspect_ratio: float = 10,
        apply_prob: float = 1.0,
    ) -> None:
        self.min_height = min_height
        self.min_width = min_width
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.apply_prob = apply_prob

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.apply_prob:
            return image
        image_np = np.array(image.convert("L"))
        cnts, _ = cv2.findContours(
            image_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered = []
        for c in cnts:
            _, _, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspect = w / h if h > 0 else 0
            if (
                h > self.min_height
                and w > self.min_width
                and area > self.min_area
                and aspect < self.max_aspect_ratio
            ):
                filtered.append(c)
        if not filtered:
            return image

        mask = np.zeros_like(image_np)
        cv2.drawContours(mask, filtered, -1, 255, -1)

        padding = 70
        mask_padded = cv2.copyMakeBorder(
            mask.copy(), padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=0,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
        closed = cv2.morphologyEx(mask_padded, cv2.MORPH_CLOSE, kernel)
        closed = closed[padding:-padding, padding:-padding]

        result = cv2.bitwise_and(image_np, image_np, mask=closed)
        return Image.fromarray(result)


class BinarizeNP:
    """Convert PIL image to binary using a fixed threshold."""

    def __init__(
        self, threshold: int = 200, fill_white: bool = True, apply_prob: float = 1.0
    ) -> None:
        self.threshold = threshold
        self.fill_white = fill_white
        self.apply_prob = apply_prob

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.apply_prob:
            return image
        img_np = np.array(image)
        if self.fill_white:
            result = np.where(img_np > self.threshold, 255, img_np).astype(np.uint8)
        else:
            result = np.where(img_np > self.threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(result)


class OnlyBrighten:
    """Increase brightness randomly, never decrease."""

    def __init__(self, max_brightness: float = 2.0) -> None:
        self.max_brightness = max_brightness

    def __call__(self, img: Image.Image) -> Image.Image:
        factor = 1.0 + random.random() * (self.max_brightness - 1.0)
        result: Image.Image = transforms.functional.adjust_brightness(img, factor)
        return result


class SquarePad:
    """Pad to square with black or white fill."""

    def __init__(self, fill_white: bool = False) -> None:
        self.fill_white = fill_white

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        max_side = max(width, height)
        pad_left = (max_side - width) // 2
        pad_top = (max_side - height) // 2
        pad_right = max_side - width - pad_left
        pad_bottom = max_side - height - pad_top
        fill = 255 if self.fill_white else 0
        return ImageOps.expand(
            img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill
        )


class SimpleThinOrThicken:
    """Erode (thin) a letter by a random number of iterations."""

    def __init__(
        self,
        p: float = 0.9,
        strength: str = "strong",
        min_thickness: int = 1,
    ) -> None:
        self.p = p
        self.min_thickness = min_thickness
        if strength == "light":
            self.iterations = 1
        elif strength == "medium":
            self.iterations = 2
        elif strength == "strong":
            self.iterations = 3
        else:
            self.iterations = int(strength)

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() > self.p:
            return img
        img_np = np.array(img)
        kernel = np.ones((3, 3), np.uint8)
        if len(img_np.shape) == 2:
            return Image.fromarray(
                cv2.erode(img_np, kernel, iterations=self.iterations)
            )
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        result = cv2.erode(gray, kernel, iterations=self.iterations)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))


class Invert:
    """Invert an image."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return Image.fromarray(255 - np.array(img))


class AddGaussianNoise:
    """Add Gaussian noise to a tensor."""

    def __init__(
        self, std_range: tuple[float, float] = (0.1, 0.8), p: float = 1.0
    ) -> None:
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return tensor
        std = np.random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0, 1)


class RandomMissingPart:
    """Simulate a missing rectangle of a letter."""

    def __init__(
        self, p: float = 0.3, cut_size: tuple[int, int] = (5, 15)
    ) -> None:
        self.p = p
        self.cut_size = cut_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        cut_h = random.randint(self.cut_size[0], min(self.cut_size[1], h // 3))
        cut_w = random.randint(self.cut_size[0], min(self.cut_size[1], w // 3))
        x = random.randint(0, w - cut_w)
        y = random.randint(0, h - cut_h)
        if len(img_np.shape) == 3:
            img_np[y : y + cut_h, x : x + cut_w, :] = 255
        else:
            img_np[y : y + cut_h, x : x + cut_w] = 255
        return Image.fromarray(img_np)


class RandomBleed:
    """Simulate ink bleed via Gaussian blur."""

    def __init__(
        self, p: float = 0.3, blur_radius: tuple[float, float] = (0.5, 1.5)
    ) -> None:
        self.p = p
        self.blur_radius = blur_radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.blur_radius)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class AddRandomBlobs:
    """Add random bright blobs."""

    def __init__(
        self,
        p: float = 0.5,
        num_blobs: tuple[int, int] = (2, 5),
        blob_size: tuple[int, int] = (4, 5),
        intensity: tuple[int, int] = (200, 255),
    ) -> None:
        self.p = p
        self.num_blobs = num_blobs
        self.blob_size = blob_size
        self.intensity = intensity

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img_copy.size
        for _ in range(random.randint(*self.num_blobs)):
            bw = random.randint(*self.blob_size)
            bh = random.randint(*self.blob_size)
            x = random.randint(0, width - bw)
            y = random.randint(0, height - bh)
            val = random.randint(*self.intensity)
            if random.choice([True, False]):
                draw.rectangle([x, y, x + bw, y + bh], fill=val)
            else:
                draw.ellipse([x, y, x + bw, y + bh], fill=val)
        return img_copy


class RandomStrokeWidth:
    """Randomly change stroke width."""

    def __init__(
        self, p: float = 0.5, thickness_range: tuple[int, int] = (-1, 2)
    ) -> None:
        self.p = p
        self.thickness_range = thickness_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img_np = np.array(img.convert("L"))
        thickness = random.randint(*self.thickness_range)
        if thickness > 0:
            kernel = np.ones((thickness + 1, thickness + 1), np.uint8)
            img_np = cv2.dilate(img_np, kernel, iterations=1)
        elif thickness < 0:
            kernel = np.ones((abs(thickness) + 1, abs(thickness) + 1), np.uint8)
            img_np = cv2.erode(img_np, kernel, iterations=1)
        return Image.fromarray(img_np)


class AddRandomBlackSpots:
    """Add random dark spots."""

    def __init__(
        self,
        p: float = 0.5,
        num_spots: tuple[int, int] = (3, 6),
        spot_size: tuple[int, int] = (3, 6),
    ) -> None:
        self.p = p
        self.num_spots = num_spots
        self.spot_size = spot_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img_copy.size
        for _ in range(random.randint(*self.num_spots)):
            sw = random.randint(*self.spot_size)
            sh = random.randint(*self.spot_size)
            x = random.randint(0, width - sw)
            y = random.randint(0, height - sh)
            draw.rectangle([x, y, x + sw, y + sh], fill=0)
        return img_copy