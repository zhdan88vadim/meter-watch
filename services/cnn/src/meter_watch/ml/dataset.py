"""Dataset builders for training."""
from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from meter_watch.ml.augmentation import AdaptiveAugmentationBuilder


def build_dataloaders(
    train_dir: str,
    val_dir: str,
    image_size: int = 28,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from ImageFolder-style dirs."""
    aug = AdaptiveAugmentationBuilder(base_size=image_size)
    train_ds = ImageFolder(
        root=train_dir,
        transform=aug.build_train_transform((image_size, image_size)),
    )
    val_ds = ImageFolder(
        root=val_dir,
        transform=aug.build_val_transform((image_size, image_size)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
