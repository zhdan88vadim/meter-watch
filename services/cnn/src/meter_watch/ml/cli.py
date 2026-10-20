"""Training CLI entry point."""
from __future__ import annotations

import argparse

import torch

from meter_watch.ml.trainer import ModelTrainer


def train() -> None:
    parser = argparse.ArgumentParser(description="Train the digit recognizer")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--output", default="trained_models/digit_recognizer.pth")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = ModelTrainer(device=device, model_save_path=args.output)
    result = trainer.train(
        dataset_path=args.train_dir,
        val_dataset_path=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    print(result)


if __name__ == "__main__":
    train()
