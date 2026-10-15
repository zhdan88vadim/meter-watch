"""Model trainer. Extracted from the original trainer.py."""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from meter_watch.ml.dataset import build_dataloaders
from meter_watch.ml.model import DigitRecognizer


class ModelTrainer:
    """Trains a DigitRecognizer from ImageFolder-style directories."""

    def __init__(self, device: torch.device, model_save_path: str) -> None:
        self.device = device
        self.model_save_path = model_save_path
        self.model: DigitRecognizer | None = None
        self.train_losses: list[float] = []
        self.val_accuracies: list[float] = []

    def train(
        self,
        dataset_path: str,
        val_dataset_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        image_size: int = 28,
        num_workers: int = 4,
    ) -> dict:
        print("\nStarting model training...")
        try:
            train_loader, val_loader = build_dataloaders(
                dataset_path, val_dataset_path, image_size, batch_size, num_workers
            )
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": str(exc)}

        classes = train_loader.dataset.classes

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("runs", f"folder_training_{timestamp}")
        writer = SummaryWriter(log_dir)

        self.model = DigitRecognizer().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3
        )

        self.train_losses = []
        self.val_accuracies = []
        all_val_preds: list[int] = []
        all_val_labels: list[int] = []

        start_time = time.time()
        global_step = 0
        best_accuracy = 0.0

        try:
            for epoch in range(epochs):
                avg_loss, global_step = self._train_epoch(
                    train_loader, optimizer, criterion, writer, global_step,
                    epoch
                )
                self.train_losses.append(avg_loss)
                writer.add_scalar("Training/Epoch Loss", avg_loss, epoch)

                val_accuracy, preds, labels = self._validate(
                    val_loader, criterion
                )
                self.val_accuracies.append(val_accuracy)
                scheduler.step(100 - val_accuracy)
                writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)
                all_val_preds.extend(preds)
                all_val_labels.extend(labels)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}] "
                        f"Loss: {avg_loss:.4f} Val Acc: {val_accuracy:.2f}%"
                    )

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    self._save_model()
                    print(f"Saved new best model ({val_accuracy:.2f}%)")

                if epoch % 5 == 0:
                    self._log_image_grids(writer, train_loader, val_loader, epoch)

            self._log_final_report(
                all_val_labels, all_val_preds, classes, writer, best_accuracy
            )
            self._log_embeddings(writer, train_loader, classes)

            training_time = time.time() - start_time
            writer.flush()
            writer.close()
            self._save_model()

            return {
                "success": True,
                "epochs": epochs,
                "final_loss": self.train_losses[-1],
                "final_accuracy": self.val_accuracies[-1] if self.val_accuracies else None,
                "training_time": training_time,
                "model_path": self.model_save_path,
                "num_samples": len(train_loader.dataset),
                "dataset_path": dataset_path,
                "tensorboard_dir": log_dir,
            }
        except Exception as exc:  # noqa: BLE001
            writer.close()
            return {"success": False, "error": str(exc)}

    # ---------- internals ----------

    def _train_epoch(self, loader, optimizer, criterion, writer, global_step, epoch):
        self.model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                writer.add_scalar("Training/Batch Loss", loss.item(), global_step)
            global_step += 1
        return running_loss / len(loader), global_step

    def _validate(self, loader, criterion):
        self.model.eval()
        correct = 0
        total = 0
        preds: list[int] = []
        labels_all: list[int] = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                preds.extend(predicted.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
        return 100 * correct / total, preds, labels_all

    def _log_image_grids(self, writer, train_loader, val_loader, epoch):
        images, _ = next(iter(train_loader))
        writer.add_image(
            f"Training/Epoch_{epoch}",
            vutils.make_grid(images[:32], nrow=4, normalize=True),
            epoch,
        )
        images_val, _ = next(iter(val_loader))
        writer.add_image(
            f"Val/Epoch_{epoch}",
            vutils.make_grid(images_val[:32], nrow=4, normalize=True),
            epoch,
        )

    def _log_final_report(self, y_true, y_pred, classes, writer, best_accuracy):
        if not y_pred:
            return
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = np.nan_to_num(
            cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax1,
        )
        ax1.set_title("Confusion Matrix (Counts)")
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Reds",
            xticklabels=classes, yticklabels=classes, ax=ax2,
        )
        ax2.set_title("Confusion Matrix (Normalized)")
        plt.tight_layout()
        writer.add_figure("Validation/Confusion_Matrix", fig, 0)

        os.makedirs("logs/confusion_matrices", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(
            f"logs/confusion_matrices/confusion_matrix_{ts}.png", dpi=300
        )
        plt.close()

        report = classification_report(
            y_true, y_pred, target_names=classes, digits=3
        )
        print("\nClassification Report:\n", report)
        Path("classification_report.txt").write_text(
            f"Best Accuracy: {best_accuracy:.2f}%\n\n{report}"
        )

    def _log_embeddings(self, writer, train_loader, classes, max_samples=500):
        images_list, labels_list = [], []
        for images, labels in train_loader:
            images_list.append(images)
            labels_list.append(labels)
            if sum(len(l) for l in labels_list) >= max_samples:
                break

        images = torch.cat(images_list)[:max_samples]
        labels = torch.cat(labels_list)[:max_samples]

        self.model.eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), 32):
                batch = images[i : i + 32].to(self.device)
                feats.append(self.model.get_features(batch).cpu())
        features = torch.cat(feats)

        metadata = [f"{classes[label]}_{i:03d}" for i, label in enumerate(labels)]
        writer.add_embedding(
            features,
            metadata=metadata,
            label_img=images,
            global_step=0,
            tag="Model_Features_Embeddings",
        )
        writer.flush()

    def _save_model(self) -> str:
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_save_path)
        return self.model_save_path
