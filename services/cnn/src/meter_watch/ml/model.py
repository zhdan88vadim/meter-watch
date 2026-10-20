"""PyTorch model definition for digit recognition."""
from __future__ import annotations

import torch
import torch.nn as nn


class DigitRecognizer(nn.Module):
    """CNN for digit recognition with batch normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        out: torch.Tensor = self.fc2(x)
        return out

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-dim features before the final classification layer."""
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        out: torch.Tensor = self.relu(self.bn3(self.fc1(x)))
        return out

    def get_intermediate_features(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        """Return features at each level (for debugging / visualization)."""
        features: dict[str, torch.Tensor | dict[str, float]] = {}

        x1 = self.pool(self.relu(self.bn1(self.conv1(x))))
        features["after_conv1"] = x1
        features["after_conv1_stats"] = _stats(x1)

        x2 = self.pool(self.relu(self.bn2(self.conv2(x1))))
        features["after_conv2"] = x2
        features["after_conv2_stats"] = _stats(x2)

        x3 = x2.view(-1, 64 * 7 * 7)
        features["flattened"] = x3

        x4 = self.relu(self.bn3(self.fc1(x3)))
        features["after_fc1"] = x4
        features["after_fc1_stats"] = _stats(x4)

        return features


def _stats(t: torch.Tensor) -> dict[str, float]:
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
    }
