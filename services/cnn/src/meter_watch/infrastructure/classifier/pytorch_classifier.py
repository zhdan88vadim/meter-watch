"""Adapter that wraps a torch.nn.Module behind DigitClassifier."""
from __future__ import annotations

import numpy as np
import torch

from meter_watch.domain.errors import ModelNotLoadedError
from meter_watch.domain.models import DigitPrediction
from meter_watch.domain.protocols import Logger
from meter_watch.image_processing.preprocessing import prepare_for_model


class PyTorchDigitClassifier:
    """Wraps a torch model to satisfy the DigitClassifier protocol."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        logger: Logger,
    ) -> None:
        if model is None:
            raise ModelNotLoadedError("model is None")
        self._model = model
        self._device = device
        self._logger = logger
        self._model.eval()

    def predict(self, roi: np.ndarray) -> DigitPrediction:
        prepared, _ = prepare_for_model(roi)
        tensor = (
            torch.from_numpy(prepared).unsqueeze(0).unsqueeze(0).to(self._device)
        )
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence = float(probs[0].max())
            value = int(logits.argmax(dim=1))
        return DigitPrediction(value=value, confidence=confidence)
