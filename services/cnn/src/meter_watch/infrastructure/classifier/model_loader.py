"""Loads a trained DigitRecognizer from disk."""
from __future__ import annotations

import os
from typing import Any, Optional, Type

import torch

from meter_watch.domain.protocols import Logger


class ModelLoader:
    """Loads a torch.nn.Module from a checkpoint file."""

    def __init__(
        self,
        model_class: Type[torch.nn.Module],
        path: str,
        device: torch.device,
        logger: Logger,
    ) -> None:
        self._model_class = model_class
        self._path = path
        self._device = device
        self._logger = logger

    def load(self) -> Optional[torch.nn.Module]:
        if not os.path.exists(self._path):
            self._logger.warning(f"no model at {self._path}")
            return None
        try:
            model = self._model_class().to(self._device)
            state: dict[str, Any] = torch.load(
                self._path, map_location=self._device
            )
            model.load_state_dict(state)
            model.eval()
            self._logger.info(f"model loaded: {self._path}")
            return model
        except Exception as exc:  # noqa: BLE001
            self._logger.error(f"could not load {self._path}: {exc}")
            return None
