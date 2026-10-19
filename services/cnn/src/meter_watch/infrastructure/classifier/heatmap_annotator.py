"""Annotators that produce base64 heatmaps."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from meter_watch.domain.models import DigitPrediction
from meter_watch.infrastructure.classifier.heatmap import SimpleHeatmap
from meter_watch.image_processing.encoding import image_to_base64
from meter_watch.image_processing.preprocessing import prepare_for_model


class GradCamAnnotator:
    """Produces Grad-CAM + saliency + digit crop base64 strings."""

    def __init__(
        self,
        model: torch.nn.Module,
        heatmap_generator: "SimpleHeatmap",
        device: torch.device,
    ) -> None:
        self._model = model
        self._heatmap = heatmap_generator
        self._device = device

    def annotate(
        self, roi: np.ndarray, prediction: DigitPrediction
    ) -> dict[str, str]:
        prepared, resized = prepare_for_model(roi)
        tensor = (
            torch.from_numpy(prepared).unsqueeze(0).unsqueeze(0).to(self._device)
        )

        cam, _, _ = self._heatmap.generate(tensor)
        saliency, _, _ = self._heatmap.generate_saliency(tensor)

        if cam is None:
            cam = np.zeros((28, 28), np.float32)
        if saliency is None:
            saliency = np.zeros((28, 28), np.float32)

        cam_vis = (cam * 255).astype(np.uint8)
        sal_vis = (saliency * 255).astype(np.uint8)

        return {
            "heatmap_gradcam": image_to_base64(
                cv2.applyColorMap(cam_vis, cv2.COLORMAP_JET)
            ),
            "heatmap_saliency": image_to_base64(
                cv2.applyColorMap(sal_vis, cv2.COLORMAP_HOT)
            ),
            "digit_image": image_to_base64(resized),
        }


class NoopAnnotator:
    """Annotator for tests / environments without torch."""

    def annotate(
        self, roi: np.ndarray, prediction: DigitPrediction
    ) -> dict[str, str]:
        return {
            "heatmap_gradcam": "",
            "heatmap_saliency": "",
            "digit_image": "",
        }
