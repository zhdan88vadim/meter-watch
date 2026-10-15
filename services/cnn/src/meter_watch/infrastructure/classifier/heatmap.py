"""Grad-CAM and saliency heatmap generation."""
from __future__ import annotations

import cv2
import numpy as np
import torch


class SimpleHeatmap:
    """Generates Grad-CAM and saliency maps for a CNN classifier."""

    def __init__(self, model) -> None:  # noqa: ANN001
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
        self.activations: dict = {}
        self.gradients: dict = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(name):
            def hook(module, input, output):  # noqa: ANN001, ARG001
                self.activations[name] = output.detach()
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):  # noqa: ANN001, ARG001
                self.gradients[name] = grad_output[0].detach()
            return hook

        self.model.conv1.register_forward_hook(forward_hook("conv1"))
        self.model.conv2.register_forward_hook(forward_hook("conv2"))
        self.model.conv1.register_full_backward_hook(backward_hook("conv1"))
        self.model.conv2.register_full_backward_hook(backward_hook("conv2"))

    def generate(self, image_tensor, target_class=None):  # noqa: ANN001
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)

        image_tensor.requires_grad_()
        output = self.model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)

        if target_class is None:
            target_class = prediction.item()

        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = self.gradients.get("conv2")
        activations = self.activations.get("conv2")
        if gradients is None or activations is None:
            return None, prediction.item(), confidence.item()

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (28, 28))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.power(cam, 0.7)
        return cam, prediction.item(), confidence.item()

    def generate_saliency(self, image_tensor, target_class=None):  # noqa: ANN001
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)

        image_tensor.requires_grad_()
        output = self.model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)

        if target_class is None:
            target_class = prediction.item()

        self.model.zero_grad()
        output[0, target_class].backward()

        saliency = image_tensor.grad.abs().squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency = np.power(saliency, 0.5)
        return saliency, prediction.item(), confidence.item()
