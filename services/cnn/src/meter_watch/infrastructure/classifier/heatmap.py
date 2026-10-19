from typing import Callable, Optional

import cv2
import numpy as np
import torch


class SimpleHeatmap:
    """Generates Grad-CAM and saliency maps for a CNN classifier."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
        self.activations: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(name: str) -> Callable[..., None]:
            def hook(
                module: object,
                input: object,
                output: torch.Tensor,
            ) -> None:
                self.activations[name] = output.detach()
            return hook

        def backward_hook(name: str) -> Callable[..., None]:
            def hook(
                module: object,
                grad_input: object,
                grad_output: tuple[torch.Tensor, ...],
            ) -> None:
                self.gradients[name] = grad_output[0].detach()
            return hook

        self.model.conv1.register_forward_hook(forward_hook("conv1"))
        self.model.conv2.register_forward_hook(forward_hook("conv2"))
        self.model.conv1.register_full_backward_hook(backward_hook("conv1"))
        self.model.conv2.register_full_backward_hook(backward_hook("conv2"))

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[Optional[np.ndarray], int, float]:
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)

        image_tensor.requires_grad_()
        output = self.model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)

        target: int = (
            target_class if target_class is not None else int(prediction.item())
        )
        pred_i = int(prediction.item())
        conf_f = float(confidence.item())

        self.model.zero_grad()
        output[0, target].backward()

        gradients = self.gradients.get("conv2")
        activations = self.activations.get("conv2")
        if gradients is None or activations is None:
            return None, pred_i, conf_f

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam_t = torch.sum(weights * activations, dim=1, keepdim=True)
        cam_np: np.ndarray = torch.relu(cam_t).squeeze().cpu().numpy()
        cam_np = cv2.resize(cam_np, (28, 28))
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        cam_np = np.power(cam_np, 0.7)
        return cam_np, pred_i, conf_f

    def generate_saliency(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)

        image_tensor.requires_grad_()
        output = self.model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)

        target: int = (
            target_class if target_class is not None else int(prediction.item())
        )
        pred_i = int(prediction.item())
        conf_f = float(confidence.item())

        self.model.zero_grad()
        output[0, target].backward()

        grad = image_tensor.grad
        assert grad is not None
        saliency: np.ndarray = grad.abs().squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (
            saliency.max() - saliency.min() + 1e-8
        )
        saliency = np.power(saliency, 0.5)
        return saliency, pred_i, conf_f
