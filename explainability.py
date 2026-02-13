"""
ShotSense Explainability — Grad-CAM Style Attribution.

Visualizes which regions of an image contribute most to its style score.
Uses Gradient-weighted Class Activation Mapping adapted for style similarity.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from style_encoder import ShotSenseEncoder


class StyleGradCAM:
    """
    Grad-CAM for style attribution — visualizes STYLE-discriminative regions.
    
    Usage:
        gradcam = StyleGradCAM(encoder)
        heatmap = gradcam.generate(image, prototype)
    """

    def __init__(self, encoder: ShotSenseEncoder):
        self.encoder = encoder
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on the last transformer block."""
        last_block = self.encoder.backbone.transformer_blocks[-1]

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()

        last_block.register_forward_hook(forward_hook)
        last_block.register_full_backward_hook(backward_hook)

    @torch.enable_grad()
    def generate(
        self,
        image: torch.Tensor,
        prototype: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Generate a style attribution heatmap.
        
        Args:
            image: (1, 3, 224, 224) single image tensor
            prototype: (128,) optional style prototype for similarity target
        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.encoder.eval()
        image = image.clone().requires_grad_(True)

        # Forward
        result = self.encoder(image)
        style_vector = result["style_vector"]

        # Target scalar for backprop
        if prototype is not None:
            target = F.cosine_similarity(
                style_vector, prototype.unsqueeze(0), dim=-1
            )
        else:
            target = style_vector.norm(dim=-1)

        # Backward
        self.encoder.zero_grad()
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros(image_size)

        # Grad-CAM: weight activations by gradient importance
        patch_activations = self.activations[:, 1:, :]  # skip CLS
        patch_gradients = self.gradients[:, 1:, :]

        cam = (patch_activations * patch_gradients).sum(dim=-1)
        cam = F.relu(cam)

        # Reshape 49 patches → 7x7 grid → upsample
        grid_size = int(cam.shape[1] ** 0.5)
        cam = cam.view(1, 1, grid_size, grid_size)
        cam = F.interpolate(cam, size=image_size, mode="bilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def generate_full_analysis(
        self,
        image: torch.Tensor,
        prototype: torch.Tensor,
        prototype_attributes: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Full style analysis with heatmap + attribute gap.
        
        Returns:
            dict with heatmap, score, attributes, and gap analysis
        """
        heatmap = self.generate(image, prototype)

        with torch.no_grad():
            result = self.encoder(image)
            score = F.cosine_similarity(
                result["style_vector"], prototype.unsqueeze(0), dim=-1
            )
            score = ((score + 1) / 2 * 100).clamp(0, 100).item()

            attributes = {
                name: val.item()
                for name, val in result["attributes"].items()
            }

        # Gap analysis if prototype attributes provided
        gaps = {}
        if prototype_attributes:
            for name in attributes:
                if name in prototype_attributes:
                    diff = attributes[name] - prototype_attributes[name]
                    direction = "too high" if diff > 0.1 else "too low" if diff < -0.1 else "good match"
                    gaps[name] = {
                        "value": attributes[name],
                        "target": prototype_attributes[name],
                        "diff": diff,
                        "assessment": direction,
                    }

        return {
            "heatmap": heatmap,
            "score": score,
            "attributes": attributes,
            "gaps": gaps,
        }
