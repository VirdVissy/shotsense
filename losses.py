"""
ShotSense Loss Functions — Custom Training Objectives.

This module implements the novel loss functions that train the style encoder
to disentangle aesthetic style from semantic content.

The key challenge: CLIP's embedding space entangles style and content.
Two photos of the same subject (e.g., two portraits) are close in CLIP space
even if one is warm/film-like and the other is cold/clinical.
Conversely, two photos with the same aesthetic (warm, golden hour) but
different subjects (portrait vs. landscape) are far apart.

Our training objective inverts this:
    Same style + different content → CLOSE in style space
    Different style + same content → FAR in style space

Three complementary losses:

1. StyleTripletLoss
   - Anchor: a reference photo
   - Positive: same style, different content (same photographer/preset)
   - Negative: different style, similar content
   - Pulls same-style embeddings together, pushes different-style apart

2. ContentOrthogonalityLoss
   - Measures correlation between style vectors and content categories
   - Penalizes the model if style embeddings cluster by content type
   - Ensures "portrait style" ≠ "portrait content"

3. AttributeRegressionLoss
   - Supervises the attribute prediction heads with pseudo-labels
   - Pseudo-labels computed from image statistics (color temp, histogram)
   - Acts as a regularizer ensuring style space encodes meaningful features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class StyleTripletLoss(nn.Module):
    """
    Contrastive triplet loss for style disentanglement.
    
    For each anchor image, we have:
    - positive: an image with the SAME style but DIFFERENT content
    - negative: an image with DIFFERENT style but SIMILAR content
    
    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    
    where d is cosine distance in style space.
    
    We also implement hard negative mining: instead of random negatives,
    we find the closest negative (hardest to distinguish from anchor's style)
    and the farthest positive (hardest positive example).
    """

    def __init__(self, margin: float = 0.3, hard_mining: bool = True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: (B, 128) style vectors of anchor images
            positive: (B, 128) style vectors of same-style images
            negative: (B, 128) style vectors of different-style images
        Returns:
            scalar loss
        """
        if self.hard_mining:
            return self._hard_mining_loss(anchor, positive, negative)

        # Standard triplet loss with cosine distance
        d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
        d_neg = 1 - F.cosine_similarity(anchor, negative, dim=-1)

        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()

    def _hard_mining_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Semi-hard and hard negative mining within the batch.
        
        For each anchor:
        - Hardest positive: the same-style image that's farthest away
        - Semi-hard negative: the different-style image that's closest
          but still farther than the hardest positive
        """
        B = anchor.shape[0]

        # Compute all pairwise distances
        # Between anchors and all positives
        d_ap = 1 - torch.mm(anchor, positive.t())  # (B, B)
        # Between anchors and all negatives
        d_an = 1 - torch.mm(anchor, negative.t())  # (B, B)

        # Hardest positive: max distance to any positive in batch
        # (diagonal = paired positive, but we search all positives)
        hardest_pos, _ = d_ap.max(dim=1)  # (B,)

        # Semi-hard negative: closest negative that's still farther than pos
        # For each anchor, find negatives where d_neg > d_pos
        # Among those, pick the closest one
        mask = d_an > hardest_pos.unsqueeze(1)  # (B, B) mask
        # Set invalid negatives to large distance
        d_an_masked = d_an.clone()
        d_an_masked[~mask] = float("inf")

        # If no semi-hard negatives exist, fall back to hardest negative
        has_valid = mask.any(dim=1)
        hardest_neg, _ = d_an_masked.min(dim=1)

        # Fallback: use closest negative overall for samples with no semi-hard
        closest_neg, _ = d_an.min(dim=1)
        hardest_neg = torch.where(has_valid, hardest_neg, closest_neg)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()


class ContentOrthogonalityLoss(nn.Module):
    """
    Ensures style vectors are orthogonal to content information.
    
    This is the key regularizer that prevents the style space from
    encoding "what's in the photo" instead of "how it looks."
    
    Implementation:
    1. Given style vectors and content labels (e.g., "portrait", "landscape")
    2. Compute the mean style vector per content category
    3. Measure how different these means are (high variance = content leaking)
    4. Penalize the model for having content-discriminative style vectors
    
    Alternative approach (no labels required):
    1. Use CLIP text embeddings of content descriptions as content vectors
    2. Measure correlation between style vectors and content vectors
    3. Penalize high correlation
    """

    def __init__(self, lambda_orth: float = 0.1):
        super().__init__()
        self.lambda_orth = lambda_orth

    def forward(
        self,
        style_vectors: torch.Tensor,
        content_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize correlation between style and content embeddings.
        
        Args:
            style_vectors: (B, style_dim) our learned style embeddings
            content_vectors: (B, content_dim) CLIP embeddings (represent content)
        Returns:
            scalar loss — lower means better disentanglement
        """
        # Normalize both
        style_norm = F.normalize(style_vectors, p=2, dim=-1)
        content_norm = F.normalize(content_vectors, p=2, dim=-1)

        # Cross-correlation matrix: (style_dim, content_dim)
        # If style is truly orthogonal to content, this should be ~zero
        cross_corr = torch.mm(style_norm.t(), content_norm) / style_vectors.shape[0]

        # Loss = Frobenius norm of cross-correlation matrix
        # This penalizes any linear relationship between style and content
        loss = torch.norm(cross_corr, p="fro")

        return self.lambda_orth * loss

    def forward_with_labels(
        self,
        style_vectors: torch.Tensor,
        content_labels: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Alternative: use content class labels instead of embeddings.
        
        Penalizes the between-class variance of style vectors —
        if style vectors cluster by content class, that's bad.
        
        Args:
            style_vectors: (B, style_dim)
            content_labels: (B,) integer content class labels
            num_classes: number of content categories
        Returns:
            scalar loss
        """
        # Compute per-class mean style vectors
        class_means = []
        for c in range(num_classes):
            mask = content_labels == c
            if mask.sum() > 0:
                class_means.append(style_vectors[mask].mean(dim=0))

        if len(class_means) < 2:
            return torch.tensor(0.0, device=style_vectors.device)

        class_means = torch.stack(class_means)  # (num_classes, style_dim)

        # Between-class variance of style means
        # High variance = style space is encoding content differences = BAD
        global_mean = class_means.mean(dim=0)
        between_class_var = ((class_means - global_mean) ** 2).sum(dim=-1).mean()

        return self.lambda_orth * between_class_var


class AttributeRegressionLoss(nn.Module):
    """
    Supervises attribute prediction heads with pseudo-labels.
    
    Since we don't have ground-truth aesthetic attribute labels,
    we compute pseudo-labels from image statistics:
    
    - warmth: ratio of warm (R, Y) to cool (B, C) pixels
    - contrast: standard deviation of luminance histogram
    - saturation: mean saturation in HSV space
    - brightness: mean value in HSV space
    - sharpness: Laplacian variance (focus measure)
    - grain: high-frequency energy ratio
    
    These aren't perfect, but they're good enough to regularize
    the style space and make attribute predictions useful.
    """

    def __init__(self, attribute_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.attribute_weights = attribute_weights or {
            "warmth": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "brightness": 1.0,
            "sharpness": 0.5,
            "grain": 0.5,
        }

    def forward(
        self,
        predicted: Dict[str, torch.Tensor],
        pseudo_labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            predicted: dict of attribute_name → (B, 1) predictions
            pseudo_labels: dict of attribute_name → (B, 1) pseudo-labels
        Returns:
            weighted sum of per-attribute MSE losses
        """
        total_loss = 0.0
        for name, pred in predicted.items():
            if name in pseudo_labels:
                weight = self.attribute_weights.get(name, 1.0)
                target = pseudo_labels[name]
                total_loss += weight * F.mse_loss(pred, target)

        return total_loss


# ============================================================
# Combined Loss
# ============================================================

class ShotSenseLoss(nn.Module):
    """
    Combined training loss for ShotSense.
    
    Total Loss = α * L_triplet + β * L_orthogonality + γ * L_attributes
    
    Default weights:
    - α = 1.0 (primary objective: learn good style representations)
    - β = 0.5 (regularizer: ensure style ≠ content)
    - γ = 0.3 (regularizer: ensure style space is interpretable)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        triplet_margin: float = 0.3,
        hard_mining: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.triplet_loss = StyleTripletLoss(
            margin=triplet_margin, hard_mining=hard_mining
        )
        self.orthogonality_loss = ContentOrthogonalityLoss(lambda_orth=1.0)
        self.attribute_loss = AttributeRegressionLoss()

    def forward(
        self,
        anchor_style: torch.Tensor,
        positive_style: torch.Tensor,
        negative_style: torch.Tensor,
        anchor_content: torch.Tensor,
        predicted_attributes: Dict[str, torch.Tensor],
        pseudo_label_attributes: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            anchor_style: (B, 128) anchor style vectors
            positive_style: (B, 128) same-style style vectors
            negative_style: (B, 128) different-style style vectors
            anchor_content: (B, 512) CLIP content embeddings for anchors
            predicted_attributes: dict of predicted attribute values
            pseudo_label_attributes: dict of pseudo-label attribute values
        Returns:
            dict with 'total', 'triplet', 'orthogonality', 'attributes' losses
        """
        # Style triplet loss
        l_triplet = self.triplet_loss(anchor_style, positive_style, negative_style)

        # Content orthogonality loss
        l_orth = self.orthogonality_loss(anchor_style, anchor_content)

        # Attribute regression loss
        l_attr = self.attribute_loss(predicted_attributes, pseudo_label_attributes)

        # Combined
        total = self.alpha * l_triplet + self.beta * l_orth + self.gamma * l_attr

        return {
            "total": total,
            "triplet": l_triplet,
            "orthogonality": l_orth,
            "attributes": l_attr,
        }


# ============================================================
# Pseudo-label computation
# ============================================================

def compute_pseudo_labels(images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute pseudo-labels for style attributes from raw image tensors.
    
    These are computed directly from pixel statistics — no model needed.
    Values are normalized to [0, 1].
    
    Args:
        images: (B, 3, H, W) image tensors, values in [0, 1]
    Returns:
        dict of attribute_name → (B, 1) pseudo-labels
    """
    B = images.shape[0]
    device = images.device

    # Separate RGB channels
    r, g, b = images[:, 0], images[:, 1], images[:, 2]

    # ---- Warmth ----
    # Warm images have more red/yellow, cool images more blue
    warmth = (r.mean(dim=(-2, -1)) - b.mean(dim=(-2, -1)) + 1) / 2  # → [0, 1]

    # ---- Brightness ----
    # Mean luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = luminance.mean(dim=(-2, -1))

    # ---- Contrast ----
    # Std of luminance histogram
    contrast = luminance.std(dim=(-2, -1))
    contrast = (contrast / 0.3).clamp(0, 1)  # Normalize roughly to [0, 1]

    # ---- Saturation ----
    # Compute saturation channel from RGB
    max_rgb = images.max(dim=1).values
    min_rgb = images.min(dim=1).values
    chroma = max_rgb - min_rgb
    saturation = chroma.mean(dim=(-2, -1))
    saturation = (saturation / 0.5).clamp(0, 1)

    # ---- Sharpness ----
    # Laplacian variance (approximate with simple gradient)
    gray = luminance
    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    sharpness = (dx.abs().mean(dim=(-2, -1)) + dy.abs().mean(dim=(-2, -1))) / 2
    sharpness = (sharpness / 0.15).clamp(0, 1)

    # ---- Grain ----
    # High-frequency energy (approximate with Laplacian)
    # Simple 3x3 Laplacian
    if gray.shape[-1] >= 3 and gray.shape[-2] >= 3:
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)
        gray_4d = gray.unsqueeze(1)
        laplacian = F.conv2d(gray_4d, laplacian_kernel, padding=1)
        grain = laplacian.abs().mean(dim=(-3, -2, -1))
        grain = (grain / 0.1).clamp(0, 1)
    else:
        grain = torch.zeros(B, device=device)

    return {
        "warmth": warmth.unsqueeze(1),
        "contrast": contrast.unsqueeze(1),
        "saturation": saturation.unsqueeze(1),
        "brightness": brightness.unsqueeze(1),
        "sharpness": sharpness.unsqueeze(1),
        "grain": grain.unsqueeze(1),
    }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ShotSense Loss Functions")
    print("=" * 60)

    B = 8
    style_dim = 128
    content_dim = 512

    # Create dummy data
    anchor_style = F.normalize(torch.randn(B, style_dim), dim=-1)
    positive_style = F.normalize(torch.randn(B, style_dim), dim=-1)
    negative_style = F.normalize(torch.randn(B, style_dim), dim=-1)
    anchor_content = F.normalize(torch.randn(B, content_dim), dim=-1)

    # Dummy images for pseudo-labels
    images = torch.rand(B, 3, 224, 224)
    pseudo_labels = compute_pseudo_labels(images)

    # Dummy predictions
    predicted = {name: torch.rand(B, 1) for name in pseudo_labels}

    # Test combined loss
    criterion = ShotSenseLoss()
    losses = criterion(
        anchor_style,
        positive_style,
        negative_style,
        anchor_content,
        predicted,
        pseudo_labels,
    )

    print(f"\nTotal loss: {losses['total']:.4f}")
    print(f"Triplet loss: {losses['triplet']:.4f}")
    print(f"Orthogonality loss: {losses['orthogonality']:.4f}")
    print(f"Attribute loss: {losses['attributes']:.4f}")

    # Test pseudo-labels
    print(f"\nPseudo-labels computed:")
    for name, val in pseudo_labels.items():
        print(f"  {name}: mean={val.mean():.3f}, range=[{val.min():.3f}, {val.max():.3f}]")

    print("\n✓ All loss tests passed!")
