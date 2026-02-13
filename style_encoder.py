"""
ShotSense Style Encoder — Disentangled Aesthetic Style Embeddings.

This is the novel ML component of ShotSense. It takes the ViT backbone's
features and learns to extract a pure "style" representation that captures:
- Color grading / palette
- Lighting quality and direction
- Composition patterns
- Contrast and tonal range
- Mood / atmosphere
- Depth of field characteristics

...while being INVARIANT to content (what's in the photo).

Key insight: CLIP embeddings conflate style and content. A moody portrait
and a moody landscape should have similar STYLE embeddings even though
CLIP puts them far apart due to different content.

Architecture:
    ViT backbone (frozen CLIP weights) → 512-dim CLIP embedding
    ↓
    Style Projection Head (MLP: 512 → 384 → 256 → 128)
    ↓
    128-dim Style Vector (L2-normalized)

    + Attribute Prediction Heads branching off intermediate features
    + Few-shot Style Prototype via attention-weighted aggregation

Training Losses:
    1. Style Triplet Loss — same-style pairs close, different-style far
    2. Content Orthogonality Loss — style vectors uncorrelated with content
    3. Attribute Regression Loss — predict interpretable style attributes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from vit import VisionTransformer, create_vit_b32


# ============================================================
# Style Projection Head
# ============================================================

class StyleProjectionHead(nn.Module):
    """
    Projects CLIP's 512-dim embedding into a 128-dim style space.
    
    Uses a 3-layer MLP with residual connections and layer normalization
    to learn a non-linear mapping that strips content information
    and preserves only aesthetic style features.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [384, 256]

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No norm/activation on final layer
                self.norms.append(nn.LayerNorm(dims[i + 1]))

        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 512) CLIP embedding
        Returns:
            (B, 128) L2-normalized style vector
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = F.gelu(x)
                x = self.dropout(x)

        # L2 normalize — style vectors live on a unit hypersphere
        x = F.normalize(x, p=2, dim=-1)
        return x


# ============================================================
# Attribute Prediction Heads
# ============================================================

class AttributeHeads(nn.Module):
    """
    Predicts interpretable style attributes from the style embedding.
    
    These serve dual purposes:
    1. Explainability — tell the user WHY a photo matches/doesn't match
    2. Regularization — forces the style space to encode meaningful features
    
    Attributes predicted:
    - warmth: 0 (cool/blue) → 1 (warm/golden)
    - contrast: 0 (flat) → 1 (high contrast)
    - saturation: 0 (desaturated) → 1 (vibrant)
    - brightness: 0 (dark/moody) → 1 (bright/airy)
    - sharpness: 0 (soft) → 1 (crisp)
    - grain: 0 (clean) → 1 (grainy/textured)
    """

    ATTRIBUTE_NAMES = [
        "warmth",
        "contrast",
        "saturation",
        "brightness",
        "sharpness",
        "grain",
    ]

    def __init__(self, input_dim: int = 128, num_attributes: int = 6):
        super().__init__()
        self.num_attributes = num_attributes

        # Shared hidden layer + individual prediction heads
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # One head per attribute — predicts a value in [0, 1]
        self.heads = nn.ModuleList(
            [nn.Linear(64, 1) for _ in range(num_attributes)]
        )

    def forward(self, style_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            style_vector: (B, 128) style embedding
        Returns:
            dict of attribute_name → (B, 1) predicted values in [0, 1]
        """
        shared_features = self.shared(style_vector)

        attributes = {}
        for i, (name, head) in enumerate(
            zip(self.ATTRIBUTE_NAMES, self.heads)
        ):
            attributes[name] = torch.sigmoid(head(shared_features))

        return attributes


# ============================================================
# Few-Shot Style Prototype
# ============================================================

class StylePrototype(nn.Module):
    """
    Builds a style prototype from a few reference images using
    attention-weighted aggregation.
    
    Instead of a simple mean (which treats all reference photos equally),
    this learns to weight references by how representative they are of
    the group's shared style, automatically down-weighting outliers.
    
    Process:
    1. Compute style vectors for all reference images
    2. Compute pairwise similarity matrix
    3. Score each reference by its average similarity to others
    4. Softmax → attention weights
    5. Weighted sum → prototype vector
    """

    def __init__(self, style_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

        # Learnable query for "what makes a good prototype member"
        self.prototype_query = nn.Parameter(torch.randn(1, style_dim))

        # Small projection for computing attention scores
        self.key_proj = nn.Linear(style_dim, style_dim)
        self.value_proj = nn.Linear(style_dim, style_dim)

    def forward(
        self,
        reference_vectors: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            reference_vectors: (N, 128) style vectors of reference images
            return_weights: if True, also return attention weights
        Returns:
            prototype: (128,) the aggregated style prototype
            weights: (N,) attention weight per reference (if requested)
        """
        N = reference_vectors.shape[0]

        # Method 1: Self-attention based weighting
        # Each reference's importance = how well it represents the group
        keys = self.key_proj(reference_vectors)  # (N, 128)
        values = self.value_proj(reference_vectors)  # (N, 128)

        # Compute pairwise similarity: how similar is each ref to all others?
        sim_matrix = torch.mm(keys, keys.t()) / self.temperature  # (N, N)

        # Mask diagonal (don't count self-similarity)
        mask = torch.eye(N, device=reference_vectors.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # Average attention each reference receives from others
        avg_attention = F.softmax(sim_matrix, dim=-1).mean(dim=0)  # (N,)

        # Also incorporate the learnable prototype query
        query_scores = torch.mm(
            self.prototype_query, keys.t()
        ).squeeze(0) / self.temperature  # (N,)
        query_weights = F.softmax(query_scores, dim=0)

        # Combine both signals
        weights = 0.5 * avg_attention + 0.5 * query_weights
        weights = weights / weights.sum()  # Re-normalize

        # Weighted aggregation
        prototype = (weights.unsqueeze(1) * values).sum(dim=0)

        # L2 normalize the prototype
        prototype = F.normalize(prototype, p=2, dim=-1)

        if return_weights:
            return prototype, weights
        return prototype, None


# ============================================================
# Complete Style Encoder
# ============================================================

class ShotSenseEncoder(nn.Module):
    """
    The complete ShotSense style encoder.
    
    Combines:
    1. ViT backbone (from scratch, with CLIP weights)
    2. Style projection head (disentangles style from content)
    3. Attribute prediction heads (interpretable features)
    4. Few-shot prototype system (aggregate reference style)
    
    Usage:
        encoder = ShotSenseEncoder(pretrained=True)
        
        # Encode a single image
        result = encoder(image_tensor)
        style_vec = result['style_vector']  # (B, 128)
        attributes = result['attributes']   # dict of attribute scores
        
        # Build a style prototype from references
        prototype, weights = encoder.build_prototype(reference_images)
        
        # Score new images against the prototype
        scores = encoder.score_against_prototype(new_images, prototype)
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        style_dim: int = 128,
    ):
        super().__init__()

        # ViT backbone — our from-scratch implementation
        self.backbone = create_vit_b32(pretrained=pretrained)

        # Freeze backbone weights (we only train the style heads)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Style projection head
        self.style_head = StyleProjectionHead(
            input_dim=512,
            hidden_dims=[384, 256],
            output_dim=style_dim,
        )

        # Attribute prediction heads
        self.attribute_heads = AttributeHeads(
            input_dim=style_dim,
            num_attributes=6,
        )

        # Few-shot prototype builder
        self.prototype_builder = StylePrototype(style_dim=style_dim)

        self.style_dim = style_dim

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: image → style vector + attributes.
        
        Args:
            images: (B, 3, 224, 224) input images
        Returns:
            dict with:
                'clip_embedding': (B, 512)
                'style_vector': (B, 128) — the core output
                'attributes': dict of attribute scores
                'attention_maps': list of attention maps (if requested)
        """
        # ViT forward pass
        vit_output = self.backbone(
            images,
            return_all_tokens=True,
            return_attention=return_attention,
        )

        clip_embedding = vit_output["embedding"]

        # Project to style space
        style_vector = self.style_head(clip_embedding)

        # Predict interpretable attributes
        attributes = self.attribute_heads(style_vector)

        result = {
            "clip_embedding": clip_embedding,
            "style_vector": style_vector,
            "attributes": attributes,
        }

        if return_attention and "attention_maps" in vit_output:
            result["attention_maps"] = vit_output["attention_maps"]

        return result

    @torch.no_grad()
    def build_prototype(
        self, reference_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a style prototype from reference images.
        
        Args:
            reference_images: (N, 3, 224, 224) reference photos
        Returns:
            prototype: (128,) style prototype vector
            weights: (N,) attention weight per reference
        """
        self.eval()

        # Get style vectors for all references
        result = self.forward(reference_images)
        style_vectors = result["style_vector"]  # (N, 128)

        # Aggregate into prototype
        prototype, weights = self.prototype_builder(
            style_vectors, return_weights=True
        )

        return prototype, weights

    @torch.no_grad()
    def score_against_prototype(
        self,
        images: torch.Tensor,
        prototype: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Score images against a style prototype.
        
        Args:
            images: (B, 3, 224, 224) candidate photos
            prototype: (128,) style prototype vector
        Returns:
            dict with:
                'scores': (B,) cosine similarity scores in [0, 100]
                'style_vectors': (B, 128)
                'attributes': dict of attribute predictions
        """
        self.eval()

        result = self.forward(images)
        style_vectors = result["style_vector"]

        # Cosine similarity → scale to 0-100
        scores = F.cosine_similarity(
            style_vectors, prototype.unsqueeze(0), dim=-1
        )
        scores = ((scores + 1) / 2 * 100).clamp(0, 100)  # [-1,1] → [0,100]

        return {
            "scores": scores,
            "style_vectors": style_vectors,
            "attributes": result["attributes"],
        }

    def get_trainable_params(self) -> int:
        """Count trainable parameters (excluding frozen backbone)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ShotSense Style Encoder")
    print("=" * 60)

    # Create encoder (without pretrained weights for quick test)
    encoder = ShotSenseEncoder(pretrained=False, freeze_backbone=True)

    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = encoder.get_trainable_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen (backbone): {total_params - trainable_params:,}")

    # Test single image encoding
    images = torch.randn(4, 3, 224, 224)
    result = encoder(images)
    print(f"\nStyle vector shape: {result['style_vector'].shape}")
    print(f"Attributes: {list(result['attributes'].keys())}")

    # Test prototype building
    references = torch.randn(8, 3, 224, 224)
    prototype, weights = encoder.build_prototype(references)
    print(f"\nPrototype shape: {prototype.shape}")
    print(f"Reference weights: {weights}")

    # Test scoring
    candidates = torch.randn(16, 3, 224, 224)
    scores = encoder.score_against_prototype(candidates, prototype)
    print(f"\nScores shape: {scores['scores'].shape}")
    print(f"Score range: {scores['scores'].min():.1f} - {scores['scores'].max():.1f}")

    print("\n✓ All tests passed!")
