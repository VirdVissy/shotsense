# ShotSense — AI Photo Curation & Style Communication

**AI-powered photo ranking and style communication tool for creative teams.**

Built from a real pain point: as Content Creation Director at BU Film & Radio, communicating visual style to a photography team and then culling hundreds of photos to find the ones that match is incredibly time-consuming. ShotSense solves both problems with a single system.

## What It Does

1. **Define your style** — Upload 5-10 reference photos that capture the vibe you want
2. **AI learns your aesthetic** — The system builds a mathematical "style prototype" that captures the shared aesthetic of your references (color grading, contrast, mood, composition) while ignoring content differences
3. **Rank incoming photos** — Upload a batch of candidate photos and instantly get them ranked by style match (0-100 score)
4. **Understand the gaps** — For any photo, see exactly WHY it matches or doesn't: "This photo matches on warmth and composition but has too much contrast compared to your reference style"
5. **Share style guides** — Generate a shareable brief that communicates your target aesthetic to photographers before a shoot

## Architecture

### Vision Transformer — Built From Scratch

The backbone is a complete **ViT-B/32** implementation, hand-coded in PyTorch — every component from patch embedding to multi-head self-attention to positional encoding. Pre-trained CLIP weights are loaded into our custom architecture for transfer learning.

```
Image (B, 3, 224, 224)
  → PatchEmbedding: split into 7×7 = 49 patches of 32×32, linear projection to 768-dim
  → 12× TransformerBlock: LayerNorm → MultiHeadSelfAttention (12 heads) → MLP (expand 4×)
  → LayerNorm → Extract [CLS] token → Linear projection to 512-dim
```

### Style-Content Disentanglement — The Novel Component

The key insight: CLIP's embedding space conflates **what's in the photo** with **how it looks**. Two photos with the same moody aesthetic but different subjects (portrait vs. landscape) are far apart in CLIP space. We fix this.

**Style Projection Head**: MLP (512 → 384 → 256 → 128) trained with a custom 3-part loss:

| Loss | Purpose | How It Works |
|------|---------|-------------|
| **Style Triplet Loss** | Same-style images close, different-style far | Hard negative mining within batch; margin = 0.3 |
| **Content Orthogonality Loss** | Style vectors uncorrelated with content | Penalizes Frobenius norm of cross-correlation matrix between style and CLIP content embeddings |
| **Attribute Regression Loss** | Style space encodes interpretable features | Supervises 6 attribute heads (warmth, contrast, saturation, brightness, sharpness, grain) with pseudo-labels from image statistics |

### Few-Shot Style Prototype

From just 5-10 reference photos, builds a style prototype using **attention-weighted aggregation** — a learned mechanism that weights each reference by how representative it is of the group's shared style, automatically down-weighting outliers.

### Explainability — Grad-CAM for Style

Adapted Gradient-weighted Class Activation Mapping for style attribution. Visualizes which image regions most influence the style score, giving photographers actionable feedback.

## Project Structure

```
shotsense/
├── models/
│   ├── vit.py              # ViT-B/32 from scratch (PatchEmbed, MHSA, TransformerBlock)
│   ├── style_encoder.py    # Style projection head, attribute heads, prototype builder
│   └── losses.py           # Triplet + orthogonality + attribute losses
├── data/
│   └── dataset.py          # StyleTripletDataset, data loaders, augmentation
├── training/
│   └── train.py            # Full training loop with validation and checkpointing
├── api/
│   └── server.py           # FastAPI REST API
├── utils/
│   └── explainability.py   # Grad-CAM style attribution
└── requirements.txt
```

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
# Organize photos into style groups (subdirectories = same style)
# data/style_groups/photographer_alice/*.jpg
# data/style_groups/photographer_bob/*.jpg
# data/style_groups/preset_film/*.jpg
# ...

python -m training.train \
    --data_root ./data/style_groups \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4
```

### Run API
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Use
```bash
# Create a project
curl -X POST http://localhost:8000/api/projects \
    -H "Content-Type: application/json" \
    -d '{"name": "BU Film Spring Shoot"}'

# Upload reference photos (define your style)
curl -X POST http://localhost:8000/api/projects/{id}/references \
    -F "files=@ref1.jpg" -F "files=@ref2.jpg" -F "files=@ref3.jpg"

# Upload candidates and get ranked results
curl -X POST http://localhost:8000/api/projects/{id}/upload \
    -F "files=@photo1.jpg" -F "files=@photo2.jpg" -F "files=@photo3.jpg"

# Deep analysis of a single photo
curl -X POST http://localhost:8000/api/projects/{id}/analyze \
    -F "file=@photo1.jpg"

# Get shareable style guide
curl http://localhost:8000/api/projects/{id}/styleguide
```

## Technical Highlights

- **From-scratch ViT**: Every layer hand-coded in PyTorch, compatible with CLIP weight loading
- **Novel loss function**: Triplet + orthogonality + attribute regression for style-content disentanglement
- **Hard negative mining**: Semi-hard negatives within batch for more effective contrastive learning
- **Few-shot learning**: Accurate style prototypes from just 5-10 reference images
- **Interpretable AI**: Attribute-level explanations + Grad-CAM spatial attribution
- **Production-ready API**: FastAPI with async file handling, project management

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch |
| Vision Backbone | ViT-B/32 (from scratch) + CLIP weights |
| Training | Custom contrastive learning pipeline |
| Backend | FastAPI + Uvicorn |
| Explainability | Grad-CAM (custom implementation) |
| Metric Learning | Triplet loss with hard negative mining |

## Model Parameters

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| ViT Backbone | ~87M | ✗ (frozen) |
| Style Head | ~180K | ✓ |
| Attribute Heads | ~50K | ✓ |
| Prototype Builder | ~33K | ✓ |
| **Total Trainable** | **~263K** | |
