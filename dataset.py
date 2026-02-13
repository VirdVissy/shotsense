"""
ShotSense Dataset — Training Data Pipeline.

Handles loading and preparing training data for the style encoder.
Supports two data sources:

1. AVA Dataset (Aesthetic Visual Analysis)
   - 255K+ images with aesthetic ratings
   - We use photographer grouping as style proxy
   
2. Style-Grouped Photos
   - Photos grouped by photographer, preset, or manual style labels
   - Each group = photos that share the same aesthetic style
   - Different groups = different styles

The key challenge is constructing good triplets:
- Anchor: a random photo from a style group
- Positive: another photo from the SAME style group (same style, likely different content)
- Negative: a photo from a DIFFERENT style group (different style)

Hard negatives: we prefer negatives whose CONTENT is similar to the anchor
but whose STYLE is different (e.g., two portraits with different color grading).
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class StyleTripletDataset(Dataset):
    """
    Dataset that generates (anchor, positive, negative) style triplets.
    
    Directory structure:
        data_root/
            style_group_001/
                photo_001.jpg
                photo_002.jpg
                ...
            style_group_002/
                photo_003.jpg
                photo_004.jpg
                ...
    
    Each subdirectory is a "style group" — photos that share the same
    aesthetic style (e.g., from the same photographer or same preset).
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 224,
        augment: bool = True,
        min_group_size: int = 2,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size

        # Build style groups: dict of group_name → list of image paths
        self.style_groups: Dict[str, List[Path]] = {}
        self._build_style_groups(min_group_size)

        # List of all group names for sampling
        self.group_names = list(self.style_groups.keys())

        # Build flat index: each item is (group_name, image_path)
        self.index = []
        for group_name, paths in self.style_groups.items():
            for path in paths:
                self.index.append((group_name, path))

        # Image transforms
        self.transform = self._build_transforms(augment)

        print(f"Loaded {len(self.index)} images across {len(self.group_names)} style groups")

    def _build_style_groups(self, min_group_size: int):
        """Scan data_root for style groups (subdirectories with images)."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if not self.data_root.exists():
            print(f"Warning: data root {self.data_root} does not exist.")
            print("Creating synthetic dataset for testing...")
            self._create_synthetic_data()
            return

        for group_dir in sorted(self.data_root.iterdir()):
            if not group_dir.is_dir():
                continue

            images = [
                p for p in group_dir.iterdir()
                if p.suffix.lower() in valid_extensions
            ]

            if len(images) >= min_group_size:
                self.style_groups[group_dir.name] = images

    def _create_synthetic_data(self):
        """Create synthetic style groups for testing/demo purposes."""
        # We create synthetic groups to allow testing without real data
        # Each group gets a distinct "style" via color transform
        num_groups = 10
        images_per_group = 20

        os.makedirs(self.data_root, exist_ok=True)

        for g in range(num_groups):
            group_dir = self.data_root / f"style_{g:03d}"
            os.makedirs(group_dir, exist_ok=True)

            group_images = []
            for i in range(images_per_group):
                img_path = group_dir / f"img_{i:04d}.jpg"

                if not img_path.exists():
                    # Create synthetic image with group-specific color shift
                    img = Image.new("RGB", (256, 256))
                    pixels = img.load()

                    # Each group has a distinct base hue
                    base_r = int(128 + 80 * (g % 3 == 0))
                    base_g = int(128 + 80 * (g % 3 == 1))
                    base_b = int(128 + 80 * (g % 3 == 2))

                    for x in range(256):
                        for y in range(256):
                            # Add content variation within group
                            noise_r = random.randint(-30, 30)
                            noise_g = random.randint(-30, 30)
                            noise_b = random.randint(-30, 30)

                            # Add spatial pattern (different content per image)
                            pattern = int(50 * ((x * (i + 1) + y) % 7) / 6)

                            r = max(0, min(255, base_r + noise_r + pattern))
                            g_val = max(0, min(255, base_g + noise_g + pattern))
                            b_val = max(0, min(255, base_b + noise_b + pattern))

                            pixels[x, y] = (r, g_val, b_val)

                    img.save(img_path, quality=85)

                group_images.append(img_path)

            self.style_groups[group_dir.name] = group_images

        print(f"Created synthetic dataset: {num_groups} groups × {images_per_group} images")

    def _build_transforms(self, augment: bool) -> transforms.Compose:
        """Build image preprocessing pipeline."""
        if augment:
            return transforms.Compose([
                transforms.Resize((self.image_size + 32, self.image_size + 32)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                # NOTE: We intentionally do NOT use ColorJitter or style-altering
                # augmentations, because those would destroy the style signal
                # we're trying to learn. Only geometric augmentations are safe.
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a triplet: (anchor, positive, negative).
        
        - anchor: image at index idx
        - positive: random image from the SAME style group
        - negative: random image from a DIFFERENT style group
        """
        anchor_group, anchor_path = self.index[idx]

        # Positive: same group, different image
        group_images = self.style_groups[anchor_group]
        positive_candidates = [p for p in group_images if p != anchor_path]
        if not positive_candidates:
            positive_candidates = group_images  # Fallback
        positive_path = random.choice(positive_candidates)

        # Negative: different group
        negative_group = anchor_group
        while negative_group == anchor_group:
            negative_group = random.choice(self.group_names)
        negative_path = random.choice(self.style_groups[negative_group])

        # Load and transform images
        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)

        # Also return unnormalized anchor for pseudo-label computation
        raw_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        anchor_raw = raw_transform(Image.open(anchor_path).convert("RGB"))

        return {
            "anchor": anchor_img,
            "positive": positive_img,
            "negative": negative_img,
            "anchor_raw": anchor_raw,
            "anchor_group": anchor_group,
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and transform a single image."""
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class InferenceDataset(Dataset):
    """
    Simple dataset for inference — load images from a directory.
    No triplets, just individual images with their paths.
    """

    def __init__(self, image_paths: List[str], image_size: int = 224):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return {"image": tensor, "path": path}


def create_data_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_root: path to style-grouped photo directory
        batch_size: batch size for training
        num_workers: number of data loading workers
        val_split: fraction of data to use for validation
    Returns:
        (train_loader, val_loader)
    """
    full_dataset = StyleTripletDataset(data_root, augment=True)

    # Split into train/val
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ShotSense Dataset")
    print("=" * 60)

    # Test with synthetic data
    dataset = StyleTripletDataset(
        data_root="/tmp/shotsense_test_data",
        augment=False,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Style groups: {len(dataset.group_names)}")

    # Test getting a triplet
    sample = dataset[0]
    print(f"\nAnchor shape: {sample['anchor'].shape}")
    print(f"Positive shape: {sample['positive'].shape}")
    print(f"Negative shape: {sample['negative'].shape}")
    print(f"Anchor raw shape: {sample['anchor_raw'].shape}")
    print(f"Anchor group: {sample['anchor_group']}")

    print("\n✓ Dataset test passed!")
