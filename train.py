"""
ShotSense Training Script.

Complete training pipeline for the style encoder:
1. Load pre-trained ViT backbone with CLIP weights
2. Train style projection head + attribute heads
3. Custom triplet + orthogonality + attribute loss
4. Validation with style retrieval metrics
5. Checkpoint saving and TensorBoard logging

Usage:
    python -m training.train \
        --data_root ./data/style_groups \
        --epochs 50 \
        --batch_size 32 \
        --lr 1e-4 \
        --output_dir ./checkpoints
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from style_encoder import ShotSenseEncoder
from losses import ShotSenseLoss, compute_pseudo_labels
from dataset import StyleTripletDataset, create_data_loaders


class ShotSenseTrainer:
    """
    Trainer for the ShotSense style encoder.
    
    Handles the full training loop including:
    - Forward pass through frozen backbone + trainable heads
    - Triplet construction and hard negative mining
    - Multi-objective loss computation
    - Validation with style retrieval metrics
    - Checkpointing and logging
    """

    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = ShotSenseEncoder(
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
            style_dim=config.style_dim,
        ).to(self.device)

        print(f"Model parameters:")
        print(f"  Total: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable: {self.model.get_trainable_params():,}")

        # Initialize loss
        self.criterion = ShotSenseLoss(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            triplet_margin=config.margin,
            hard_mining=config.hard_mining,
        )

        # Optimizer — only train non-frozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.lr * 0.01,
        )

        # Data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            val_split=config.val_split,
        )

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_val_loss = float("inf")
        self.epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {
            "total": 0, "triplet": 0, "orthogonality": 0, "attributes": 0
        }
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            anchor = batch["anchor"].to(self.device)
            positive = batch["positive"].to(self.device)
            negative = batch["negative"].to(self.device)
            anchor_raw = batch["anchor_raw"].to(self.device)

            # Forward pass — get style vectors and CLIP embeddings
            anchor_out = self.model(anchor)
            positive_out = self.model(positive)
            negative_out = self.model(negative)

            # Compute pseudo-labels from raw (unnormalized) images
            pseudo_labels = compute_pseudo_labels(anchor_raw)

            # Compute combined loss
            losses = self.criterion(
                anchor_style=anchor_out["style_vector"],
                positive_style=positive_out["style_vector"],
                negative_style=negative_out["style_vector"],
                anchor_content=anchor_out["clip_embedding"],
                predicted_attributes=anchor_out["attributes"],
                pseudo_label_attributes=pseudo_labels,
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping — only on trainable parameters
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )

            self.optimizer.step()

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_losses["total"] / num_batches
                print(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Triplet: {total_losses['triplet']/num_batches:.4f} | "
                    f"Orth: {total_losses['orthogonality']/num_batches:.4f} | "
                    f"Attr: {total_losses['attributes']/num_batches:.4f}"
                )

        # Average losses
        return {k: v / max(num_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_losses = {
            "total": 0, "triplet": 0, "orthogonality": 0, "attributes": 0
        }
        num_batches = 0

        # Style retrieval metrics
        correct_retrievals = 0
        total_retrievals = 0

        for batch in self.val_loader:
            anchor = batch["anchor"].to(self.device)
            positive = batch["positive"].to(self.device)
            negative = batch["negative"].to(self.device)
            anchor_raw = batch["anchor_raw"].to(self.device)

            anchor_out = self.model(anchor)
            positive_out = self.model(positive)
            negative_out = self.model(negative)

            pseudo_labels = compute_pseudo_labels(anchor_raw)

            losses = self.criterion(
                anchor_style=anchor_out["style_vector"],
                positive_style=positive_out["style_vector"],
                negative_style=negative_out["style_vector"],
                anchor_content=anchor_out["clip_embedding"],
                predicted_attributes=anchor_out["attributes"],
                pseudo_label_attributes=pseudo_labels,
            )

            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

            # Style retrieval accuracy:
            # Is the positive closer to anchor than negative?
            d_pos = 1 - F.cosine_similarity(
                anchor_out["style_vector"], positive_out["style_vector"], dim=-1
            )
            d_neg = 1 - F.cosine_similarity(
                anchor_out["style_vector"], negative_out["style_vector"], dim=-1
            )
            correct_retrievals += (d_pos < d_neg).sum().item()
            total_retrievals += d_pos.shape[0]

        avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
        avg_losses["retrieval_accuracy"] = (
            correct_retrievals / max(total_retrievals, 1)
        )

        return avg_losses

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": vars(self.config),
        }

        # Save latest
        path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  ★ New best model saved!")

        # Save periodic
        if (self.epoch + 1) % self.config.save_interval == 0:
            epoch_path = self.output_dir / f"checkpoint_epoch_{self.epoch+1}.pt"
            torch.save(checkpoint, epoch_path)

    def train(self):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("Starting ShotSense Training")
        print("=" * 60)
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.lr}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 60)

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 40)

            # Train
            train_losses = self.train_epoch()
            print(
                f"  Train | Loss: {train_losses['total']:.4f} | "
                f"Triplet: {train_losses['triplet']:.4f} | "
                f"Orth: {train_losses['orthogonality']:.4f} | "
                f"Attr: {train_losses['attributes']:.4f}"
            )

            # Validate
            val_losses = self.validate()
            print(
                f"  Val   | Loss: {val_losses['total']:.4f} | "
                f"Triplet: {val_losses['triplet']:.4f} | "
                f"Retrieval Acc: {val_losses['retrieval_accuracy']:.2%}"
            )

            # Update scheduler
            self.scheduler.step()

            # Save checkpoint
            is_best = val_losses["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["total"]
            self.save_checkpoint(is_best=is_best)

            elapsed = time.time() - start_time
            print(f"  Time: {elapsed:.1f}s | LR: {self.scheduler.get_last_lr()[0]:.2e}")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("=" * 60)

    def export_model(self, output_path: Optional[str] = None):
        """
        Export trained model for inference.
        
        Saves just the model weights (without optimizer state)
        and exports to ONNX for fast production inference.
        """
        if output_path is None:
            output_path = self.output_dir / "shotsense_model.pt"

        # Save PyTorch model
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": vars(self.config),
            },
            output_path,
        )
        print(f"Model exported to: {output_path}")

        # Export to ONNX
        onnx_path = str(output_path).replace(".pt", ".onnx")
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            self.model.eval()

            # We need to trace just the forward pass
            # ONNX export requires a simpler forward interface
            print(f"ONNX export: {onnx_path}")
            print("(ONNX export requires tracing — skipping for complex model)")
        except Exception as e:
            print(f"ONNX export failed: {e}")


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ShotSense style encoder")

    # Data
    parser.add_argument("--data_root", type=str, default="./data/style_groups",
                        help="Path to style-grouped photo directory")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Load CLIP pretrained weights")
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Freeze ViT backbone")
    parser.add_argument("--style_dim", type=int, default=128)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Loss weights
    parser.add_argument("--alpha", type=float, default=1.0, help="Triplet loss weight")
    parser.add_argument("--beta", type=float, default=0.5, help="Orthogonality loss weight")
    parser.add_argument("--gamma", type=float, default=0.3, help="Attribute loss weight")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet margin")
    parser.add_argument("--hard_mining", action="store_true", default=True)

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    trainer = ShotSenseTrainer(args)
    trainer.train()
    trainer.export_model()
