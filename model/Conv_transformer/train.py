"""
Training script for Conv2DTransformer.

Usage:
    python train.py \
        --csv merged_datasets/split8c_frames_signer_disjoint.csv \
        --batch-size 8 \
        --num-epochs 30 \
        --lr 1e-4 \
        --output-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Conv_transformer.model_2d_conv_transformer import Conv2DTransformer
from model.torch_sign_frames_dataset import SignFramesDataset


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    videos = torch.stack([x["video"] for x in batch], dim=0)
    labels = torch.stack([x["label"] for x in batch], dim=0)
    return {"video": videos, "label": labels}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=True)

    for batch in pbar:
        videos = batch["video"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(videos)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> tuple[float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=True)

    with torch.no_grad():
        for batch in pbar:
            videos = batch["video"].to(device)
            labels = batch["label"].to(device)

            logits = model(videos)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.shape[0]

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conv2DTransformer")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to split CSV.",
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Defaults
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.csv is None:
        args.csv = str(PROJECT_ROOT / "merged_datasets" / "split8c_frames_signer_disjoint.csv")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Conv2DTransformer Training")
    print("=" * 70)
    print(f"CSV: {csv_path}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Sequence length: {args.sequence_length}")
    print("=" * 70 + "\n")

    # Load datasets
    print("Loading datasets...")
    train_ds = SignFramesDataset(
        csv_path=csv_path,
        project_root=PROJECT_ROOT,
        split="train",
        mode="train",
        sequence_length=args.sequence_length,
        base_seed=args.seed,
    )

    val_ds = SignFramesDataset(
        csv_path=csv_path,
        project_root=PROJECT_ROOT,
        split="val",
        mode="val",
        sequence_length=args.sequence_length,
        label_to_index=train_ds.label_to_index,
        base_seed=args.seed,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Num classes: {len(train_ds.label_to_index)}\n")

    # Create data loaders
    sampler = train_ds.build_weighted_sampler(power=0.5, replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_batch,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_batch,
        drop_last=False,
    )

    # Create model
    print("Creating model...")
    model = Conv2DTransformer(num_classes=len(train_ds.label_to_index), dropout_rate=0.3)
    model = model.to(device)

    total_params, trainable_params = model.get_num_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    print("=" * 70)
    print("Training")
    print("=" * 70 + "\n")

    for epoch in range(args.num_epochs):
        train_ds.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        scheduler.step()

        print(f"Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}\n")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            print(f"✓ Saved best checkpoint: {checkpoint_path}\n")

    print("=" * 70)
    print(f"Training complete!")
    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch+1}")
    print("=" * 70)


if __name__ == "__main__":
    main()
