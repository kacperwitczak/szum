import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.sign_model import SignModel
from src.data.dataset import SignFramesDataset
from src.engine.trainer import Trainer


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        if self.alpha is not None:
            ce = ce * self.alpha
        loss = (1.0 - pt) ** self.gamma * ce
        return loss.mean()

def parse_args():
    parser = argparse.ArgumentParser("Train Sign Language Model (SOTA)")
    parser.add_argument("--csv", type=str, required=True, help="Path to split frames CSV")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4, help="Dataloader workers")
    parser.add_argument("--epochs", type=int, default=30, help="Max number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--unfreeze-epoch", type=int, default=None, help="Epoch to unfreeze backbone")
    parser.add_argument(
        "--unfreeze-lr-mult",
        type=float,
        default=0.1,
        help="LR multiplier for newly unfrozen backbone parameters",
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for speed")
    parser.add_argument("--no-aug", action="store_true", help="Disable spatial augmentations")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Save directory")

    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "focal"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for cross entropy",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma value for focal loss",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Alpha value for focal loss (use null to disable)",
    )

    parser.add_argument(
        "--best-metric",
        type=str,
        default="acc",
        choices=["acc", "f1", "precision", "recall"],
        help="Metric used for best model selection",
    )
    
    parser.add_argument("--backbone", type=str, default="efficientnet", choices=["efficientnet", "dinov2", "dinov3"], help="Vision backbone")
    parser.add_argument("--temporal", type=str, default="transformer", choices=["transformer", "gru", "lstm"], help="Temporal aggregation model")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing Datasets...")
    train_aug_p = 0.0 if args.no_aug else 0.9
    train_dataset = SignFramesDataset(args.csv, split="train", sequence_length=32, spatial_aug_p=train_aug_p)
    val_dataset = SignFramesDataset(args.csv, split="val", sequence_length=32, spatial_aug_p=0.0)
    
    train_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 4 if args.num_workers > 0 else None,
    }
    
    val_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 2 if args.num_workers > 0 else None,
    }
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **train_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **val_kwargs
    )
    
    num_classes = len(train_dataset.label_to_idx)
    print(f"Classes: {num_classes} | Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    model = SignModel(
        num_classes=num_classes, 
        freeze_backbone=True,
        backbone_type=args.backbone,
        temporal_type=args.temporal,
        dropout_rate=args.dropout
    )
    
    if args.compile and device.type == "cuda":
        print("Compiling model via torch.compile() for optimal speed...")
        model = torch.compile(model)
        
    model.to(device)
    
    # Napisane z pomocą AI
    if hasattr(model, "get_num_parameters"):
        total_p, trainable_p = model.get_num_parameters()
    elif hasattr(model, "_orig_mod") and hasattr(model._orig_mod, "get_num_parameters"):
        total_p, trainable_p = model._orig_mod.get_num_parameters()
    else:
        total_p = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    if args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")

    print("=" * 60)
    print("Training Configuration:")
    print(f"  Device:             {device}")
    print(f"  Model Architecture: {args.backbone.upper()} + {args.temporal.upper()}")
    print(f"  Batch size:         {args.batch_size} (Val: {args.batch_size})")
    print(f"  Num Epochs:         {args.epochs}")
    print(f"  Learning Rate:      {args.lr}")
    print(f"  Unfreeze Epoch:     {args.unfreeze_epoch}")
    print(f"  Unfreeze LR Mult:   {args.unfreeze_lr_mult}")
    print(f"  Loss:               {args.loss}")
    print(f"  Best Metric:        {args.best_metric}")
    print(f"  Total Params:       {total_p:,}")
    print(f"  Trainable Params:   {trainable_p:,}")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = Path(args.csv).stem
    run_output_dir = os.path.join(args.output_dir, f"{csv_name}_{args.backbone}_{args.temporal}_{timestamp}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=args.patience,
        best_metric=args.best_metric,
        unfreeze_lr_mult=args.unfreeze_lr_mult,
        output_dir=run_output_dir,
        config=vars(args)
    )

    trainer.fit(num_epochs=args.epochs, unfreeze_backbone_epoch=args.unfreeze_epoch)

if __name__ == "__main__":
    main()
