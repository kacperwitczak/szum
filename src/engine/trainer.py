import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class Trainer:
    """
    SOTA Training Engine featuring:
    - Automatic Mixed Precision (AMP)
    - Gradient Clipping
    - Dynamic learning rate interactions
    - Early Stopping & Best Checkpoint Tracking
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        device: torch.device = None,
        patience: int = 10,
        output_dir: str | Path = "checkpoints",
        config: dict = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        
        self.patience = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        self.config = config or {}
        self._save_config()
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }

    def fit(self, num_epochs: int, unfreeze_backbone_epoch: int = None) -> None:
        print(f"Starting training on {self.device}...")
        for epoch in range(num_epochs):
            # ── Unfreeze backbone scheduling ──
            if unfreeze_backbone_epoch is not None and epoch == unfreeze_backbone_epoch:
                print(f"--> Unfreezing backbone at epoch {epoch}")
                # Assuming model supports `unfreeze_encoder()` or `backbone.unfreeze()`
                if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "unfreeze"):
                    self.model.backbone.unfreeze()
                    # Add newly unfrozen parameters to optimizer with a lower LR
                    self.optimizer.add_param_group({
                        'params': filter(lambda p: p.requires_grad, self.model.backbone.parameters()),
                        'lr': self.optimizer.param_groups[0]['lr'] * 0.1
                    })

            # ── Train ──
            train_loss = self._train_one_epoch(epoch, num_epochs)
            
            # ── Validate ──
            val_loss, val_metrics = self._validate(epoch, num_epochs)
            val_acc = val_metrics["acc"]
            val_f1 = val_metrics["f1"]
            val_prec = val_metrics["precision"]
            val_rec = val_metrics["recall"]
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["val_precision"].append(val_prec)
            self.history["val_recall"].append(val_rec)
            
            # Save history to JSON
            self._save_history()
            
            # ── LR Scheduler step ──
            if self.scheduler is not None:
                self.scheduler.step()
                
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | LR: {current_lr:.2e}")
            
            # ── Early Stopping & Checkpointing ──
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt")
                print(f"  [*] Best model saved! (Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
            else:
                self.epochs_without_improvement += 1
                if self.patience > 0 and self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training completed. Best Val Acc: {self.best_val_acc:.4f}")
        self._plot_history()

    def _train_one_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
        for batch in pbar:
            videos = batch["video"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(videos)
                loss = self.criterion(logits, labels)
                
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self, epoch: int, total_epochs: int) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
        for batch in pbar:
            videos = batch["video"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(videos)
                loss = self.criterion(logits, labels)
                
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        loss_val = total_loss / len(self.val_loader)
        
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        
        metrics = {
            "acc": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        
        return loss_val, metrics

    def _save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_val_acc": self.best_val_acc,
        }, path)

    def _plot_history(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history["val_acc"], label="Val Acc")
        plt.title("Accuracy over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        # Other metrics plot
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.history["val_f1"], label="Val F1")
        plt.plot(epochs, self.history["val_precision"], label="Val Precision")
        plt.plot(epochs, self.history["val_recall"], label="Val Recall")
        plt.title("Macro Metrics over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_history.png")
        plt.close()
        
    def _save_history(self):
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=4)

    def _save_config(self):
        if self.config:
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(self.config, f, indent=4)
