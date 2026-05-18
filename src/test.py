import argparse
import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix

from src.models.sign_model import SignModel
from src.data.dataset import SignFramesDataset

def parse_args():
    parser = argparse.ArgumentParser("Test Sign Language Model")
    parser.add_argument("--csv", type=str, required=True, help="Path to split frames CSV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "test_unseen"], help="Dataset split to test on")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4, help="Dataloader workers")
    
    parser.add_argument("--backbone", type=str, default="efficientnet", choices=["efficientnet", "dinov2", "dinov3"], help="Vision backbone")
    parser.add_argument("--temporal", type=str, default="transformer", choices=["transformer", "gru", "lstm"], help="Temporal aggregation model")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset & Dataloader
    print(f"Initializing {args.split.upper()} Dataset...")
    test_dataset = SignFramesDataset(args.csv, split=args.split, sequence_length=32)
    
    test_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 2 if args.num_workers > 0 else None,
    }
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **test_kwargs
    )
    
    num_classes = len(test_dataset.label_to_idx)
    print(f"Classes: {num_classes} | Test size: {len(test_dataset)}")
    
    if len(test_dataset) == 0:
        print("Error: Test dataset is empty! Check if the 'test' split exists in your CSV.")
        return

    # 2. Model setup
    print(f"Initializing Model: {args.backbone.upper()} + {args.temporal.upper()}")
    model = SignModel(
        num_classes=num_classes, 
        freeze_backbone=True,
        backbone_type=args.backbone,
        temporal_type=args.temporal
    )
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Starting evaluation...")
    all_preds = []
    all_labels = []
    
    if device.type == "cuda":
        autocast_context = torch.amp.autocast(device_type="cuda")
    else:
        from contextlib import nullcontext
        autocast_context = nullcontext()
        
    pbar = tqdm(test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for batch in pbar:
            videos = batch["video"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            with autocast_context:
                logits = model(videos)
                
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    print("\n" + "=" * 40)
    print(f"{args.split.upper()} RESULTS")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 40)
    
    base_out_dir = Path(args.checkpoint).parent
    out_dir = base_out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
        
    out_file = out_dir / f"{args.split}_results.json"
    
    results = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "test_acc": acc, # kept key as test_acc for backwards compatibility if needed, or better change it
        f"{args.split}_acc": acc,
        f"{args.split}_precision": prec,
        f"{args.split}_recall": rec,
        f"{args.split}_f1": f1
    }
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    idx_to_label = {v: k for k, v in test_dataset.label_to_idx.items()}
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    class_indices = list(range(len(idx_to_label)))
    
    # Save Classification Report
    report_dict = classification_report(all_labels, all_preds, labels=class_indices, target_names=target_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = out_dir / f"{args.split}_classification_report.csv"
    report_df.to_csv(report_csv_path)
    print(f"Classification report saved to: {report_csv_path}")

    # Save Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds, labels=class_indices)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix ({args.split})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_plot_path = out_dir / f"{args.split}_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_plot_path}")
    
    results_records = []
    for i, (pred_idx, true_idx) in enumerate(zip(all_preds, all_labels)):
        sample = test_dataset.samples[i]
        
        if sample["frames"]:
            video_id = sample["frames"][0].parent.name
        else:
            video_id = f"unknown_{i}"
            
        pred_label = idx_to_label[pred_idx]
        true_label = idx_to_label[true_idx]
        
        results_records.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted_label": pred_label,
            "is_match": bool(pred_idx == true_idx)
        })
        
    df_results = pd.DataFrame(results_records)
    csv_out_path = out_dir / f"{args.split}_predictions.csv"
    df_results.to_csv(csv_out_path, index=False)
    
    print(f"Results JSON saved to: {out_file}")
    print(f"Predictions saved to: {csv_out_path}")
    
    print("\n" + "=" * 40)
    print("TOP MISCLASSIFICATIONS")
    print("=" * 40)
    errors_df = df_results[~df_results["is_match"]]
    if not errors_df.empty:
        confusions = errors_df.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")
        confusions = confusions.sort_values(by="count", ascending=False).head(10)
        
        for _, row in confusions.iterrows():
            print(f"True: {row['true_label']:<15} | Pred: {row['predicted_label']:<15} | Count: {row['count']}")
            
        confusions_out_path = out_dir / f"{args.split}_top_confusions.csv"
        confusions.to_csv(confusions_out_path, index=False)
        print(f"\nTop confusions saved to: {confusions_out_path}")
    else:
        print("No misclassifications found! Perfect score.")

if __name__ == "__main__":
    main()