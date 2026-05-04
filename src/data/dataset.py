import os
from pathlib import Path
from typing import Sequence
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A

from .transforms import build_spatial_transform

class SignFramesDataset(Dataset):
    """
    Highly optimized lazy-loading dataset for sign language frames.
    Loads raw JPGs, applies temporal sampling, and coherent spatial augmentations.
    """
    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        sequence_length: int = 32,
        target_size: tuple[int, int] = (224, 224),
        spatial_aug_p: float = 0.9,
    ):
        self.csv_path = Path(csv_path)
        self.project_root = self.csv_path.parent.parent
        self.split = split
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.is_train = (split == "train")
        
        # Load and filter CSV
        df_full = pd.read_csv(self.csv_path)
        # Mapping labels globally using the FULL dataset so sizes match across train/val
        labels = sorted(df_full["label"].unique().tolist())
        self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        
        # Assuming the generated CSV format from 7c/7e / 8c
        df = df_full[df_full["split"] == split].reset_index(drop=True)
        self.df = df
        
        # Transforms
        self.transform = build_spatial_transform(is_train=self.is_train, p_apply=spatial_aug_p)
        
        # Pre-scan directories to avoid disk IO bottlenecks during training
        self.samples = []
        for idx, row in df.iterrows():
            frames_dir = self.project_root / row["frames_path"]
            if frames_dir.exists():
                frames = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
                if frames:
                    self.samples.append({
                        "frames": frames,
                        "label": self.label_to_idx[row["label"]]
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        frame_paths = sample["frames"]
        label = sample["label"]
        
        # Uniform Temporal Sampling: Sample `sequence_length` frames evenly across the entire video.
        # This allows the model to see the full gesture regardless of video length, without blowing up VRAM.
        total_frames = len(frame_paths)
        
        if total_frames == self.sequence_length:
            selected_paths = frame_paths
        else:
            # Create evenly spaced indices across the available frames
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
            
            if self.is_train and total_frames > self.sequence_length:
                # Add a tiny bit of random jitter (shift indices slightly) to augment temporal sampling during training
                max_jitter = (total_frames / self.sequence_length) * 0.5
                jitter = np.random.uniform(-max_jitter, max_jitter, size=self.sequence_length)
                indices = np.clip(indices + jitter, 0, total_frames - 1).astype(int)
                
            selected_paths = [frame_paths[i] for i in indices]

        frames_np = []
        for p in selected_paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
            else:
                img = cv2.resize(img, self.target_size)
            frames_np.append(img)
            
        # Apply coherent spatial augmentations
        if self.transform is not None:
            # Albumentations might need a dummy channel for some transforms depending on how they are defined, 
            # but usually works on 2D images for grayscale if not using color-specific ones.
            # To be safe, we can add a channel dim for Albumentations and remove it later if needed,
            # or just apply them directly on the 2D arrays.
            saved_augmentations = self.transform(image=frames_np[0])["replay"]
            aug_frames = []
            for img in frames_np:
                res = A.ReplayCompose.replay(saved_augmentations, image=img)
                aug_frames.append(res["image"])
            frames_np = aug_frames

        # Convert to Tensor [T, C, H, W]
        # Currently frames_np is list of [H, W], stacking -> [T, H, W]
        video_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).float() / 255.0
        # Add channel dim -> [T, 1, H, W]
        video_tensor = video_tensor.unsqueeze(1)

        return {
            "video": video_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }
