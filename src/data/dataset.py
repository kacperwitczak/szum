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

# Wygenerowanie przy pomocy AI
class SignFramesDataset(Dataset):
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
        
        df_full = pd.read_csv(self.csv_path)
        labels = sorted(df_full["label"].unique().tolist())
        self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        
        df = df_full[df_full["split"] == split].reset_index(drop=True)
        self.df = df
        
        self.transform = build_spatial_transform(is_train=self.is_train, p_apply=spatial_aug_p)
        
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
        
        total_frames = len(frame_paths)
        
        if total_frames == self.sequence_length:
            selected_paths = frame_paths
        else:
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
            
            if self.is_train and total_frames > self.sequence_length:
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
            
        if self.transform is not None:
            saved_augmentations = self.transform(image=frames_np[0])["replay"]
            aug_frames = []
            for img in frames_np:
                res = A.ReplayCompose.replay(saved_augmentations, image=img)
                aug_frames.append(res["image"])
            frames_np = aug_frames

        video_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).float() / 255.0
        video_tensor = video_tensor.unsqueeze(1)

        return {
            "video": video_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }
