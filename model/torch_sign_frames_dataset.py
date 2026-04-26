from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


DEFAULT_AUGMENTATIONS: tuple[dict[str, Any], ...] = (
    {"name": "slow_x0_8", "kind": "temporal_resample", "factor": 0.8},
    {"name": "fast_x1_25", "kind": "temporal_resample", "factor": 1.25},
    {"name": "interp_plus20", "kind": "temporal_resample", "factor": 1.0 / 1.2},
    {"name": "drop_every_5", "kind": "drop_stride", "stride": 5},
    {"name": "shift_left_8", "kind": "shift", "dx": -8, "dy": 0},
    {"name": "shift_right_8", "kind": "shift", "dx": 8, "dy": 0},
    {"name": "shift_up_8", "kind": "shift", "dx": 0, "dy": -8},
    {"name": "shift_down_8", "kind": "shift", "dx": 0, "dy": 8},
    {"name": "rot_left_5", "kind": "rotate", "angle": -5.0},
    {"name": "rot_right_5", "kind": "rotate", "angle": 5.0},
    {"name": "zoom_in_110", "kind": "zoom", "scale": 1.10},
    {"name": "zoom_out_90", "kind": "zoom", "scale": 0.90},
)


@dataclass
class OnTheFlyAugmentationConfig:
    enabled: bool = True
    p_apply: float = 0.90
    p_no_augmentation: float = 0.20
    max_ops_per_sample: int = 3
    augmentations: tuple[dict[str, Any], ...] = field(default_factory=lambda: DEFAULT_AUGMENTATIONS)


class SignFramesDataset(Dataset):
    """
    Torch Dataset for sign-language frame sequences.

    Expected CSV schema is compatible with split output from the project pipeline,
    especially files like merged_datasets/split8c_frames_signer_disjoint.csv.

    Key behavior:
    - loads frames from frames_path folder
    - returns fixed-length temporal sequences (default T=64)
    - uses uniform temporal resampling
      * N > T: downsample over entire clip (no tail truncation)
      * N < T: upsample by repeated indices
    - train mode uses random variant of uniform sampling
    - val/test use deterministic uniform sampling
    - train mode can apply on-the-fly augmentations from notebook 08 list

    Call set_epoch(epoch) from the training loop if you want deterministic-but-changing
    augment/sampling choices across epochs.
    """

    def __init__(
        self,
        csv_path: str | Path,
        project_root: str | Path | None = None,
        split: str | None = "train",
        mode: str = "train",
        sequence_length: int = 64,
        frames_path_col: str = "frames_path",
        label_col: str = "label",
        split_col: str = "split",
        process_status_col: str = "process_status",
        require_process_ok: bool = True,
        frame_extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
        target_size_hw: tuple[int, int] = (224, 224),
        normalize_to_01: bool = True,
        label_to_index: dict[str, int] | None = None,
        augmentation: OnTheFlyAugmentationConfig | None = None,
        random_sampling_jitter: bool = True,
        base_seed: int = 42,
    ) -> None:
        super().__init__()

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV does not exist: {self.csv_path}")

        self.project_root = Path(project_root) if project_root is not None else self.csv_path.parent.parent

        self.mode = mode.lower().strip()
        if self.mode not in {"train", "val", "test"}:
            raise ValueError("mode must be one of: train, val, test")

        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")

        self.frames_path_col = frames_path_col
        self.label_col = label_col
        self.split_col = split_col
        self.process_status_col = process_status_col
        self.require_process_ok = bool(require_process_ok)
        self.frame_extensions = {ext.lower() for ext in frame_extensions}
        self.target_h, self.target_w = int(target_size_hw[0]), int(target_size_hw[1])
        self.normalize_to_01 = bool(normalize_to_01)
        self.random_sampling_jitter = bool(random_sampling_jitter)
        self.base_seed = int(base_seed)
        self.epoch = 0

        self.augmentation = augmentation if augmentation is not None else OnTheFlyAugmentationConfig()

        df = pd.read_csv(self.csv_path)

        required = [self.frames_path_col, self.label_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        if self.require_process_ok and self.process_status_col in df.columns:
            df = df[df[self.process_status_col] == "ok"].copy()

        if split is not None:
            if self.split_col not in df.columns:
                raise ValueError(
                    f"Requested split='{split}', but column '{self.split_col}' does not exist in CSV"
                )
            df = df[df[self.split_col].astype(str) == str(split)].copy()

        df = df.dropna(subset=[self.frames_path_col, self.label_col]).copy()
        df[self.frames_path_col] = df[self.frames_path_col].astype(str)
        df[self.label_col] = df[self.label_col].astype(str)

        if len(df) == 0:
            raise ValueError("No rows available after filtering. Check split/process_status filters.")

        if label_to_index is None:
            labels = sorted(df[self.label_col].unique().tolist())
            self.label_to_index = {name: i for i, name in enumerate(labels)}
        else:
            self.label_to_index = dict(label_to_index)

        unknown_labels = sorted(set(df[self.label_col]) - set(self.label_to_index.keys()))
        if unknown_labels:
            raise ValueError(f"Found labels not present in label_to_index: {unknown_labels[:10]}")

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def build_weighted_sampler(
        self,
        power: float = 0.5,
        replacement: bool = True,
        num_samples: int | None = None,
    ) -> WeightedRandomSampler:
        if power <= 0:
            raise ValueError("power must be > 0")

        counts = self.df[self.label_col].value_counts().to_dict()
        weights = self.df[self.label_col].map(lambda x: 1.0 / (float(counts[x]) ** power)).to_numpy(dtype=np.float64)
        weights_t = torch.from_numpy(weights)

        if num_samples is None:
            num_samples = len(weights)

        return WeightedRandomSampler(weights=weights_t, num_samples=int(num_samples), replacement=bool(replacement))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[int(index)]
        frames_path = str(row[self.frames_path_col])
        frames_dir = self._resolve_frames_dir(frames_path)

        frames = self._load_frames(frames_dir)
        raw_num_frames = len(frames)

        rng = self._rng_for_index(int(index))
        applied_augs: list[str] = []

        if self.mode == "train":
            frames, applied_augs = self._apply_random_augmentations(frames, rng)

        indices = self._uniform_temporal_indices(
            num_frames=len(frames),
            target_len=self.sequence_length,
            train_mode=(self.mode == "train"),
            rng=rng,
        )
        sampled_frames = [frames[i] for i in indices]

        clip = np.stack(sampled_frames, axis=0).astype(np.float32)
        if self.normalize_to_01:
            clip = clip / 255.0

        # Final tensor shape: [C, T, H, W] where C=1 for grayscale.
        video_tensor = torch.from_numpy(clip).unsqueeze(0)

        label_name = str(row[self.label_col])
        label_idx = int(self.label_to_index[label_name])

        out: dict[str, Any] = {
            "video": video_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_name": label_name,
            "frames_path": frames_path,
            "split": str(row[self.split_col]) if self.split_col in row else "",
            "raw_num_frames": int(raw_num_frames),
            "indices": torch.from_numpy(indices.astype(np.int64)),
            "applied_augs": tuple(applied_augs),
        }
        return out

    def _rng_for_index(self, index: int) -> np.random.Generator:
        seed = self.base_seed + self.epoch * 1_000_003 + index * 977
        return np.random.default_rng(seed)

    def _resolve_frames_dir(self, frames_path: str) -> Path:
        p = Path(frames_path)
        if p.is_absolute():
            return p
        return self.project_root / p

    def _load_frames(self, frames_dir: Path) -> list[np.ndarray]:
        if not frames_dir.exists():
            raise FileNotFoundError(f"frames_path does not exist: {frames_dir}")

        frame_files = [
            p for p in sorted(frames_dir.iterdir()) if p.is_file() and p.suffix.lower() in self.frame_extensions
        ]
        if len(frame_files) == 0:
            raise RuntimeError(f"No frame files found in: {frames_dir}")

        frames: list[np.ndarray] = []
        for fp in frame_files:
            frame = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue

            if frame.shape[0] != self.target_h or frame.shape[1] != self.target_w:
                frame = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

            frames.append(frame)

        if len(frames) == 0:
            raise RuntimeError(f"Frames exist but all reads failed in: {frames_dir}")

        return frames

    def _uniform_temporal_indices(
        self,
        num_frames: int,
        target_len: int,
        train_mode: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if num_frames <= 0:
            raise ValueError("num_frames must be > 0")
        if target_len <= 0:
            raise ValueError("target_len must be > 0")

        # Bins cover full clip, so N>T keeps full temporal span, N<T duplicates indices.
        bin_edges = np.linspace(0.0, float(num_frames), num=target_len + 1)
        out = np.empty(target_len, dtype=np.int64)

        for i in range(target_len):
            left = int(np.floor(bin_edges[i]))
            right = int(np.ceil(bin_edges[i + 1])) - 1

            left = max(0, min(left, num_frames - 1))
            right = max(left, min(right, num_frames - 1))

            if train_mode and self.random_sampling_jitter and right > left:
                out[i] = int(rng.integers(left, right + 1))
            else:
                out[i] = int(round((left + right) / 2.0))

        return out

    def _apply_random_augmentations(
        self,
        frames: list[np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], list[str]]:
        cfg = self.augmentation
        if not cfg.enabled:
            return frames, []
        if len(frames) == 0:
            return frames, []
        if rng.random() > cfg.p_apply:
            return frames, []
        if rng.random() < cfg.p_no_augmentation:
            return frames, []

        max_ops = max(1, int(cfg.max_ops_per_sample))
        n_ops = int(rng.integers(1, max_ops + 1))

        candidates = list(cfg.augmentations)
        rng.shuffle(candidates)

        temporal_kinds = {"temporal_resample", "drop_stride"}
        selected: list[dict[str, Any]] = []
        used_temporal = False

        for aug in candidates:
            is_temporal = str(aug.get("kind", "")) in temporal_kinds
            if is_temporal and used_temporal:
                continue
            selected.append(aug)
            if is_temporal:
                used_temporal = True
            if len(selected) >= n_ops:
                break

        out = frames
        names: list[str] = []
        for aug in selected:
            out = self._apply_augmentation(out, aug)
            if len(out) == 0:
                out = [frames[-1]]
            names.append(str(aug.get("name", "unknown")))

        return out, names

    def _apply_augmentation(self, frames: list[np.ndarray], aug: dict[str, Any]) -> list[np.ndarray]:
        kind = str(aug["kind"])

        if kind == "temporal_resample":
            return self._temporal_resample(frames, float(aug["factor"]))

        if kind == "drop_stride":
            return self._drop_stride(frames, int(aug["stride"]))

        if kind == "shift":
            dx = int(aug["dx"])
            dy = int(aug["dy"])
            return [self._shift_frame(f, dx, dy) for f in frames]

        if kind == "rotate":
            angle = float(aug["angle"])
            return [self._rotate_frame(f, angle) for f in frames]

        if kind == "zoom":
            scale = float(aug["scale"])
            return [self._zoom_frame(f, scale) for f in frames]

        raise ValueError(f"Unknown augmentation kind: {kind}")

    @staticmethod
    def _temporal_resample(frames: list[np.ndarray], factor: float) -> list[np.ndarray]:
        if len(frames) < 2:
            return frames

        n_in = len(frames)
        n_out = max(2, int(round(n_in / factor)))
        out: list[np.ndarray] = []

        for i in range(n_out):
            pos = i * (n_in - 1) / (n_out - 1)
            left = int(np.floor(pos))
            right = min(left + 1, n_in - 1)
            alpha = float(pos - left)

            if right == left:
                out.append(frames[left].copy())
            else:
                mixed = cv2.addWeighted(frames[left], 1.0 - alpha, frames[right], alpha, 0.0)
                out.append(mixed)

        return out

    @staticmethod
    def _drop_stride(frames: list[np.ndarray], stride: int) -> list[np.ndarray]:
        if len(frames) == 0:
            return frames
        stride = max(2, int(stride))
        out = [f for i, f in enumerate(frames) if (i + 1) % stride != 0]
        return out if len(out) > 0 else [frames[0]]

    @staticmethod
    def _shift_frame(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
        h, w = frame.shape[:2]
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _rotate_frame(frame: np.ndarray, angle: float) -> np.ndarray:
        h, w = frame.shape[:2]
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _zoom_frame(frame: np.ndarray, scale: float) -> np.ndarray:
        h, w = frame.shape[:2]
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        if scale >= 1.0:
            y0 = (nh - h) // 2
            x0 = (nw - w) // 2
            return resized[y0 : y0 + h, x0 : x0 + w]

        out = np.zeros_like(frame)
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        out[y0 : y0 + nh, x0 : x0 + nw] = resized
        return out


def build_weighted_sampler_from_df(
    df: pd.DataFrame,
    label_col: str = "label",
    power: float = 0.5,
    replacement: bool = True,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler from a dataframe.

    Weight formula: w_i = 1 / count(label_i)^power.
    Using power=0.5 is usually less aggressive than full inverse frequency.
    """
    if label_col not in df.columns:
        raise ValueError(f"Column not found in df: {label_col}")
    if power <= 0:
        raise ValueError("power must be > 0")

    counts = df[label_col].value_counts().to_dict()
    sample_weights = df[label_col].map(lambda x: 1.0 / (float(counts[x]) ** power)).to_numpy(dtype=np.float64)
    weights_t = torch.from_numpy(sample_weights)

    if num_samples is None:
        num_samples = len(sample_weights)

    return WeightedRandomSampler(weights=weights_t, num_samples=int(num_samples), replacement=bool(replacement))
