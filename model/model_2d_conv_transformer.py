"""
2D CNN + Transformer for sign language recognition.

Architecture:
  Input: [B, 1, T, 224, 224] (grayscale video)
  → Grayscale to RGB
  → Per-frame ResNet50 encoder: [B, T, 2048]
  → Temporal Transformer: [B, 2048]
  → Classification head: [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class Conv2DTransformer(nn.Module):
    """
    2D ResNet50 per-frame encoder + Temporal Transformer + Classifier.
    """

    def __init__(self, num_classes: int = 30, dropout_rate: float = 0.3) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Grayscale (1ch) to RGB (3ch) adapter
        self.adapter = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        nn.init.constant_(self.adapter.weight, 1.0 / 3.0)
        nn.init.constant_(self.adapter.bias, 0.0)

        # ResNet50 backbone (pretrained ImageNet)
        resnet50 = models.resnet50(weights="IMAGENET1K_V2")
        self.encoder = nn.Sequential(*list(resnet50.children())[:-1])  # Remove avgpool + fc
        self.encoder_dim = 2048

        # Positional embedding for temporal dimension
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, self.encoder_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout_rate,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(self.encoder_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T, 224, 224] grayscale video
        Returns:
            [B, num_classes] logits
        """
        B, C, T, H, W = x.shape
        assert C == 1, f"Expected grayscale (1 channel), got {C}"

        # Reshape to [B*T, 1, H, W] for per-frame processing
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
        x = x.reshape(B * T, 1, H, W)  # [B*T, 1, H, W]

        # Convert grayscale to RGB
        x = self.adapter(x)  # [B*T, 3, H, W]

        # Per-frame ResNet50 encoding
        x = self.encoder(x)  # [B*T, 2048, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B*T, 2048]

        # Reshape back to [B, T, 2048]
        x = x.reshape(B, T, -1)

        # Add positional embeddings
        x = x + self.pos_embed[:, :T, :]

        # Temporal Transformer
        x = self.transformer(x)  # [B, T, 2048]
        x = self.norm(x)

        # Global average pooling over time
        x = x.mean(dim=1)  # [B, 2048]

        # Classification
        logits = self.head(x)  # [B, num_classes]

        return logits

    def get_num_parameters(self) -> tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
