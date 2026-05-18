import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from .backbones import EfficientNetBackbone, DinoV2Backbone, DinoV3Backbone
from .transformer import TemporalTransformer
from .rnn import TemporalGRU, TemporalLSTM


class SignModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = True,
        max_seq_len: int = 64,
        backbone_type: str = "efficientnet_b0",
        temporal_type: str = "lstm",
    ):
        super().__init__()

        self.backbone_type = backbone_type.lower()
        if self.backbone_type == "efficientnet":
            self.backbone_type = "efficientnet_b0"
            
        self.temporal_type = temporal_type.lower()
        self.is_video_model = False

        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if self.backbone_type == "dinov2":
            self.backbone = DinoV2Backbone(freeze=freeze_backbone)
            
        elif self.backbone_type == "dinov3":
            self.backbone = DinoV3Backbone(freeze=freeze_backbone)

        elif self.backbone_type == "efficientnet_b0":
            self.backbone = EfficientNetBackbone(
                variant="b0",
                freeze=freeze_backbone
            )

        elif self.backbone_type == "efficientnet_v2_s":
            self.backbone = EfficientNetBackbone(
                variant="v2_s",
                freeze=freeze_backbone
            )

        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        embed_dim = self.backbone.out_channels

        if self.temporal_type == "gru":
            self.temporal_model = TemporalGRU(
                input_dim=embed_dim,
                hidden_dim=512,
                num_layers=2,
                dropout=dropout_rate
            )
            temporal_out_dim = self.temporal_model.out_dim

        elif self.temporal_type == "lstm":
            self.temporal_model = TemporalLSTM(
                input_dim=embed_dim,
                hidden_dim=512,
                num_layers=2,
                dropout=dropout_rate
            )
            temporal_out_dim = self.temporal_model.out_dim

        elif self.temporal_type == "transformer":
            self.temporal_model = TemporalTransformer(
                embed_dim=embed_dim,
                dropout=dropout_rate,
                max_seq_len=max_seq_len
            )
            temporal_out_dim = embed_dim

        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(temporal_out_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected input:
            x: [B, T, C, H, W]
        """

        if self.is_video_model:
            return self.video_model(x)

        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)

        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif C != 3:
            raise ValueError(f"C must be 1 or 3, got C={C}")

        x = self.normalize(x)

        x = self.backbone(x)
        x = x.reshape(B, T, -1)

        x = self.temporal_model(x)

        return self.head(x)

    def unfreeze_backbone(self):
        if self.is_video_model:
            self.video_model.unfreeze()
        else:
            self.backbone.unfreeze()

    def get_num_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
        return total, trainable