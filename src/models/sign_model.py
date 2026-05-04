import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from .backbones import EfficientNetBackbone, DinoV2Backbone
from .transformer import TemporalTransformer
from .rnn import TemporalGRU, TemporalLSTM
from .video_swin import VideoSwinModel


class SignModel(nn.Module):
    """
    Main sign language classification model.

    Supported modes:

    1. Frame backbone + temporal model:
        backbone_type:
            - efficientnet_b0
            - efficientnet_v2_s
            - dinov2

        temporal_type:
            - gru
            - lstm
            - transformer

    2. End-to-end video transformer:
        backbone_type:
            - video_swin_t
            - video_swin_s
            - video_swin_b

        In this mode temporal_type is ignored.
    """
    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = True,
        max_seq_len: int = 64,
        backbone_type: str = "efficientnet_b0",
        temporal_type: str = "lstm",
        video_swin_pretrained: bool = True
    ):
        super().__init__()

        self.backbone_type = backbone_type.lower()
        self.temporal_type = temporal_type.lower()
        self.is_video_model = False

        # =========================
        # Video Swin mode
        # =========================

        if self.backbone_type in [
            "video_swin_t",
            "video_swin_tiny",
            "video_swin_s",
            "video_swin_small",
            "video_swin_b",
            "video_swin_base"
        ]:
            self.is_video_model = True

            if self.backbone_type in ["video_swin_t", "video_swin_tiny"]:
                variant = "tiny"
            elif self.backbone_type in ["video_swin_s", "video_swin_small"]:
                variant = "small"
            elif self.backbone_type in ["video_swin_b", "video_swin_base"]:
                variant = "base"
            else:
                raise ValueError(f"Unknown backbone_type: {backbone_type}")

            self.video_model = VideoSwinModel(
                num_classes=num_classes,
                variant=variant,
                pretrained=video_swin_pretrained,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate
            )

            return

        # =========================
        # Frame backbone + temporal model mode
        # =========================

        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if self.backbone_type == "dinov2":
            self.backbone = DinoV2Backbone(freeze=freeze_backbone)

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