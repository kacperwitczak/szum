import torch
import torch.nn as nn
import torchvision.models.video as video_models


class VideoSwinModel(nn.Module):
    """
    Video Swin Transformer for video classification.

    Input:
        x: [B, T, C, H, W]

    Internally torchvision expects:
        [B, C, T, H, W]
    """
    def __init__(
        self,
        num_classes: int,
        variant: str = "tiny",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        variant = variant.lower()
        self.variant = variant
        self.freeze = freeze_backbone

        if variant in ["tiny", "t"]:
            weights = (
                video_models.Swin3D_T_Weights.DEFAULT
                if pretrained else None
            )
            self.model = video_models.swin3d_t(weights=weights)

        elif variant in ["small", "s"]:
            weights = (
                video_models.Swin3D_S_Weights.DEFAULT
                if pretrained else None
            )
            self.model = video_models.swin3d_s(weights=weights)

        elif variant in ["base", "b"]:
            weights = (
                video_models.Swin3D_B_Weights.DEFAULT
                if pretrained else None
            )
            self.model = video_models.swin3d_b(weights=weights)

        else:
            raise ValueError(
                f"Unknown Video Swin variant: {variant}. "
                f"Use: tiny, small, base."
            )

        in_features = self.model.head.in_features

        self.model.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        if self.freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        self.freeze = True

        for name, param in self.model.named_parameters():
            if not name.startswith("head."):
                param.requires_grad = False

    def unfreeze(self):
        self.freeze = False

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected x shape [B, T, C, H, W], got {x.shape}"
            )

        B, T, C, H, W = x.shape

        if C == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif C != 3:
            raise ValueError(f"C must be 1 or 3, got C={C}")

        # [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return self.model(x)