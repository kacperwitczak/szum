import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetBackbone(nn.Module):
    def __init__(self, variant: str = "b0", freeze: bool = True):
        super().__init__()
        self.freeze = freeze

        variant = variant.lower()

        if variant == "v2_s":
            eff = models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.DEFAULT
            )
            self.out_channels = 1280
        elif variant == "b0":
            eff = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            self.out_channels = 1280
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")

        self.features = nn.Sequential(
            eff.features,
            eff.avgpool,
            nn.Flatten(1)
        )

        if self.freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        self.freeze = True
        self.features.eval()
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        self.freeze = False
        self.features.train()
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            self.features.eval()
            with torch.no_grad():
                return self.features(x)
        return self.features(x)


class DinoV2Backbone(nn.Module):
    def __init__(self, freeze: bool = True):
        super().__init__()
        self.freeze = freeze
        self.out_channels = 384

        self.features = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14"
        )

        if self.freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        self.freeze = True
        self.features.eval()
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        self.freeze = False
        self.features.train()
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            self.features.eval()
            with torch.no_grad():
                return self.features(x)
        return self.features(x)