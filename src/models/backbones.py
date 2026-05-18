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


class DinoV3Backbone(nn.Module):
    def __init__(self, size: str = "vits16", freeze: bool = True):
        super().__init__()
        self.freeze = freeze
        
        # Oczekiwane wymiary wyjściowe
        if "vits" in size:
            self.out_channels = 384
        elif "vitb" in size:
            self.out_channels = 768
        elif "vitl" in size:
            self.out_channels = 1024
        else:
            self.out_channels = 384
            
        try:
            # Mapowanie modeli do linków z wagami Mety
            weights_url = None
            if size == "vits16":
                weights_url = "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoieGN6NGpueGY0MzdmZWF4cWVhdHYwbzc2IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NzkyMjc3NTN9fX1dfQ__&Signature=pxu1GM7GgZbfozsm5sAYSpR4-OC2sGCIXL%7E6Ke3IpiVdMZ%7EZUrgG8avej3C4JeshoO2%7EHawflibjzl9tNXZjYLwYRcZEbk8ItOsmOlCG9EEJQaEExh0Ujssf%7Epy0jhUd4-O1yasH%7EO63kfGS5ZYjhbNOK7D4wB9jmebcHF9H4DfKYDpyZlP1brmyFdrL%7EY%7EvgsUn-yEYsfeGny1%7EHd4Ay7e0DzMr2jj5THH-eqMwijtoUb75kTQT05KSk83W1b3%7E8BMLrRWN8wM5kvGISlygXDh7FE9mtyL9X-zWRAgBujjF4fjNxbQMpL6iONN1KiGBCTBRl8VcBi644NTs5nFBvw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1458323905534205"

            if weights_url:
                self.features = torch.hub.load("facebookresearch/dinov3", f"dinov3_{size}", trust_repo=True, weights=weights_url)
            else:
                # Wczytujemy od zera (losowe wagi) dla innych wymiarów, których adresów URL nam brakło
                self.features = torch.hub.load("facebookresearch/dinov3", f"dinov3_{size}", trust_repo=True, pretrained=False)
        except Exception as e:
            print(f"Nie udało się pobrać DINOv3 z hub'a: {e}")
            print("Możesz musieć ustawić niestandardową ścieżkę do repozytorium GitHub/lokalnych wag DINOv3.")
            raise e

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