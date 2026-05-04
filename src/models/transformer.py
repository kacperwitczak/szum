import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer over frame embeddings.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
        max_seq_len: int = 64
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        x = self.norm(x)

        return x.mean(dim=1)