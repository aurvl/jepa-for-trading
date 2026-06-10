from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalHorizonEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, horizon: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = horizon.device
        freqs = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / max(half - 1, 1)))
        args = horizon.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class TemporalAssetEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        max_assets: int,
        d_model: int = 128,
        latent_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_assets = max_assets
        self.feature_proj = nn.Linear(n_features, d_model)
        self.asset_embedding = nn.Embedding(max_assets, d_model)
        self.time_embedding = nn.Parameter(torch.randn(1, 1, 512, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: batch x assets x time x features
        bsz, n_assets, steps, _ = x.shape
        if n_assets > self.max_assets:
            raise ValueError(f"n_assets={n_assets} exceeds max_assets={self.max_assets}")
        asset_ids = torch.arange(n_assets, device=x.device)
        h = self.feature_proj(x)
        h = h + self.asset_embedding(asset_ids)[None, :, None, :]
        h = h + self.time_embedding[:, :, :steps, :]
        h = h.reshape(bsz * n_assets, steps, -1)
        encoded = self.encoder(h)
        encoded = self.norm(encoded)

        if mask is not None:
            flat_mask = mask.reshape(bsz * n_assets, steps).float()
            denom = flat_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (encoded * flat_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = encoded.mean(dim=1)
        return self.to_latent(pooled).reshape(bsz, n_assets, -1)

