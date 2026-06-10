from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F

from jepa_trading.models.encoders import SinusoidalHorizonEmbedding, TemporalAssetEncoder


class MarketJEPA(nn.Module):
    def __init__(
        self,
        n_features: int,
        max_assets: int,
        d_model: int = 128,
        latent_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        ema_decay: float = 0.996,
        asset_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.online_encoder = TemporalAssetEncoder(
            n_features, max_assets, d_model, latent_dim, n_heads, n_layers, dropout
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.horizon_embedding = SinusoidalHorizonEmbedding(latent_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(2 * latent_dim),
            nn.Linear(2 * latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.ema_decay = ema_decay

    def encode_context(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.online_encoder(x, mask)

    @torch.no_grad()
    def encode_target(self, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.target_encoder(y, mask)

    def predict_future(self, z_context: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        h = self.horizon_embedding(horizon).unsqueeze(1).expand_as(z_context)
        return self.predictor(torch.cat([z_context, h], dim=-1))

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        horizon: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_context = self.encode_context(context, context_mask)
        z_hat = self.predict_future(z_context, horizon)
        z_target = self.encode_target(target, target_mask)
        return z_hat, z_target, z_context

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        decay = self.ema_decay
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(decay).add_(online.data, alpha=1 - decay)


def jepa_latent_loss(
    z_hat: torch.Tensor,
    z_target: torch.Tensor,
    valid_asset_mask: torch.Tensor,
) -> torch.Tensor:
    z_hat = F.normalize(z_hat, dim=-1)
    z_target = F.normalize(z_target.detach(), dim=-1)
    loss = 2 - 2 * (z_hat * z_target).sum(dim=-1)
    return (loss * valid_asset_mask.float()).sum() / valid_asset_mask.float().sum().clamp_min(1.0)
