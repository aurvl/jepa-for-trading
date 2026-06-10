from __future__ import annotations

import torch
from torch import nn


QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


class MarketHeads(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.return_quantiles = nn.Linear(hidden_dim, len(QUANTILES))
        self.sigma = nn.Linear(hidden_dim, 1)
        self.drawdown = nn.Linear(hidden_dim, 1)
        self.positive_logit = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(z)
        return {
            "return_quantiles": self.return_quantiles(h),
            "sigma": self.sigma(h).squeeze(-1),
            "drawdown": self.drawdown(h).squeeze(-1),
            "positive_logit": self.positive_logit(h).squeeze(-1),
        }


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...] = QUANTILES,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    target = target.unsqueeze(-1)
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    err = target - pred
    loss = torch.maximum(q * err, (q - 1) * err)
    if mask is not None:
        loss = loss * mask.float().unsqueeze(-1)
        return loss.sum() / (mask.float().sum().clamp_min(1.0) * len(quantiles))
    return loss.mean()
