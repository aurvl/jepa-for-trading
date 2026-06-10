from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


class PortfolioPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_assets: int,
        hidden_dim: int = 256,
        max_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.max_weight = max_weight
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Assets + cash logits. Softmax gives long-only, no-leverage weights.
        self.actor = nn.Linear(hidden_dim, n_assets + 1)
        self.critic = nn.Linear(hidden_dim, 1)

    def _masked_logits(self, obs: torch.Tensor, tradable_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        logits = self.actor(h)
        asset_logits = logits[..., : self.n_assets]
        cash_logit = logits[..., self.n_assets :]
        asset_logits = asset_logits.masked_fill(~tradable_mask.bool(), -1e9)
        return torch.cat([asset_logits, cash_logit], dim=-1), self.critic(h).squeeze(-1)

    def _weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(logits, dim=-1)
        asset_weights = torch.clamp(weights[..., : self.n_assets], max=self.max_weight)
        cash = 1.0 - asset_weights.sum(dim=-1, keepdim=True)
        cash = cash.clamp_min(0.0)
        total = asset_weights.sum(dim=-1, keepdim=True) + cash
        return torch.cat([asset_weights, cash], dim=-1) / total.clamp_min(1e-8)

    def forward(self, obs: torch.Tensor, tradable_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value = self._masked_logits(obs, tradable_mask)
        weights = self._weights_from_logits(logits)
        return weights, value

    def entropy_proxy(self, weights: torch.Tensor) -> torch.Tensor:
        return -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        tradable_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self._masked_logits(obs, tradable_mask)
        weights = self._weights_from_logits(logits)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        if action is None:
            action = weights
        # Continuous target weights are evaluated with a cross-entropy style proxy.
        log_prob = (action.detach() * log_probs).sum(dim=-1)
        return action, log_prob, entropy, value
