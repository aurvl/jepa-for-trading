from __future__ import annotations

import torch
from torch import nn

from jepa_trading.models.encoders import SinusoidalHorizonEmbedding
from jepa_trading.models.jepa import MarketJEPA


class PortfolioOutcomeModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        portfolio_state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_outputs: int = 6,
    ) -> None:
        super().__init__()
        self.horizon_embedding = SinusoidalHorizonEmbedding(latent_dim)
        input_dim = 2 * latent_dim + latent_dim + portfolio_state_dim + action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(
        self,
        z_now: torch.Tensor,
        z_future_hat: torch.Tensor,
        portfolio_state: torch.Tensor,
        action: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        h = self.horizon_embedding(horizon)
        global_now = z_now.mean(dim=1)
        global_future = z_future_hat.mean(dim=1)
        x = torch.cat([global_now, global_future, h, portfolio_state, action], dim=-1)
        return self.net(x)


class EnergyModel(nn.Module):
    def __init__(
        self,
        outcome_dim: int = 6,
        portfolio_state_dim: int = 0,
        action_dim: int = 0,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = outcome_dim + portfolio_state_dim + action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, outcome: torch.Tensor, portfolio_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([outcome, portfolio_state, action], dim=-1)).squeeze(-1)


class PlannerPolicyMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_abs_weight: float = 0.15,
        mode: str = "long_only",
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.max_abs_weight = max_abs_weight
        self.mode = mode
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, tradable_mask: torch.Tensor) -> torch.Tensor:
        raw = self.net(obs)
        asset_raw = raw[:, :-1].masked_fill(~tradable_mask.bool(), -1e9)
        cash_raw = raw[:, -1:]
        if self.mode == "long_only":
            weights = torch.softmax(torch.cat([asset_raw, cash_raw], dim=-1), dim=-1)
            asset_weights = torch.clamp(weights[:, :-1], max=self.max_abs_weight)
            cash = torch.clamp(1.0 - asset_weights.sum(dim=-1, keepdim=True), min=0.0)
            return torch.cat([asset_weights, cash], dim=-1)
        asset_weights = torch.tanh(asset_raw) * self.max_abs_weight
        asset_weights = asset_weights.masked_fill(~tradable_mask.bool(), 0.0)
        gross = asset_weights.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        asset_weights = asset_weights / gross
        cash = torch.clamp(1.0 - asset_weights.abs().sum(dim=-1, keepdim=True), min=0.0)
        return torch.cat([asset_weights, cash], dim=-1)


class V2WorldModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        max_assets: int,
        portfolio_state_dim: int,
        action_dim: int,
        d_model: int = 128,
        latent_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        ema_decay: float = 0.996,
        hidden_dim: int = 256,
        policy_mode: str = "long_only",
        max_abs_weight: float = 0.15,
        asset_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.market_jepa = MarketJEPA(
            n_features=n_features,
            max_assets=max_assets,
            d_model=d_model,
            latent_dim=latent_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            ema_decay=ema_decay,
            asset_embedding_dim=asset_embedding_dim,
        )
        self.outcome_model = PortfolioOutcomeModel(
            latent_dim=latent_dim,
            portfolio_state_dim=portfolio_state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_outputs=6,
        )
        self.energy_model = EnergyModel(
            outcome_dim=6,
            portfolio_state_dim=portfolio_state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim // 2,
        )
        obs_dim = max_assets * (latent_dim + 1) + portfolio_state_dim
        self.policy = PlannerPolicyMLP(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_abs_weight=max_abs_weight,
            mode=policy_mode,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_hat, z_target, z_now = self.market_jepa(
            batch["context"],
            batch["target"],
            batch["horizon"],
            batch["context_mask"],
            batch["target_mask"],
        )
        outcome_hat = self.outcome_model(
            z_now,
            z_hat,
            batch["portfolio_state"],
            batch["action"],
            batch["horizon"],
        )
        energy_hat = self.energy_model(outcome_hat, batch["portfolio_state"], batch["action"])
        obs = torch.cat(
            [
                z_now.reshape(z_now.shape[0], -1),
                batch["tradable_mask"].float(),
                batch["portfolio_state"],
            ],
            dim=-1,
        )
        policy_action = self.policy(obs, batch["tradable_mask"])
        return {
            "z_hat": z_hat,
            "z_target": z_target,
            "z_now": z_now,
            "outcome_hat": outcome_hat,
            "energy_hat": energy_hat,
            "policy_action": policy_action,
        }

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        self.market_jepa.update_target_encoder()

