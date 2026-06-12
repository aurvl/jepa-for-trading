from __future__ import annotations

import copy

import torch
from torch import nn

from jepa_trading.models.encoders import SinusoidalHorizonEmbedding
from jepa_trading.models.jepa import MarketJEPA


class PortfolioStateEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionConditionedTransition(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        portfolio_state_dim: int,
        action_dim: int,
        goal_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.horizon_embedding = SinusoidalHorizonEmbedding(latent_dim)
        input_dim = 4 * latent_dim + portfolio_state_dim + action_dim + goal_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z_market_now: torch.Tensor,
        z_market_future: torch.Tensor,
        z_portfolio_now: torch.Tensor,
        portfolio_state: torch.Tensor,
        action: torch.Tensor,
        horizon: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        h = self.horizon_embedding(horizon)
        global_now = z_market_now.mean(dim=1)
        global_future = z_market_future.mean(dim=1)
        x = torch.cat([global_now, global_future, z_portfolio_now, h, portfolio_state, action, goal], dim=-1)
        return self.net(x)


class V5OutcomeCostHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256, outcome_dim: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.outcome = nn.Linear(hidden_dim, outcome_dim)
        self.cost = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z)
        return self.outcome(h), self.cost(h).squeeze(-1)


class V5ActionConditionedWorldModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        max_assets: int,
        portfolio_state_dim: int,
        action_dim: int,
        goal_dim: int = 6,
        d_model: int = 128,
        latent_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        ema_decay: float = 0.996,
        hidden_dim: int = 256,
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
        self.portfolio_encoder = PortfolioStateEncoder(portfolio_state_dim, latent_dim, hidden_dim)
        self.target_portfolio_encoder = copy.deepcopy(self.portfolio_encoder)
        for p in self.target_portfolio_encoder.parameters():
            p.requires_grad_(False)
        self.transition = ActionConditionedTransition(
            latent_dim=latent_dim,
            portfolio_state_dim=portfolio_state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            hidden_dim=hidden_dim,
        )
        self.head = V5OutcomeCostHead(latent_dim, hidden_dim=hidden_dim)
        self.ema_decay = ema_decay

    def forward_candidates(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_market_hat, z_market_target, z_market_now = self.market_jepa(
            batch["context"],
            batch["target"],
            batch["horizon"],
            batch["context_mask"],
            batch["target_mask"],
        )
        batch_size, n_candidates, action_dim = batch["actions"].shape
        n_assets, latent_dim = z_market_now.shape[1], z_market_now.shape[2]
        z_now_flat = z_market_now[:, None].expand(batch_size, n_candidates, n_assets, latent_dim).reshape(
            batch_size * n_candidates, n_assets, latent_dim
        )
        z_future_flat = z_market_hat[:, None].expand(batch_size, n_candidates, n_assets, latent_dim).reshape(
            batch_size * n_candidates, n_assets, latent_dim
        )
        portfolio_flat = batch["portfolio_state"][:, None].expand(batch_size, n_candidates, -1).reshape(
            batch_size * n_candidates, -1
        )
        z_portfolio_now = self.portfolio_encoder(portfolio_flat)
        goal_flat = batch["goal"][:, None].expand(batch_size, n_candidates, -1).reshape(batch_size * n_candidates, -1)
        action_flat = batch["actions"].reshape(batch_size * n_candidates, action_dim)
        horizon_flat = batch["horizon"][:, None].expand(batch_size, n_candidates).reshape(batch_size * n_candidates)
        z_portfolio_hat = self.transition(
            z_now_flat,
            z_future_flat,
            z_portfolio_now,
            portfolio_flat,
            action_flat,
            horizon_flat,
            goal_flat,
        )
        outcome_hat, cost_hat = self.head(z_portfolio_hat)
        with torch.no_grad():
            z_portfolio_target = self.target_portfolio_encoder(
                batch["future_portfolio_states"].reshape(batch_size * n_candidates, -1)
            )
        return {
            "z_market_hat": z_market_hat,
            "z_market_target": z_market_target,
            "z_market_now": z_market_now,
            "z_portfolio_now": z_portfolio_now.reshape(batch_size, n_candidates, -1),
            "z_portfolio_hat": z_portfolio_hat.reshape(batch_size, n_candidates, -1),
            "z_portfolio_target": z_portfolio_target.reshape(batch_size, n_candidates, -1),
            "outcome_hat": outcome_hat.reshape(batch_size, n_candidates, -1),
            "cost_hat": cost_hat.reshape(batch_size, n_candidates),
        }

    @torch.no_grad()
    def update_target_encoders(self) -> None:
        self.market_jepa.update_target_encoder()
        for online, target in zip(self.portfolio_encoder.parameters(), self.target_portfolio_encoder.parameters()):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1 - self.ema_decay)
