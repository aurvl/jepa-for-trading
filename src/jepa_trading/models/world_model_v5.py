from __future__ import annotations

import copy

import torch
from torch import nn

from jepa_trading.data.v5_dataset import V5_QUANTILES
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


class ActionConditionedPortfolioTransition(nn.Module):
    """Portfolio transition. The goal is intentionally absent from this module."""

    def __init__(
        self,
        latent_dim: int,
        portfolio_state_dim: int,
        primitive_action_dim: int,
        executable_action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.horizon_embedding = SinusoidalHorizonEmbedding(latent_dim)
        input_dim = 4 * latent_dim + portfolio_state_dim + primitive_action_dim + executable_action_dim
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
        primitive_action: torch.Tensor,
        executable_action: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        h = self.horizon_embedding(horizon)
        global_now = z_market_now.mean(dim=1)
        global_future = z_market_future.mean(dim=1)
        x = torch.cat(
            [global_now, global_future, z_portfolio_now, h, portfolio_state, primitive_action, executable_action],
            dim=-1,
        )
        return self.net(x)


class V5ProbabilisticOutcomeHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256, n_quantiles: int = len(V5_QUANTILES)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.return_quantiles = nn.Linear(hidden_dim, n_quantiles)
        self.drawdown_quantiles = nn.Linear(hidden_dim, n_quantiles)
        self.volatility = nn.Linear(hidden_dim, 1)
        self.turnover = nn.Linear(hidden_dim, 1)
        self.cost = nn.Linear(hidden_dim, 1)
        self.cvar = nn.Linear(hidden_dim, 1)
        self.prob_loss_logit = nn.Linear(hidden_dim, 1)
        self.prob_drawdown_breach_logit = nn.Linear(hidden_dim, 1)
        self.future_equity_ratio = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(z)
        return {
            "return_quantiles": self.return_quantiles(h),
            "drawdown_quantiles": self.drawdown_quantiles(h),
            "volatility": torch.nn.functional.softplus(self.volatility(h)).squeeze(-1),
            "turnover": torch.nn.functional.softplus(self.turnover(h)).squeeze(-1),
            "cost": torch.nn.functional.softplus(self.cost(h)).squeeze(-1),
            "cvar": self.cvar(h).squeeze(-1),
            "prob_loss_logit": self.prob_loss_logit(h).squeeze(-1),
            "prob_drawdown_breach_logit": self.prob_drawdown_breach_logit(h).squeeze(-1),
            "future_equity_ratio": torch.nn.functional.softplus(self.future_equity_ratio(h)).squeeze(-1),
        }


def outcome_dict_to_features(outcomes: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            outcomes["return_quantiles"],
            outcomes["drawdown_quantiles"],
            outcomes["volatility"].unsqueeze(-1),
            outcomes["turnover"].unsqueeze(-1),
            outcomes["cost"].unsqueeze(-1),
            outcomes["cvar"].unsqueeze(-1),
            torch.sigmoid(outcomes["prob_loss_logit"]).unsqueeze(-1),
            torch.sigmoid(outcomes["prob_drawdown_breach_logit"]).unsqueeze(-1),
            outcomes["future_equity_ratio"].unsqueeze(-1),
        ],
        dim=-1,
    )


class V5CostEnergyModel(nn.Module):
    """Goal-conditioned evaluator. It scores outcomes; it does not change dynamics."""

    def __init__(self, outcome_feature_dim: int, goal_dim: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(outcome_feature_dim + goal_dim),
            nn.Linear(outcome_feature_dim + goal_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, outcome_features: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([outcome_features, goal], dim=-1)
        return self.net(x).squeeze(-1)


class V5ActionConditionedWorldModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        max_assets: int,
        portfolio_state_dim: int,
        primitive_action_dim: int | None = None,
        executable_action_dim: int | None = None,
        action_dim: int | None = None,
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
        if primitive_action_dim is None:
            primitive_action_dim = action_dim
        if primitive_action_dim is None:
            raise ValueError("primitive_action_dim or action_dim must be provided")
        executable_action_dim = executable_action_dim or (max_assets + 1)
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
        self.transition = ActionConditionedPortfolioTransition(
            latent_dim=latent_dim,
            portfolio_state_dim=portfolio_state_dim,
            primitive_action_dim=primitive_action_dim,
            executable_action_dim=executable_action_dim,
            hidden_dim=hidden_dim,
        )
        self.outcome_head = V5ProbabilisticOutcomeHead(latent_dim, hidden_dim=hidden_dim)
        self.cost_model = V5CostEnergyModel(2 * len(V5_QUANTILES) + 7, goal_dim=goal_dim, hidden_dim=hidden_dim)
        self.ema_decay = ema_decay

    def forward_candidates(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        z_market_hat, z_market_target, z_market_now = self.market_jepa(
            batch["context"],
            batch["target"],
            batch["horizon"],
            batch["context_mask"],
            batch["target_mask"],
        )
        batch_size, n_candidates, primitive_dim = batch["primitive_actions"].shape
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
        primitive_flat = batch["primitive_actions"].reshape(batch_size * n_candidates, primitive_dim)
        executable_flat = batch["executable_actions"].reshape(batch_size * n_candidates, -1)
        horizon_flat = batch["horizon"][:, None].expand(batch_size, n_candidates).reshape(batch_size * n_candidates)
        z_portfolio_hat = self.transition(
            z_now_flat,
            z_future_flat,
            z_portfolio_now,
            portfolio_flat,
            primitive_flat,
            executable_flat,
            horizon_flat,
        )
        outcomes = self.outcome_head(z_portfolio_hat)
        outcome_features = outcome_dict_to_features(outcomes)
        goal_flat = batch["goal"][:, None].expand(batch_size, n_candidates, -1).reshape(batch_size * n_candidates, -1)
        cost_hat = self.cost_model(outcome_features, goal_flat)
        with torch.no_grad():
            z_portfolio_target = self.target_portfolio_encoder(
                batch["future_portfolio_states"].reshape(batch_size * n_candidates, -1)
            )

        reshaped_outcomes = {}
        for key, value in outcomes.items():
            trailing = value.shape[1:]
            reshaped_outcomes[key] = value.reshape(batch_size, n_candidates, *trailing)
        return {
            "z_market_hat": z_market_hat,
            "z_market_target": z_market_target,
            "z_market_now": z_market_now,
            "z_portfolio_now": z_portfolio_now.reshape(batch_size, n_candidates, -1),
            "z_portfolio_hat": z_portfolio_hat.reshape(batch_size, n_candidates, -1),
            "z_portfolio_target": z_portfolio_target.reshape(batch_size, n_candidates, -1),
            "outcome_hat": reshaped_outcomes,
            "cost_hat": cost_hat.reshape(batch_size, n_candidates),
        }

    @torch.no_grad()
    def update_target_encoders(self) -> None:
        self.market_jepa.update_target_encoder()
        for online, target in zip(self.portfolio_encoder.parameters(), self.target_portfolio_encoder.parameters()):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1 - self.ema_decay)
