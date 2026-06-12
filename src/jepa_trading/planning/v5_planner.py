from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jepa_trading.data.actions import portfolio_state_features
from jepa_trading.data.v2_dataset import PortfolioActionConfig
from jepa_trading.data.v5_dataset import V5CostConfig, goal_vector, v5_candidate_actions
from jepa_trading.models.world_model_v5 import V5ActionConditionedWorldModel


@dataclass
class V5PlannerResult:
    action: np.ndarray
    horizon: int
    predicted_cost: float
    selected_name: str
    predicted_outcome: np.ndarray
    candidate_names: list[str]
    candidate_horizons: list[int]
    candidate_costs: np.ndarray


class V5ActionWorldModelPlanner:
    CODE_VERSION = "v5.0-action-conditioned-world-model"

    def __init__(
        self,
        model: V5ActionConditionedWorldModel,
        action_config: PortfolioActionConfig,
        cost_config: V5CostConfig,
        horizons: list[int],
        n_sampled_actions: int = 128,
        max_turnover: float | None = 0.40,
        chunk_size: int = 256,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.model = model.to(device).eval()
        self.action_config = action_config
        self.cost_config = cost_config
        self.horizons = horizons
        self.n_sampled_actions = n_sampled_actions
        self.max_turnover = max_turnover
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def choose_action(
        self,
        context: np.ndarray,
        context_mask: np.ndarray,
        portfolio_state: np.ndarray,
        tradable_mask: np.ndarray,
        sigma_now: np.ndarray,
        recent_returns: np.ndarray,
    ) -> V5PlannerResult:
        context_t = torch.tensor(context[None], dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(context_mask[None], dtype=torch.bool, device=self.device)
        z_market_now = self.model.market_jepa.encode_context(context_t, mask_t)
        n_assets = len(tradable_mask)
        current_weights = np.asarray(portfolio_state[: n_assets + 1], dtype=np.float32)

        actions: list[np.ndarray] = []
        names: list[str] = []
        horizons: list[int] = []
        goals: list[np.ndarray] = []
        for horizon in self.horizons:
            candidates = v5_candidate_actions(
                rng=self.rng,
                valid=tradable_mask.astype(bool),
                sigma=sigma_now,
                recent_returns=recent_returns,
                current_weights=current_weights,
                n_assets=n_assets,
                config=self.action_config,
                n_sampled_actions=self.n_sampled_actions,
                max_turnover=self.max_turnover,
            )
            for name, action in candidates:
                actions.append(action.astype(np.float32))
                names.append(name)
                horizons.append(horizon)
                goals.append(goal_vector(self.cost_config, horizon))

        actions_np = np.asarray(actions, dtype=np.float32)
        horizons_np = np.asarray(horizons, dtype=np.int64)
        goals_np = np.asarray(goals, dtype=np.float32)
        portfolio_np = np.repeat(portfolio_state[None].astype(np.float32), len(actions), axis=0)
        costs: list[np.ndarray] = []
        outcomes: list[np.ndarray] = []
        for start in range(0, len(actions), self.chunk_size):
            end = min(start + self.chunk_size, len(actions))
            action_t = torch.tensor(actions_np[start:end], dtype=torch.float32, device=self.device)
            horizon_t = torch.tensor(horizons_np[start:end], dtype=torch.long, device=self.device)
            goal_t = torch.tensor(goals_np[start:end], dtype=torch.float32, device=self.device)
            portfolio_t = torch.tensor(portfolio_np[start:end], dtype=torch.float32, device=self.device)
            z_now_rep = z_market_now.repeat(end - start, 1, 1)
            z_market_future = self.model.market_jepa.predict_future(z_now_rep, horizon_t)
            z_portfolio_now = self.model.portfolio_encoder(portfolio_t)
            z_portfolio = self.model.transition(
                z_now_rep,
                z_market_future,
                z_portfolio_now,
                portfolio_t,
                action_t,
                horizon_t,
                goal_t,
            )
            outcome_hat, cost_hat = self.model.head(z_portfolio)
            costs.append(cost_hat.detach().cpu().numpy())
            outcomes.append(outcome_hat.detach().cpu().numpy())

        cost_np = np.nan_to_num(np.concatenate(costs), nan=np.inf, posinf=np.inf, neginf=np.inf)
        outcome_np = np.nan_to_num(np.concatenate(outcomes), nan=0.0, posinf=0.0, neginf=0.0)
        selected_idx = int(np.argmin(cost_np))
        return V5PlannerResult(
            action=actions_np[selected_idx],
            horizon=int(horizons_np[selected_idx]),
            predicted_cost=float(cost_np[selected_idx]),
            selected_name=names[selected_idx],
            predicted_outcome=outcome_np[selected_idx],
            candidate_names=names,
            candidate_horizons=[int(h) for h in horizons_np.tolist()],
            candidate_costs=cost_np,
        )


def portfolio_state_from_weights(weights: np.ndarray, equity: float, peak_equity: float, last_turnover: float) -> np.ndarray:
    return portfolio_state_features(weights, equity=equity, peak_equity=peak_equity, last_turnover=last_turnover)
