from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jepa_trading.data.v2_dataset import PortfolioActionConfig, _sample_weights
from jepa_trading.models.world_model_v2 import V2WorldModel


@dataclass
class PlannerResult:
    action: np.ndarray
    horizon: int
    score: float
    predicted_outcome: np.ndarray
    candidate_scores: np.ndarray


class V2ImaginationPlanner:
    def __init__(
        self,
        model: V2WorldModel,
        action_config: PortfolioActionConfig,
        horizons: list[int],
        n_action_samples: int = 128,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.model = model.to(device).eval()
        self.action_config = action_config
        self.horizons = horizons
        self.n_action_samples = n_action_samples
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def choose_action(
        self,
        context: np.ndarray,
        context_mask: np.ndarray,
        portfolio_state: np.ndarray,
        tradable_mask: np.ndarray,
    ) -> PlannerResult:
        context_t = torch.tensor(context[None], dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(context_mask[None], dtype=torch.bool, device=self.device)
        z_now = self.model.market_jepa.encode_context(context_t, mask_t)
        candidates: list[np.ndarray] = []
        horizons: list[int] = []
        for horizon in self.horizons:
            for _ in range(self.n_action_samples):
                action = _sample_weights(
                    self.rng,
                    tradable_mask.astype(bool),
                    len(tradable_mask),
                    self.action_config,
                )
                candidates.append(action)
                horizons.append(horizon)

        action_t = torch.tensor(np.asarray(candidates), dtype=torch.float32, device=self.device)
        horizon_t = torch.tensor(horizons, dtype=torch.long, device=self.device)
        portfolio_t = torch.tensor(np.repeat(portfolio_state[None], len(candidates), axis=0), dtype=torch.float32, device=self.device)
        z_now_rep = z_now.repeat(len(candidates), 1, 1)
        z_future = self.model.market_jepa.predict_future(z_now_rep, horizon_t)
        outcome = self.model.outcome_model(z_now_rep, z_future, portfolio_t, action_t, horizon_t)
        energy = self.model.energy_model(outcome, portfolio_t, action_t)
        scores = energy.detach().cpu().numpy()
        best = int(np.argmax(scores))
        return PlannerResult(
            action=candidates[best],
            horizon=horizons[best],
            score=float(scores[best]),
            predicted_outcome=outcome[best].detach().cpu().numpy(),
            candidate_scores=scores,
        )

