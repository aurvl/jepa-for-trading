from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jepa_trading.data.v2_dataset import PortfolioActionConfig, v3_candidate_actions
from jepa_trading.models.world_model_v2 import V2WorldModel


@dataclass
class V3PlannerResult:
    action: np.ndarray
    horizon: int
    score: float
    raw_score: float
    selected_name: str
    predicted_outcome: np.ndarray
    candidate_scores: np.ndarray
    candidate_names: list[str]
    candidate_horizons: list[int]
    risk_off: bool


class V3AbstentionPlanner:
    def __init__(
        self,
        model: V2WorldModel,
        action_config: PortfolioActionConfig,
        horizons: list[int],
        n_sampled_actions: int = 96,
        no_trade_margin: float = 0.002,
        cash_margin: float = 0.001,
        turnover_energy_penalty: float = 0.20,
        risk_off_drawdown_threshold: float = -0.03,
        chunk_size: int = 256,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.model = model.to(device).eval()
        self.action_config = action_config
        self.horizons = horizons
        self.n_sampled_actions = n_sampled_actions
        self.no_trade_margin = no_trade_margin
        self.cash_margin = cash_margin
        self.turnover_energy_penalty = turnover_energy_penalty
        self.risk_off_drawdown_threshold = risk_off_drawdown_threshold
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
    ) -> V3PlannerResult:
        context_t = torch.tensor(context[None], dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(context_mask[None], dtype=torch.bool, device=self.device)
        z_now = self.model.market_jepa.encode_context(context_t, mask_t)
        n_assets = len(tradable_mask)
        current_weights = np.asarray(portfolio_state[: n_assets + 1], dtype=np.float32)

        candidates: list[np.ndarray] = []
        names: list[str] = []
        horizons: list[int] = []
        for horizon in self.horizons:
            for name, action in v3_candidate_actions(
                rng=self.rng,
                valid=tradable_mask.astype(bool),
                sigma=sigma_now,
                current_weights=current_weights,
                n_assets=n_assets,
                config=self.action_config,
                n_sampled_actions=self.n_sampled_actions,
            ):
                candidates.append(action.astype(np.float32))
                names.append(name)
                horizons.append(horizon)

        actions_np = np.asarray(candidates, dtype=np.float32)
        horizons_np = np.asarray(horizons, dtype=np.int64)
        portfolio_np = np.repeat(portfolio_state[None].astype(np.float32), len(candidates), axis=0)
        outcomes: list[np.ndarray] = []
        raw_scores: list[np.ndarray] = []

        for start in range(0, len(candidates), self.chunk_size):
            end = min(start + self.chunk_size, len(candidates))
            action_t = torch.tensor(actions_np[start:end], dtype=torch.float32, device=self.device)
            horizon_t = torch.tensor(horizons_np[start:end], dtype=torch.long, device=self.device)
            portfolio_t = torch.tensor(portfolio_np[start:end], dtype=torch.float32, device=self.device)
            z_now_rep = z_now.repeat(end - start, 1, 1)
            z_future = self.model.market_jepa.predict_future(z_now_rep, horizon_t)
            outcome = self.model.outcome_model(z_now_rep, z_future, portfolio_t, action_t, horizon_t)
            energy = self.model.energy_model(outcome, portfolio_t, action_t)
            outcomes.append(outcome.detach().cpu().numpy())
            raw_scores.append(energy.detach().cpu().numpy())

        outcome_np = np.concatenate(outcomes, axis=0)
        raw_score_np = np.concatenate(raw_scores, axis=0)
        turnover = np.abs(actions_np - current_weights[None]).sum(axis=1)
        scores = raw_score_np - self.turnover_energy_penalty * turnover

        best_idx = int(np.argmax(scores))
        hold_idx = self._best_named_idx(scores, names, "hold")
        cash_idx = self._best_named_idx(scores, names, "cash")
        derisk_idx = self._best_named_idx(scores, names, "derisk")
        selected_idx = best_idx
        risk_off = bool(outcome_np[best_idx, 1] <= self.risk_off_drawdown_threshold)

        if scores[best_idx] <= scores[hold_idx] + self.no_trade_margin:
            selected_idx = hold_idx
        elif risk_off:
            defensive_idx = cash_idx if scores[cash_idx] >= scores[derisk_idx] + self.cash_margin else derisk_idx
            if scores[defensive_idx] >= scores[best_idx] - self.cash_margin:
                selected_idx = defensive_idx

        return V3PlannerResult(
            action=actions_np[selected_idx],
            horizon=int(horizons_np[selected_idx]),
            score=float(scores[selected_idx]),
            raw_score=float(raw_score_np[selected_idx]),
            selected_name=names[selected_idx],
            predicted_outcome=outcome_np[selected_idx],
            candidate_scores=scores,
            candidate_names=names,
            candidate_horizons=[int(h) for h in horizons_np.tolist()],
            risk_off=risk_off,
        )

    @staticmethod
    def _best_named_idx(scores: np.ndarray, names: list[str], name: str) -> int:
        idx = [i for i, candidate_name in enumerate(names) if candidate_name == name]
        if not idx:
            return int(np.argmax(scores))
        local = int(np.argmax(scores[idx]))
        return idx[local]
