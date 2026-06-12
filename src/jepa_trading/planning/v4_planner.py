from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jepa_trading.data.v2_dataset import PortfolioActionConfig, v3_candidate_actions
from jepa_trading.models.world_model_v2 import V2WorldModel


@dataclass
class V4PlannerResult:
    action: np.ndarray
    horizon: int
    score: float
    raw_energy: float
    selected_name: str
    selected_reason: str
    predicted_outcome: np.ndarray
    candidate_scores: np.ndarray
    candidate_names: list[str]
    candidate_horizons: list[int]
    hard_risk_off: bool
    best_name: str
    best_score: float
    hold_score: float
    cash_score: float
    derisk_score: float
    best_risk_name: str
    best_risk_score: float
    hold_predicted_outcome: np.ndarray
    best_risk_predicted_outcome: np.ndarray


class V4RiskOffPlanner:
    def __init__(
        self,
        model: V2WorldModel,
        action_config: PortfolioActionConfig,
        horizons: list[int],
        n_sampled_actions: int = 128,
        model_score_weight: float = 0.25,
        return_weight: float = 1.0,
        drawdown_penalty: float = 2.0,
        volatility_penalty: float = 0.12,
        turnover_penalty: float = 0.01,
        hard_risk_off_drawdown: float = -0.06,
        hard_risk_off_return: float = -0.03,
        min_action_advantage: float = 0.005,
        rebalance_every: int = 5,
        force_recheck_after: int = 20,
        chunk_size: int = 256,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.model = model.to(device).eval()
        self.action_config = action_config
        self.horizons = horizons
        self.n_sampled_actions = n_sampled_actions
        self.model_score_weight = model_score_weight
        self.return_weight = return_weight
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.turnover_penalty = turnover_penalty
        self.hard_risk_off_drawdown = hard_risk_off_drawdown
        self.hard_risk_off_return = hard_risk_off_return
        self.min_action_advantage = min_action_advantage
        self.rebalance_every = max(1, rebalance_every)
        self.force_recheck_after = max(1, force_recheck_after)
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
        step_count: int = 0,
        days_since_trade: int = 999,
    ) -> V4PlannerResult:
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
        scores = self._transparent_scores(outcome_np, raw_score_np, turnover)

        hold_idx = self._best_named_idx(scores, names, "hold")
        cash_idx = self._best_named_idx(scores, names, "cash")
        derisk_idx = self._best_named_idx(scores, names, "derisk")
        best_idx = int(np.argmax(scores))
        best_risk_idx = self._best_risk_idx(scores, names)
        hold_outcome = outcome_np[hold_idx]
        hard_risk_off = bool(
            hold_outcome[1] <= self.hard_risk_off_drawdown
            or hold_outcome[0] <= self.hard_risk_off_return
        )
        due_rebalance = (step_count % self.rebalance_every == 0) or (days_since_trade >= self.force_recheck_after)

        selected_idx = hold_idx
        selected_reason = "hold_default"
        if hard_risk_off:
            selected_idx = cash_idx if scores[cash_idx] >= scores[derisk_idx] else derisk_idx
            selected_reason = "hard_risk_off_defensive"
        elif scores[best_idx] >= scores[hold_idx] + self.min_action_advantage:
            selected_idx = best_idx
            selected_reason = "best_advantage"
        elif due_rebalance and scores[best_risk_idx] >= scores[hold_idx] - self.min_action_advantage:
            selected_idx = best_risk_idx
            selected_reason = "scheduled_rebalance_best_risk"
        else:
            selected_idx = hold_idx
            selected_reason = "hold_until_signal"

        return V4PlannerResult(
            action=actions_np[selected_idx],
            horizon=int(horizons_np[selected_idx]),
            score=float(scores[selected_idx]),
            raw_energy=float(raw_score_np[selected_idx]),
            selected_name=names[selected_idx],
            selected_reason=selected_reason,
            predicted_outcome=outcome_np[selected_idx],
            candidate_scores=scores,
            candidate_names=names,
            candidate_horizons=[int(h) for h in horizons_np.tolist()],
            hard_risk_off=hard_risk_off,
            best_name=names[best_idx],
            best_score=float(scores[best_idx]),
            hold_score=float(scores[hold_idx]),
            cash_score=float(scores[cash_idx]),
            derisk_score=float(scores[derisk_idx]),
            best_risk_name=names[best_risk_idx],
            best_risk_score=float(scores[best_risk_idx]),
            hold_predicted_outcome=hold_outcome,
            best_risk_predicted_outcome=outcome_np[best_risk_idx],
        )

    def _transparent_scores(self, outcomes: np.ndarray, raw_energy: np.ndarray, turnover: np.ndarray) -> np.ndarray:
        return (
            self.model_score_weight * raw_energy
            + self.return_weight * outcomes[:, 0]
            - self.drawdown_penalty * np.abs(np.minimum(outcomes[:, 1], 0.0))
            - self.volatility_penalty * outcomes[:, 2]
            - self.turnover_penalty * turnover
        )

    @staticmethod
    def _best_named_idx(scores: np.ndarray, names: list[str], name: str) -> int:
        idx = [i for i, candidate_name in enumerate(names) if candidate_name == name]
        if not idx:
            return int(np.argmax(scores))
        return idx[int(np.argmax(scores[idx]))]

    @staticmethod
    def _best_risk_idx(scores: np.ndarray, names: list[str]) -> int:
        defensive = {"hold", "cash", "derisk"}
        idx = [i for i, candidate_name in enumerate(names) if candidate_name not in defensive]
        if not idx:
            return int(np.argmax(scores))
        return idx[int(np.argmax(scores[idx]))]
