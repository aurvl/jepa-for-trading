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
        cash_bootstrap_min_return: float = -0.03,
        cash_bootstrap_max_drawdown: float = -0.12,
        cash_bootstrap_score_tolerance: float = 0.15,
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
        self.cash_bootstrap_min_return = cash_bootstrap_min_return
        self.cash_bootstrap_max_drawdown = cash_bootstrap_max_drawdown
        self.cash_bootstrap_score_tolerance = cash_bootstrap_score_tolerance
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
        outcome_np = self._sanitize_outcomes(outcome_np, actions_np, current_weights, turnover)
        scores = self._transparent_scores(outcome_np, raw_score_np, turnover)

        hold_idx = self._best_named_idx(scores, names, "hold")
        cash_idx = self._best_named_idx(scores, names, "cash")
        derisk_idx = self._best_named_idx(scores, names, "derisk")
        best_idx = int(np.argmax(scores))
        best_risk_idx = self._best_risk_idx(scores, names)
        hold_outcome = outcome_np[hold_idx]
        is_all_cash = bool(current_weights[-1] >= 0.98 and np.abs(current_weights[:-1]).sum() <= 0.02)
        hard_risk_off = bool(
            not is_all_cash
            and (
                hold_outcome[1] <= self.hard_risk_off_drawdown
                or hold_outcome[0] <= self.hard_risk_off_return
            )
        )
        due_rebalance = (step_count % self.rebalance_every == 0) or (days_since_trade >= self.force_recheck_after)

        selected_idx = hold_idx
        selected_reason = "hold_default"
        if is_all_cash and self._cash_bootstrap_ok(scores, outcome_np, hold_idx, best_risk_idx):
            selected_idx = best_risk_idx
            selected_reason = "cash_bootstrap_best_risk"
        elif hard_risk_off:
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

    def _sanitize_outcomes(
        self,
        outcomes: np.ndarray,
        actions: np.ndarray,
        current_weights: np.ndarray,
        turnover: np.ndarray,
    ) -> np.ndarray:
        out = np.nan_to_num(outcomes.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        # Financial constraints the neural head should not be allowed to violate.
        out[:, 1] = np.minimum(out[:, 1], 0.0)
        out[:, 2] = np.maximum(out[:, 2], 0.0)
        out[:, 3] = np.maximum(out[:, 3], 0.0)
        out[:, 4] = np.maximum(out[:, 4], 0.0)

        gross = np.abs(actions[:, :-1]).sum(axis=1)
        cash_like = gross <= 1e-6
        if cash_like.any():
            cost = np.clip(turnover[cash_like] * self.action_config.transaction_cost_bps / 10000.0, 0.0, 1.0)
            log_after_cost = np.log(np.maximum(1.0 - cost, 1e-6))
            out[cash_like, 0] = log_after_cost
            out[cash_like, 1] = np.minimum(log_after_cost, 0.0)
            out[cash_like, 2] = 0.0
            out[cash_like, 3] = turnover[cash_like]
            out[cash_like, 4] = cost
            out[cash_like, 5] = log_after_cost
        return out

    def _cash_bootstrap_ok(
        self,
        scores: np.ndarray,
        outcomes: np.ndarray,
        hold_idx: int,
        best_risk_idx: int,
    ) -> bool:
        risk_outcome = outcomes[best_risk_idx]
        return bool(
            scores[best_risk_idx] >= scores[hold_idx] - self.cash_bootstrap_score_tolerance
            and risk_outcome[0] >= self.cash_bootstrap_min_return
            and risk_outcome[1] >= self.cash_bootstrap_max_drawdown
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
