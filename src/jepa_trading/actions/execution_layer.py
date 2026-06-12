from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jepa_trading.actions.primitive_actions import PrimitiveAction, primitive_from_vector
from jepa_trading.data.actions import project_portfolio_action


@dataclass(frozen=True)
class ExecutionConstraints:
    mode: str = "long_only"
    max_long_weight: float = 0.15
    max_short_weight: float = 0.05
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_turnover: float | None = 0.40
    transaction_cost_bps: float = 5.0
    borrow_cost_bps: float = 2.0


class PrimitiveExecutionLayer:
    def __init__(self, constraints: ExecutionConstraints) -> None:
        self.constraints = constraints

    def primitive_to_proposed_weights(
        self,
        action: PrimitiveAction | np.ndarray,
        current_weights: np.ndarray,
        tradable_mask: np.ndarray,
    ) -> np.ndarray:
        current = np.asarray(current_weights, dtype=np.float32)
        valid = np.asarray(tradable_mask, dtype=bool)
        n_assets = len(valid)
        if not isinstance(action, PrimitiveAction):
            action = primitive_from_vector(np.asarray(action, dtype=np.float32), n_assets)

        scores = np.asarray(action.asset_scores, dtype=np.float32).copy()
        scores = np.where(valid, scores, -np.inf)
        proposed = current.copy()
        current_gross = float(np.abs(current[:-1]).sum())
        gross_target = np.clip(current_gross + action.gross_exposure_delta, 0.0, self.constraints.max_gross_exposure)
        gross_target = min(float(action.risk_budget), float(gross_target))
        cash_target = np.clip(current[-1] + action.cash_target_delta, 0.0, 1.0)
        gross_target = min(gross_target, max(0.0, 1.0 - cash_target))

        if np.isfinite(scores).any() and gross_target > 1e-8:
            centered = scores - np.nanmax(scores[np.isfinite(scores)])
            exp_scores = np.where(np.isfinite(centered), np.exp(centered), 0.0)
            if exp_scores.sum() <= 1e-8:
                exp_scores = valid.astype(np.float32)
            asset_weights = exp_scores / max(float(exp_scores.sum()), 1e-8) * gross_target
            if self.constraints.mode == "long_only":
                proposed[:-1] = asset_weights
            else:
                bias = np.clip(action.long_short_bias, -1.0, 1.0)
                signed = asset_weights * np.sign(scores + 1e-8)
                proposed[:-1] = bias * asset_weights + (1.0 - abs(bias)) * signed
        else:
            proposed[:-1] = 0.0
        proposed[-1] = max(1.0 - float(np.abs(proposed[:-1]).sum()), 0.0)

        intensity = float(np.clip(action.rebalance_intensity, 0.0, 1.0))
        proposed = current + intensity * (proposed - current)
        return proposed.astype(np.float32)

    def execute(
        self,
        action: PrimitiveAction | np.ndarray,
        current_weights: np.ndarray,
        tradable_mask: np.ndarray,
    ) -> np.ndarray:
        proposed = self.primitive_to_proposed_weights(action, current_weights, tradable_mask)
        c = self.constraints
        return project_portfolio_action(
            current_weights,
            proposed,
            tradable_mask,
            mode=c.mode,
            max_long_weight=c.max_long_weight,
            max_short_weight=c.max_short_weight,
            max_gross_exposure=c.max_gross_exposure,
            max_net_exposure=c.max_net_exposure,
            max_turnover=c.max_turnover,
        )

    def transaction_cost(self, current_weights: np.ndarray, executable_weights: np.ndarray) -> float:
        turnover = float(np.abs(np.asarray(executable_weights) - np.asarray(current_weights)).sum())
        borrow = float(np.abs(np.minimum(np.asarray(executable_weights)[:-1], 0.0)).sum())
        return turnover * self.constraints.transaction_cost_bps / 10000.0 + borrow * self.constraints.borrow_cost_bps / 10000.0
