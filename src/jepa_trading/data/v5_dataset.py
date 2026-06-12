from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from jepa_trading.actions import ExecutionConstraints, PrimitiveExecutionLayer, sample_primitive_actions
from jepa_trading.data.actions import portfolio_state_features
from jepa_trading.data.dataset import MarketArrays
from jepa_trading.data.v2_dataset import PortfolioActionConfig


V5_OUTCOME_KEYS = (
    "realized_return",
    "drawdown",
    "volatility",
    "turnover",
    "transaction_cost",
    "cvar",
    "prob_loss",
    "prob_drawdown_breach",
    "future_equity_ratio",
)
V5_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass
class V5CostConfig:
    annual_return_target: float = 0.10
    max_drawdown: float = -0.12
    max_annual_vol: float = 0.15
    turnover_budget: float = 0.20
    cost_budget: float = 0.01
    return_shortfall_weight: float = 1.0
    drawdown_weight: float = 2.0
    volatility_weight: float = 0.4
    turnover_weight: float = 0.2
    cost_weight: float = 2.0
    cvar_weight: float = 0.8
    loss_prob_weight: float = 0.4
    breach_prob_weight: float = 0.6
    concentration_weight: float = 0.05
    cost_clip: float = 3.0


def horizon_target_log_return(annual_return_target: float, horizon: int) -> float:
    return float(np.log1p(annual_return_target) * horizon / 252.0)


def goal_vector(config: V5CostConfig, horizon: int) -> np.ndarray:
    return np.array(
        [
            float(horizon),
            horizon_target_log_return(config.annual_return_target, horizon),
            float(config.max_drawdown),
            float(config.max_annual_vol),
            float(config.turnover_budget),
            float(config.cost_budget),
        ],
        dtype=np.float32,
    )


def _constraints_from_portfolio_config(
    config: PortfolioActionConfig,
    max_turnover: float | None,
) -> ExecutionConstraints:
    return ExecutionConstraints(
        mode=config.mode,
        max_long_weight=config.max_long_weight,
        max_short_weight=config.max_short_weight,
        max_gross_exposure=config.max_gross_exposure,
        max_net_exposure=config.max_net_exposure,
        max_turnover=max_turnover,
        transaction_cost_bps=config.transaction_cost_bps,
        borrow_cost_bps=config.borrow_cost_bps,
    )


def _project_long_only(weights: np.ndarray, valid: np.ndarray, max_weight: float) -> np.ndarray:
    out = np.zeros(len(valid) + 1, dtype=np.float32)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        out[-1] = 1.0
        return out
    raw = np.asarray(weights, dtype=np.float32)[: len(valid)]
    raw = np.where(valid, np.clip(raw, 0.0, max_weight), 0.0)
    if raw.sum() <= 1e-8:
        raw[idx] = min(1.0 / len(idx), max_weight)
    gross = min(float(raw.sum()), 1.0)
    if gross > 1e-8:
        out[:-1] = raw * min(1.0, gross / max(float(raw.sum()), 1e-8))
    out[-1] = max(1.0 - float(out[:-1].sum()), 0.0)
    return out / max(float(out.sum()), 1e-8)


def sample_v5_portfolio_state(
    rng: np.random.Generator,
    valid: np.ndarray,
    sigma: np.ndarray,
    recent_returns: np.ndarray,
    n_assets: int,
    config: PortfolioActionConfig,
) -> tuple[str, np.ndarray]:
    weights = np.zeros(n_assets + 1, dtype=np.float32)
    idx = np.where(valid)[0]
    draw = rng.random()
    if draw < 0.25 or len(idx) == 0:
        weights[-1] = 1.0
        return "cash_state", weights
    if draw < 0.55:
        raw = np.zeros(n_assets, dtype=np.float32)
        selected = rng.choice(idx, size=min(len(idx), max(2, int(np.sqrt(len(idx))) + 1)), replace=False)
        raw[selected] = rng.dirichlet(np.ones(len(selected)))
        return "random_state", _project_long_only(raw, valid, config.max_long_weight)
    if draw < 0.75:
        inv_sigma = np.where(valid, 1.0 / np.maximum(np.nan_to_num(sigma, nan=np.inf), 1e-6), 0.0)
        return "low_vol_state", _project_long_only(inv_sigma, valid, config.max_long_weight)

    signed_strength = np.where(valid, np.maximum(np.nan_to_num(recent_returns, nan=0.0), 0.0), 0.0)
    return "positive_return_state", _project_long_only(signed_strength, valid, config.max_long_weight)


def _portfolio_path(
    future_returns: np.ndarray,
    current_weights: np.ndarray,
    executable_weights: np.ndarray,
    execution_layer: PrimitiveExecutionLayer,
) -> tuple[np.ndarray, float]:
    returns = np.nan_to_num(future_returns, nan=0.0, posinf=0.0, neginf=0.0)
    asset_weights = executable_weights[:-1].astype(np.float32)
    daily_log_returns = returns @ asset_weights
    cost = execution_layer.transaction_cost(current_weights, executable_weights)
    if len(daily_log_returns):
        daily_log_returns = daily_log_returns.copy()
        daily_log_returns[0] -= cost
    equity_path = np.exp(np.cumsum(daily_log_returns))
    return equity_path.astype(np.float32), float(cost)


def realized_outcome_from_action(
    future_returns: np.ndarray,
    current_weights: np.ndarray,
    executable_weights: np.ndarray,
    execution_layer: PrimitiveExecutionLayer,
    drawdown_breach: float,
) -> np.ndarray:
    equity_path, cost = _portfolio_path(future_returns, current_weights, executable_weights, execution_layer)
    if len(equity_path) == 0:
        return np.zeros(len(V5_OUTCOME_KEYS), dtype=np.float32)
    peak = np.maximum.accumulate(equity_path)
    drawdown = equity_path / np.maximum(peak, 1e-8) - 1.0
    daily = np.diff(np.concatenate([[1.0], equity_path]))
    realized_return = float(np.log(max(float(equity_path[-1]), 1e-8)))
    volatility = float(np.std(daily) * np.sqrt(252.0)) if len(daily) > 1 else 0.0
    turnover = float(np.abs(executable_weights - current_weights).sum())
    cvar = float(np.mean(np.sort(daily)[: max(1, int(np.ceil(0.05 * len(daily))))]))
    row = np.array(
        [
            realized_return,
            float(np.min(drawdown)),
            volatility,
            turnover,
            cost,
            cvar,
            float(realized_return < 0.0),
            float(np.min(drawdown) < drawdown_breach),
            float(equity_path[-1]),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def realized_goal_cost(
    outcome: np.ndarray,
    executable_action: np.ndarray,
    config: V5CostConfig,
    horizon: int,
) -> float:
    target_return = horizon_target_log_return(config.annual_return_target, horizon)
    realized_return, drawdown, volatility, turnover, cost, cvar, prob_loss, breach, _ = outcome
    concentration = float(np.sum(np.square(executable_action[:-1])))
    energy = (
        config.return_shortfall_weight * max(0.0, target_return - float(realized_return))
        + config.drawdown_weight * max(0.0, abs(float(drawdown)) - abs(config.max_drawdown))
        + config.volatility_weight * max(0.0, float(volatility) - config.max_annual_vol)
        + config.turnover_weight * max(0.0, float(turnover) - config.turnover_budget)
        + config.cost_weight * max(0.0, float(cost) - config.cost_budget)
        + config.cvar_weight * max(0.0, -float(cvar))
        + config.loss_prob_weight * float(prob_loss)
        + config.breach_prob_weight * float(breach)
        + config.concentration_weight * concentration
    )
    return float(np.clip(energy, 0.0, config.cost_clip))


class V5ActionWorldModelDataset(Dataset):
    def __init__(
        self,
        arrays: MarketArrays,
        split_dates: pd.DatetimeIndex,
        lookback: int,
        horizons: list[int],
        action_config: PortfolioActionConfig,
        cost_config: V5CostConfig,
        seed: int = 42,
        n_sampled_actions: int = 8,
        max_turnover: float | None = 0.40,
    ) -> None:
        self.arrays = arrays
        self.lookback = lookback
        self.horizons = horizons
        self.action_config = action_config
        self.cost_config = cost_config
        self.seed = seed
        self.n_sampled_actions = n_sampled_actions
        self.execution_layer = PrimitiveExecutionLayer(_constraints_from_portfolio_config(action_config, max_turnover))
        allowed = set(pd.DatetimeIndex(split_dates))
        self.samples: list[tuple[int, int]] = []
        for end_idx, date in enumerate(arrays.dates):
            if date not in allowed or end_idx < lookback - 1:
                continue
            for horizon in horizons:
                target_end = end_idx + horizon
                if target_end >= len(arrays.dates):
                    continue
                if arrays.dates[target_end] not in allowed:
                    continue
                self.samples.append((end_idx, horizon))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | list[str]]:
        end_idx, horizon = self.samples[idx]
        rng = np.random.default_rng(self.seed + idx * 10007)
        ctx = slice(end_idx - self.lookback + 1, end_idx + 1)
        target_end = end_idx + horizon
        tgt = slice(target_end - self.lookback + 1, target_end + 1)
        valid = self.arrays.tradable[end_idx].astype(bool)
        n_assets = len(self.arrays.tickers)

        context = np.nan_to_num(self.arrays.features[ctx], nan=0.0).transpose(1, 0, 2)
        target = np.nan_to_num(self.arrays.features[tgt], nan=0.0).transpose(1, 0, 2)
        sigma_now = self.arrays.sigma[end_idx]
        recent_returns = np.nan_to_num(self.arrays.log_returns[max(0, end_idx - 20) : end_idx + 1], nan=0.0).sum(axis=0)
        current_kind, current_weights = sample_v5_portfolio_state(
            rng, valid, sigma_now, recent_returns, n_assets, self.action_config
        )
        future_returns = np.nan_to_num(self.arrays.log_returns[end_idx + 1 : target_end + 1], nan=0.0)

        primitive_batch = sample_primitive_actions(
            rng,
            n_assets=n_assets,
            horizons=[horizon],
            n_actions=self.n_sampled_actions,
        )
        executable_actions = []
        outcomes = []
        costs = []
        future_states = []
        for primitive in primitive_batch.vectors:
            executable = self.execution_layer.execute(primitive, current_weights, valid)
            outcome = realized_outcome_from_action(
                future_returns,
                current_weights,
                executable,
                self.execution_layer,
                drawdown_breach=self.cost_config.max_drawdown,
            )
            future_equity = float(outcome[V5_OUTCOME_KEYS.index("future_equity_ratio")])
            future_states.append(
                portfolio_state_features(
                    executable,
                    equity=future_equity,
                    peak_equity=max(1.0, future_equity),
                    last_turnover=float(outcome[V5_OUTCOME_KEYS.index("turnover")]),
                    recent_log_return=float(outcome[V5_OUTCOME_KEYS.index("realized_return")]),
                    recent_vol=float(outcome[V5_OUTCOME_KEYS.index("volatility")]),
                )
            )
            executable_actions.append(executable)
            outcomes.append(outcome)
            costs.append(realized_goal_cost(outcome, executable, self.cost_config, horizon))

        executable_actions_np = np.asarray(executable_actions, dtype=np.float32)
        outcomes_np = np.asarray(outcomes, dtype=np.float32)
        costs_np = np.asarray(costs, dtype=np.float32)
        best_idx = int(np.argmin(costs_np))

        return {
            "context": torch.tensor(context, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "context_mask": torch.tensor(self.arrays.tradable[ctx].T, dtype=torch.bool),
            "target_mask": torch.tensor(self.arrays.tradable[tgt].T, dtype=torch.bool),
            "tradable_mask": torch.tensor(valid, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "portfolio_state": torch.tensor(portfolio_state_features(current_weights), dtype=torch.float32),
            "goal": torch.tensor(goal_vector(self.cost_config, horizon), dtype=torch.float32),
            "primitive_actions": torch.tensor(primitive_batch.vectors, dtype=torch.float32),
            "executable_actions": torch.tensor(executable_actions_np, dtype=torch.float32),
            "realized_outcomes": torch.tensor(outcomes_np, dtype=torch.float32),
            "realized_costs": torch.tensor(costs_np, dtype=torch.float32),
            "future_portfolio_states": torch.tensor(np.asarray(future_states, dtype=np.float32), dtype=torch.float32),
            "best_action_index": torch.tensor(best_idx, dtype=torch.long),
            "current_state_kind": current_kind,
            "candidate_names": primitive_batch.names,
        }


def summarize_v5_batch(batch: dict[str, torch.Tensor]) -> dict[str, float | bool]:
    outcomes = batch["realized_outcomes"].detach().cpu()
    costs = batch["realized_costs"].detach().cpu()
    primitive_actions = batch["primitive_actions"].detach().cpu()
    executable_actions = batch["executable_actions"].detach().cpu()
    return {
        "outcomes_finite": bool(torch.isfinite(outcomes).all().item()),
        "costs_finite": bool(torch.isfinite(costs).all().item()),
        "primitive_actions_finite": bool(torch.isfinite(primitive_actions).all().item()),
        "executable_actions_finite": bool(torch.isfinite(executable_actions).all().item()),
        "outcome_min": float(outcomes.min().item()),
        "outcome_max": float(outcomes.max().item()),
        "cost_min": float(costs.min().item()),
        "cost_max": float(costs.max().item()),
        "n_candidates": int(primitive_actions.shape[1]),
    }
