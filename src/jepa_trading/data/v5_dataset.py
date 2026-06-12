from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from jepa_trading.data.actions import portfolio_state_features, project_portfolio_action
from jepa_trading.data.dataset import MarketArrays
from jepa_trading.data.v2_dataset import PortfolioActionConfig, _portfolio_outcome


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


def realized_goal_cost(
    outcome: np.ndarray,
    action: np.ndarray,
    config: V5CostConfig,
    horizon: int,
) -> float:
    target_return = horizon_target_log_return(config.annual_return_target, horizon)
    pred_return = float(outcome[0])
    pred_drawdown = float(outcome[1])
    pred_vol = float(outcome[2])
    pred_turnover = float(outcome[3])
    pred_cost = float(outcome[4])
    concentration = float(np.sum(action[:-1] ** 2))
    cost = (
        config.return_shortfall_weight * max(0.0, target_return - pred_return)
        + config.drawdown_weight * max(0.0, abs(pred_drawdown) - abs(config.max_drawdown))
        + config.volatility_weight * max(0.0, pred_vol - config.max_annual_vol)
        + config.turnover_weight * max(0.0, pred_turnover - config.turnover_budget)
        + config.cost_weight * max(0.0, pred_cost - config.cost_budget)
        + config.concentration_weight * concentration
    )
    return float(np.clip(cost, 0.0, config.cost_clip))


def _normalize_long_only(weights: np.ndarray, valid: np.ndarray, max_weight: float) -> np.ndarray:
    out = np.zeros_like(weights, dtype=np.float32)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        out[-1] = 1.0
        return out
    clipped = np.minimum(np.clip(weights[idx], 0.0, None), max_weight)
    if clipped.sum() <= 1e-8:
        clipped = np.ones(len(idx), dtype=np.float32) / len(idx)
        clipped = np.minimum(clipped, max_weight)
    out[idx] = clipped * min(1.0, 1.0 / max(float(clipped.sum()), 1e-8))
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
    draw = rng.random()
    weights = np.zeros(n_assets + 1, dtype=np.float32)
    if draw < 0.30:
        weights[-1] = 1.0
        return "cash_state", weights
    if draw < 0.50:
        idx = np.where(valid)[0]
        if len(idx):
            weights[idx] = min(1.0 / len(idx), config.max_long_weight)
        weights[-1] = max(1.0 - float(weights[:-1].sum()), 0.0)
        return "equal_state", weights / max(float(weights.sum()), 1e-8)
    if draw < 0.70:
        inv = np.where(valid, 1.0 / np.maximum(np.nan_to_num(sigma, nan=np.inf), 1e-6), 0.0)
        if inv.sum() > 0:
            weights[:-1] = np.minimum(inv / inv.sum(), config.max_long_weight)
        weights[-1] = max(1.0 - float(weights[:-1].sum()), 0.0)
        return "vol_state", weights / max(float(weights.sum()), 1e-8)
    if draw < 0.85:
        mom = np.nan_to_num(recent_returns, nan=-np.inf)
        mom = np.where(valid, mom, -np.inf)
        selected = [i for i in np.argsort(mom)[-5:] if np.isfinite(mom[i]) and mom[i] > 0]
        if selected:
            weights[selected] = min(1.0 / len(selected), config.max_long_weight)
        weights[-1] = max(1.0 - float(weights[:-1].sum()), 0.0)
        return "momentum_state", weights / max(float(weights.sum()), 1e-8)
    raw = np.zeros(n_assets + 1, dtype=np.float32)
    idx = np.where(valid)[0]
    if len(idx):
        selected = rng.choice(idx, size=min(len(idx), max(2, int(np.sqrt(len(idx))) + 2)), replace=False)
        raw[selected] = rng.dirichlet(np.ones(len(selected)))
    return "sampled_state", _normalize_long_only(raw, valid, config.max_long_weight)


def v5_candidate_actions(
    rng: np.random.Generator,
    valid: np.ndarray,
    sigma: np.ndarray,
    recent_returns: np.ndarray,
    current_weights: np.ndarray,
    n_assets: int,
    config: PortfolioActionConfig,
    n_sampled_actions: int,
    max_turnover: float | None,
) -> list[tuple[str, np.ndarray]]:
    proposals: list[tuple[str, np.ndarray]] = []
    proposals.append(("hold", current_weights.copy()))
    cash = np.zeros(n_assets + 1, dtype=np.float32)
    cash[-1] = 1.0
    proposals.append(("cash", cash))
    derisk_25 = current_weights.copy()
    derisk_25[:-1] *= 0.75
    derisk_25[-1] = max(1.0 - float(np.abs(derisk_25[:-1]).sum()), 0.0)
    proposals.append(("derisk_25", derisk_25))
    derisk_50 = current_weights.copy()
    derisk_50[:-1] *= 0.50
    derisk_50[-1] = max(1.0 - float(np.abs(derisk_50[:-1]).sum()), 0.0)
    proposals.append(("derisk_50", derisk_50))

    eq = np.zeros(n_assets + 1, dtype=np.float32)
    idx = np.where(valid)[0]
    if len(idx):
        eq[idx] = min(1.0 / len(idx), config.max_long_weight)
    eq[-1] = max(1.0 - float(eq[:-1].sum()), 0.0)
    proposals.append(("equal_weight", eq))

    inv = np.where(valid, 1.0 / np.maximum(np.nan_to_num(sigma, nan=np.inf), 1e-6), 0.0)
    vt = np.zeros(n_assets + 1, dtype=np.float32)
    if inv.sum() > 0:
        vt[:-1] = np.minimum(inv / inv.sum(), config.max_long_weight)
    vt[-1] = max(1.0 - float(vt[:-1].sum()), 0.0)
    proposals.append(("vol_target", vt))

    mom = np.where(valid, np.nan_to_num(recent_returns, nan=-np.inf), -np.inf)
    selected = [i for i in np.argsort(mom)[-5:] if np.isfinite(mom[i]) and mom[i] > 0]
    mt = np.zeros(n_assets + 1, dtype=np.float32)
    if selected:
        mt[selected] = min(1.0 / len(selected), config.max_long_weight)
    mt[-1] = max(1.0 - float(mt[:-1].sum()), 0.0)
    proposals.append(("momentum_tilt", mt))

    for i in range(n_sampled_actions):
        raw = np.zeros(n_assets + 1, dtype=np.float32)
        if len(idx):
            selected = rng.choice(idx, size=min(len(idx), max(2, int(np.sqrt(len(idx))) + 2)), replace=False)
            raw[selected] = rng.dirichlet(np.ones(len(selected)))
        raw[-1] = max(1.0 - float(raw[:-1].sum()), 0.0)
        proposals.append((f"sampled_{i}", raw))

    projected = []
    for name, action in proposals:
        projected.append(
            (
                name,
                project_portfolio_action(
                    current_weights,
                    action,
                    valid,
                    mode=config.mode,
                    max_long_weight=config.max_long_weight,
                    max_short_weight=config.max_short_weight,
                    max_gross_exposure=config.max_gross_exposure,
                    max_net_exposure=config.max_net_exposure,
                    max_turnover=max_turnover,
                ),
            )
        )
    return projected


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
        self.max_turnover = max_turnover
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end_idx, horizon = self.samples[idx]
        rng = np.random.default_rng(self.seed + idx * 10007)
        ctx = slice(end_idx - self.lookback + 1, end_idx + 1)
        target_end = end_idx + horizon
        tgt = slice(target_end - self.lookback + 1, target_end + 1)
        valid = self.arrays.tradable[end_idx]
        n_assets = len(self.arrays.tickers)

        context = np.nan_to_num(self.arrays.features[ctx], nan=0.0).transpose(1, 0, 2)
        target = np.nan_to_num(self.arrays.features[tgt], nan=0.0).transpose(1, 0, 2)
        sigma_now = self.arrays.sigma[end_idx]
        recent_returns = np.nan_to_num(self.arrays.log_returns[max(0, end_idx - 20) : end_idx + 1], nan=0.0).sum(axis=0)
        current_kind, current_weights = sample_v5_portfolio_state(
            rng, valid, sigma_now, recent_returns, n_assets, self.action_config
        )
        future_returns = np.nan_to_num(self.arrays.log_returns[end_idx + 1 : target_end + 1], nan=0.0)
        candidates = v5_candidate_actions(
            rng,
            valid,
            sigma_now,
            recent_returns,
            current_weights,
            n_assets,
            self.action_config,
            self.n_sampled_actions,
            self.max_turnover,
        )
        names = [name for name, _ in candidates]
        actions = np.stack([action for _, action in candidates]).astype(np.float32)
        outcome_rows = []
        costs = []
        future_states = []
        for _, action in candidates:
            outcome = _portfolio_outcome(future_returns, current_weights, action, self.action_config)
            row = np.array(
                [
                    outcome["portfolio_log_return"],
                    outcome["portfolio_drawdown"],
                    outcome["portfolio_vol"],
                    outcome["portfolio_turnover"],
                    outcome["portfolio_cost"],
                    outcome["future_equity_ratio"],
                ],
                dtype=np.float32,
            )
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            outcome_rows.append(row)
            costs.append(realized_goal_cost(row, action, self.cost_config, horizon))
            future_equity = float(np.exp(np.clip(row[5], -3.0, 3.0)))
            future_states.append(
                portfolio_state_features(
                    action,
                    equity=future_equity,
                    peak_equity=max(1.0, future_equity),
                    last_turnover=float(row[3]),
                    recent_log_return=float(row[0]),
                    recent_vol=float(row[2]),
                )
            )

        outcomes = np.asarray(outcome_rows, dtype=np.float32)
        goal_costs = np.asarray(costs, dtype=np.float32)
        future_portfolio_states = np.asarray(future_states, dtype=np.float32)
        best_idx = int(np.argmin(goal_costs))
        portfolio_state = portfolio_state_features(current_weights)
        goal = goal_vector(self.cost_config, horizon)

        return {
            "context": torch.tensor(context, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "context_mask": torch.tensor(self.arrays.tradable[ctx].T, dtype=torch.bool),
            "target_mask": torch.tensor(self.arrays.tradable[tgt].T, dtype=torch.bool),
            "tradable_mask": torch.tensor(valid, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "portfolio_state": torch.tensor(portfolio_state, dtype=torch.float32),
            "goal": torch.tensor(goal, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "outcomes": torch.tensor(outcomes, dtype=torch.float32),
            "goal_costs": torch.tensor(goal_costs, dtype=torch.float32),
            "future_portfolio_states": torch.tensor(future_portfolio_states, dtype=torch.float32),
            "best_action_index": torch.tensor(best_idx, dtype=torch.long),
            "best_action": torch.tensor(actions[best_idx], dtype=torch.float32),
            "best_cost": torch.tensor(goal_costs[best_idx], dtype=torch.float32),
            "current_state_kind": current_kind,
            "candidate_names": names,
        }


def summarize_v5_batch(batch: dict[str, torch.Tensor]) -> dict[str, float | bool]:
    outcomes = batch["outcomes"].detach().cpu()
    costs = batch["goal_costs"].detach().cpu()
    return {
        "outcomes_finite": bool(torch.isfinite(outcomes).all().item()),
        "costs_finite": bool(torch.isfinite(costs).all().item()),
        "outcome_min": float(outcomes.min().item()),
        "outcome_max": float(outcomes.max().item()),
        "cost_min": float(costs.min().item()),
        "cost_max": float(costs.max().item()),
        "n_candidates": int(batch["actions"].shape[1]),
    }
