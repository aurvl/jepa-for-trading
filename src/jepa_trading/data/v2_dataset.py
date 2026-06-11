from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from jepa_trading.data.dataset import MarketArrays


@dataclass
class PortfolioActionConfig:
    mode: str = "long_only"
    cash_initial: float = 1000.0
    transaction_cost_bps: float = 5.0
    max_long_weight: float = 0.15
    max_short_weight: float = 0.05
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    borrow_cost_bps: float = 2.0
    n_action_samples: int = 4
    max_abs_daily_log_return: float = 0.35
    max_daily_gross: float = 2.0
    min_daily_gross: float = 0.05


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
    scale = min(1.0, 1.0 / clipped.sum())
    out[idx] = clipped * scale
    out[-1] = max(1.0 - out[:-1].sum(), 0.0)
    total = out.sum()
    return out / max(total, 1e-8)


def _sample_weights(
    rng: np.random.Generator,
    valid: np.ndarray,
    n_assets: int,
    config: PortfolioActionConfig,
) -> np.ndarray:
    weights = np.zeros(n_assets + 1, dtype=np.float32)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        weights[-1] = 1.0
        return weights

    if config.mode == "long_only":
        raw = rng.dirichlet(np.ones(len(valid_idx), dtype=np.float32))
        weights[valid_idx] = raw
        return _normalize_long_only(weights, valid, config.max_long_weight)

    if config.mode in {"long_short", "market_neutral"}:
        k = min(len(valid_idx), max(2, int(np.sqrt(len(valid_idx))) + 2))
        selected = rng.choice(valid_idx, size=k, replace=False)
        signs = rng.choice([-1.0, 1.0], size=k)
        if config.mode == "market_neutral":
            half = k // 2
            signs[:half] = 1.0
            signs[half:] = -1.0
            rng.shuffle(signs)
        raw = rng.dirichlet(np.ones(k, dtype=np.float32))
        signed = raw * signs
        signed = np.where(
            signed >= 0,
            np.minimum(signed, config.max_long_weight),
            np.maximum(signed, -config.max_short_weight),
        )
        weights[selected] = signed
        gross = np.abs(weights[:-1]).sum()
        if gross > config.max_gross_exposure:
            weights[:-1] *= config.max_gross_exposure / gross
        if config.mode == "market_neutral":
            weights[:-1] -= weights[:-1].mean()
        net = weights[:-1].sum()
        if abs(net) > config.max_net_exposure:
            weights[:-1] *= config.max_net_exposure / abs(net)
        weights[-1] = max(1.0 - np.abs(weights[:-1]).sum(), 0.0)
        return weights.astype(np.float32)

    raise ValueError(f"Unknown portfolio mode: {config.mode}")


def _portfolio_outcome(
    future_log_returns: np.ndarray,
    current_weights: np.ndarray,
    action_weights: np.ndarray,
    config: PortfolioActionConfig,
) -> dict[str, float]:
    turnover = float(np.abs(action_weights - current_weights).sum())
    cost = turnover * config.transaction_cost_bps / 10000.0
    borrow = float(np.abs(np.minimum(action_weights[:-1], 0.0)).sum() * config.borrow_cost_bps / 10000.0)

    safe_log_returns = np.nan_to_num(future_log_returns, nan=0.0, posinf=0.0, neginf=0.0)
    safe_log_returns = np.clip(
        safe_log_returns,
        -config.max_abs_daily_log_return,
        config.max_abs_daily_log_return,
    )
    simple_returns = np.expm1(safe_log_returns)
    portfolio_simple_returns = simple_returns @ action_weights[:-1]
    daily_gross = np.clip(
        1.0 + portfolio_simple_returns,
        config.min_daily_gross,
        config.max_daily_gross,
    )
    daily_log_returns = np.log(daily_gross)
    log_cost = np.log(max(1.0 - cost - borrow, 1e-6))
    log_equity_path = np.cumsum(daily_log_returns) + log_cost
    equity_path = np.exp(np.clip(log_equity_path, -20.0, 20.0))
    running_max = np.maximum.accumulate(equity_path)
    drawdown = equity_path / np.maximum(running_max, 1e-12) - 1.0
    log_return = float(np.clip(log_equity_path[-1], -2.0, 2.0))
    realized_vol = float(np.clip(np.std(daily_log_returns) * np.sqrt(252), 0.0, 5.0))
    concentration = float(np.sum(action_weights[:-1] ** 2))
    downside = float(np.clip(np.minimum(drawdown.min(), 0.0), -1.0, 0.0))
    utility = log_return - 0.5 * abs(downside) - 0.05 * turnover - 0.02 * concentration - cost - borrow
    utility = float(np.clip(utility, -3.0, 3.0))
    future_log_equity = float(np.clip(log_equity_path[-1], -3.0, 3.0))
    outcome = {
        "portfolio_log_return": log_return,
        "portfolio_drawdown": downside,
        "portfolio_vol": realized_vol,
        "portfolio_turnover": float(np.clip(turnover, 0.0, 3.0)),
        "portfolio_cost": float(np.clip(cost + borrow, 0.0, 1.0)),
        "portfolio_utility": utility,
        "future_equity_ratio": future_log_equity,
    }
    if not np.isfinite(np.array(list(outcome.values()), dtype=np.float64)).all():
        return {
            "portfolio_log_return": 0.0,
            "portfolio_drawdown": 0.0,
            "portfolio_vol": 0.0,
            "portfolio_turnover": float(np.clip(turnover, 0.0, 3.0)),
            "portfolio_cost": float(np.clip(cost + borrow, 0.0, 1.0)),
            "portfolio_utility": -1.0,
            "future_equity_ratio": 0.0,
        }
    return {
        **outcome,
    }


class V2WorldModelDataset(Dataset):
    def __init__(
        self,
        arrays: MarketArrays,
        split_dates: pd.DatetimeIndex,
        lookback: int,
        horizons: list[int],
        action_config: PortfolioActionConfig,
        seed: int = 42,
    ) -> None:
        self.arrays = arrays
        self.lookback = lookback
        self.horizons = horizons
        self.action_config = action_config
        self.seed = seed
        allowed = set(pd.DatetimeIndex(split_dates))
        self.samples: list[tuple[int, int, int]] = []
        max_h = max(horizons)
        for end_idx, date in enumerate(arrays.dates):
            if date not in allowed:
                continue
            if end_idx < lookback - 1 or end_idx + max_h >= len(arrays.dates):
                continue
            for h in horizons:
                for action_id in range(action_config.n_action_samples):
                    self.samples.append((end_idx, h, action_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end_idx, horizon, action_id = self.samples[idx]
        rng = np.random.default_rng(self.seed + idx * 9973)
        ctx = slice(end_idx - self.lookback + 1, end_idx + 1)
        target_end = end_idx + horizon
        tgt = slice(target_end - self.lookback + 1, target_end + 1)
        valid = self.arrays.tradable[end_idx]
        n_assets = len(self.arrays.tickers)

        x = np.nan_to_num(self.arrays.features[ctx], nan=0.0).transpose(1, 0, 2)
        y = np.nan_to_num(self.arrays.features[tgt], nan=0.0).transpose(1, 0, 2)
        current_weights = _sample_weights(rng, valid, n_assets, self.action_config)
        action_weights = _sample_weights(rng, valid, n_assets, self.action_config)
        future_returns = np.nan_to_num(self.arrays.log_returns[end_idx + 1 : target_end + 1], nan=0.0)
        outcome = _portfolio_outcome(future_returns, current_weights, action_weights, self.action_config)
        portfolio_state = np.concatenate(
            [
                current_weights,
                np.array(
                    [
                        current_weights[-1],
                        np.abs(current_weights[:-1]).sum(),
                        np.maximum(current_weights[:-1], 0.0).sum(),
                        np.abs(np.minimum(current_weights[:-1], 0.0)).sum(),
                        0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        outcome_vec = np.array(
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
        outcome_vec = np.nan_to_num(outcome_vec, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "context": torch.tensor(x, dtype=torch.float32),
            "target": torch.tensor(y, dtype=torch.float32),
            "context_mask": torch.tensor(self.arrays.tradable[ctx].T, dtype=torch.bool),
            "target_mask": torch.tensor(self.arrays.tradable[tgt].T, dtype=torch.bool),
            "tradable_mask": torch.tensor(valid, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "portfolio_state": torch.tensor(portfolio_state, dtype=torch.float32),
            "action": torch.tensor(action_weights, dtype=torch.float32),
            "outcome": torch.tensor(outcome_vec, dtype=torch.float32),
            "utility": torch.tensor(outcome["portfolio_utility"], dtype=torch.float32),
        }


def summarize_v2_batch(batch: dict[str, torch.Tensor]) -> dict[str, float | bool]:
    outcome = batch["outcome"].detach().cpu()
    utility = batch["utility"].detach().cpu()
    return {
        "outcome_finite": bool(torch.isfinite(outcome).all().item()),
        "utility_finite": bool(torch.isfinite(utility).all().item()),
        "outcome_min": float(outcome.min().item()),
        "outcome_max": float(outcome.max().item()),
        "utility_min": float(utility.min().item()),
        "utility_max": float(utility.max().item()),
    }

