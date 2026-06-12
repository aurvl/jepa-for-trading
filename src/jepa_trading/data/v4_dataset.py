from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from jepa_trading.data.dataset import MarketArrays
from jepa_trading.data.v2_dataset import (
    PortfolioActionConfig,
    _portfolio_outcome,
    _sample_weights,
    v3_candidate_actions,
)


@dataclass
class V4RiskUtilityConfig:
    return_weight: float = 1.0
    drawdown_penalty: float = 2.5
    volatility_penalty: float = 0.15
    turnover_penalty: float = 0.04
    cost_penalty: float = 2.0
    concentration_penalty: float = 0.04
    risk_off_drawdown: float = -0.06
    risk_off_return: float = -0.03
    defensive_bonus: float = 0.20
    hold_risk_off_penalty: float = 0.25
    utility_clip: float = 3.0


def v4_realized_utilities(
    outcomes: np.ndarray,
    actions: np.ndarray,
    names: list[str],
    utility_config: V4RiskUtilityConfig,
) -> tuple[np.ndarray, bool]:
    log_return = outcomes[:, 0]
    drawdown = outcomes[:, 1]
    volatility = outcomes[:, 2]
    turnover = outcomes[:, 3]
    cost = outcomes[:, 4]
    concentration = np.sum(actions[:, :-1] ** 2, axis=1)

    utilities = (
        utility_config.return_weight * log_return
        - utility_config.drawdown_penalty * np.abs(np.minimum(drawdown, 0.0))
        - utility_config.volatility_penalty * volatility
        - utility_config.turnover_penalty * turnover
        - utility_config.cost_penalty * cost
        - utility_config.concentration_penalty * concentration
    )

    hold_idx = names.index("hold")
    cash_idx = names.index("cash")
    derisk_idx = names.index("derisk")
    hold_is_bad = bool(
        outcomes[hold_idx, 1] <= utility_config.risk_off_drawdown
        or outcomes[hold_idx, 0] <= utility_config.risk_off_return
    )
    if hold_is_bad:
        utilities[hold_idx] -= utility_config.hold_risk_off_penalty
        utilities[cash_idx] += utility_config.defensive_bonus
        utilities[derisk_idx] += 0.75 * utility_config.defensive_bonus
        gross_exposure = np.abs(actions[:, :-1]).sum(axis=1)
        defensive_exposure = actions[:, -1] + np.maximum(0.0, 1.0 - gross_exposure)
        utilities += 0.25 * utility_config.defensive_bonus * defensive_exposure

    utilities = np.nan_to_num(utilities, nan=-1.0, posinf=utility_config.utility_clip, neginf=-utility_config.utility_clip)
    utilities = np.clip(utilities, -utility_config.utility_clip, utility_config.utility_clip)
    return utilities.astype(np.float32), hold_is_bad


class V4WorldModelDataset(Dataset):
    def __init__(
        self,
        arrays: MarketArrays,
        split_dates: pd.DatetimeIndex,
        lookback: int,
        horizons: list[int],
        action_config: PortfolioActionConfig,
        utility_config: V4RiskUtilityConfig,
        seed: int = 42,
        n_sampled_actions: int = 4,
    ) -> None:
        self.arrays = arrays
        self.lookback = lookback
        self.horizons = horizons
        self.action_config = action_config
        self.utility_config = utility_config
        self.seed = seed
        self.n_sampled_actions = n_sampled_actions
        allowed = set(pd.DatetimeIndex(split_dates))
        self.samples: list[tuple[int, int]] = []
        max_h = max(horizons)
        for end_idx, date in enumerate(arrays.dates):
            if date not in allowed:
                continue
            if end_idx < lookback - 1 or end_idx + max_h >= len(arrays.dates):
                continue
            for horizon in horizons:
                self.samples.append((end_idx, horizon))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end_idx, horizon = self.samples[idx]
        rng = np.random.default_rng(self.seed + idx * 9973)
        ctx = slice(end_idx - self.lookback + 1, end_idx + 1)
        target_end = end_idx + horizon
        tgt = slice(target_end - self.lookback + 1, target_end + 1)
        valid = self.arrays.tradable[end_idx]
        n_assets = len(self.arrays.tickers)

        context = np.nan_to_num(self.arrays.features[ctx], nan=0.0).transpose(1, 0, 2)
        target = np.nan_to_num(self.arrays.features[tgt], nan=0.0).transpose(1, 0, 2)
        current_weights = _sample_weights(rng, valid, n_assets, self.action_config)
        sigma_now = self.arrays.sigma[end_idx]
        future_returns = np.nan_to_num(self.arrays.log_returns[end_idx + 1 : target_end + 1], nan=0.0)

        candidates = v3_candidate_actions(
            rng=rng,
            valid=valid,
            sigma=sigma_now,
            current_weights=current_weights,
            n_assets=n_assets,
            config=self.action_config,
            n_sampled_actions=self.n_sampled_actions,
        )
        names = [name for name, _ in candidates]
        actions = np.stack([action for _, action in candidates]).astype(np.float32)
        outcome_rows = []
        for _, action in candidates:
            out = _portfolio_outcome(future_returns, current_weights, action, self.action_config)
            outcome_rows.append(
                [
                    out["portfolio_log_return"],
                    out["portfolio_drawdown"],
                    out["portfolio_vol"],
                    out["portfolio_turnover"],
                    out["portfolio_cost"],
                    out["future_equity_ratio"],
                ]
            )
        outcomes = np.nan_to_num(np.asarray(outcome_rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        utilities, risk_off_label = v4_realized_utilities(outcomes, actions, names, self.utility_config)

        best_idx = int(np.argmax(utilities))
        hold_idx = names.index("hold")
        cash_idx = names.index("cash")
        derisk_idx = names.index("derisk")
        advantage_vs_hold = float(utilities[best_idx] - utilities[hold_idx])
        advantage_vs_cash = float(utilities[best_idx] - utilities[cash_idx])

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

        return {
            "context": torch.tensor(context, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "context_mask": torch.tensor(self.arrays.tradable[ctx].T, dtype=torch.bool),
            "target_mask": torch.tensor(self.arrays.tradable[tgt].T, dtype=torch.bool),
            "tradable_mask": torch.tensor(valid, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "portfolio_state": torch.tensor(portfolio_state, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "outcomes": torch.tensor(outcomes, dtype=torch.float32),
            "utilities": torch.tensor(utilities, dtype=torch.float32),
            "best_action_index": torch.tensor(best_idx, dtype=torch.long),
            "hold_action_index": torch.tensor(hold_idx, dtype=torch.long),
            "cash_action_index": torch.tensor(cash_idx, dtype=torch.long),
            "derisk_action_index": torch.tensor(derisk_idx, dtype=torch.long),
            "best_action": torch.tensor(actions[best_idx], dtype=torch.float32),
            "best_outcome": torch.tensor(outcomes[best_idx], dtype=torch.float32),
            "best_utility": torch.tensor(utilities[best_idx], dtype=torch.float32),
            "advantage_vs_hold": torch.tensor(advantage_vs_hold, dtype=torch.float32),
            "advantage_vs_cash": torch.tensor(advantage_vs_cash, dtype=torch.float32),
            "risk_off_label": torch.tensor(risk_off_label, dtype=torch.bool),
        }


def summarize_v4_batch(batch: dict[str, torch.Tensor]) -> dict[str, float | bool]:
    outcomes = batch["outcomes"].detach().cpu()
    utilities = batch["utilities"].detach().cpu()
    return {
        "outcomes_finite": bool(torch.isfinite(outcomes).all().item()),
        "utilities_finite": bool(torch.isfinite(utilities).all().item()),
        "outcome_min": float(outcomes.min().item()),
        "outcome_max": float(outcomes.max().item()),
        "utility_min": float(utilities.min().item()),
        "utility_max": float(utilities.max().item()),
        "mean_advantage_vs_hold": float(batch["advantage_vs_hold"].float().mean().item()),
        "mean_advantage_vs_cash": float(batch["advantage_vs_cash"].float().mean().item()),
        "risk_off_label_rate": float(batch["risk_off_label"].float().mean().item()),
        "n_candidates": int(batch["actions"].shape[1]),
    }
