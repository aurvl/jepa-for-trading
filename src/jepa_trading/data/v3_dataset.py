from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from jepa_trading.data.dataset import MarketArrays
from jepa_trading.data.v2_dataset import (
    PortfolioActionConfig,
    _portfolio_outcome,
    _sample_weights,
    cash_action,
    derisk_action,
    hold_action,
    v3_candidate_actions,
)


class V3WorldModelDataset(Dataset):
    def __init__(
        self,
        arrays: MarketArrays,
        split_dates: pd.DatetimeIndex,
        lookback: int,
        horizons: list[int],
        action_config: PortfolioActionConfig,
        seed: int = 42,
        n_sampled_actions: int = 4,
    ) -> None:
        self.arrays = arrays
        self.lookback = lookback
        self.horizons = horizons
        self.action_config = action_config
        self.seed = seed
        self.n_sampled_actions = n_sampled_actions
        allowed = set(pd.DatetimeIndex(split_dates))
        self.samples: list[tuple[int, int, int]] = []
        max_h = max(horizons)
        # One sample per date/horizon; action candidates are evaluated inside
        # each item so the ranking loss sees hold/cash/de-risk alternatives.
        for end_idx, date in enumerate(arrays.dates):
            if date not in allowed:
                continue
            if end_idx < lookback - 1 or end_idx + max_h >= len(arrays.dates):
                continue
            for h in horizons:
                self.samples.append((end_idx, h, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end_idx, horizon, _ = self.samples[idx]
        rng = np.random.default_rng(self.seed + idx * 9973)
        ctx = slice(end_idx - self.lookback + 1, end_idx + 1)
        target_end = end_idx + horizon
        tgt = slice(target_end - self.lookback + 1, target_end + 1)
        valid = self.arrays.tradable[end_idx]
        n_assets = len(self.arrays.tickers)

        x = np.nan_to_num(self.arrays.features[ctx], nan=0.0).transpose(1, 0, 2)
        y = np.nan_to_num(self.arrays.features[tgt], nan=0.0).transpose(1, 0, 2)
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
        actions = np.stack([a for _, a in candidates]).astype(np.float32)
        names = [name for name, _ in candidates]
        outcomes = []
        utilities = []
        for _, action in candidates:
            out = _portfolio_outcome(future_returns, current_weights, action, self.action_config)
            outcomes.append(
                [
                    out["portfolio_log_return"],
                    out["portfolio_drawdown"],
                    out["portfolio_vol"],
                    out["portfolio_turnover"],
                    out["portfolio_cost"],
                    out["future_equity_ratio"],
                ]
            )
            utilities.append(out["portfolio_utility"])
        outcomes_arr = np.nan_to_num(np.asarray(outcomes, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        utilities_arr = np.nan_to_num(np.asarray(utilities, dtype=np.float32), nan=-1.0, posinf=3.0, neginf=-3.0)
        best_idx = int(np.argmax(utilities_arr))
        hold_idx = names.index("hold")
        cash_idx = names.index("cash")
        derisk_idx = names.index("derisk")
        advantage_vs_hold = float(utilities_arr[best_idx] - utilities_arr[hold_idx])
        advantage_vs_cash = float(utilities_arr[best_idx] - utilities_arr[cash_idx])

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
            "context": torch.tensor(x, dtype=torch.float32),
            "target": torch.tensor(y, dtype=torch.float32),
            "context_mask": torch.tensor(self.arrays.tradable[ctx].T, dtype=torch.bool),
            "target_mask": torch.tensor(self.arrays.tradable[tgt].T, dtype=torch.bool),
            "tradable_mask": torch.tensor(valid, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "portfolio_state": torch.tensor(portfolio_state, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "outcomes": torch.tensor(outcomes_arr, dtype=torch.float32),
            "utilities": torch.tensor(utilities_arr, dtype=torch.float32),
            "best_action_index": torch.tensor(best_idx, dtype=torch.long),
            "hold_action_index": torch.tensor(hold_idx, dtype=torch.long),
            "cash_action_index": torch.tensor(cash_idx, dtype=torch.long),
            "derisk_action_index": torch.tensor(derisk_idx, dtype=torch.long),
            "best_action": torch.tensor(actions[best_idx], dtype=torch.float32),
            "best_outcome": torch.tensor(outcomes_arr[best_idx], dtype=torch.float32),
            "best_utility": torch.tensor(utilities_arr[best_idx], dtype=torch.float32),
            "advantage_vs_hold": torch.tensor(advantage_vs_hold, dtype=torch.float32),
            "advantage_vs_cash": torch.tensor(advantage_vs_cash, dtype=torch.float32),
        }


def summarize_v3_batch(batch: dict[str, torch.Tensor]) -> dict[str, float | bool]:
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
        "n_candidates": int(batch["actions"].shape[1]),
    }

