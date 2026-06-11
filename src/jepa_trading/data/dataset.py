from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class MarketArrays:
    dates: pd.DatetimeIndex
    tickers: list[str]
    features: np.ndarray
    log_returns: np.ndarray
    sigma: np.ndarray
    close: np.ndarray
    tradable: np.ndarray
    feature_columns: list[str]


def build_market_arrays(df: pd.DataFrame, feature_columns: list[str]) -> MarketArrays:
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"]).drop_duplicates().sort_values())
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    date_index = {d: i for i, d in enumerate(dates)}
    ticker_index = {t: i for i, t in enumerate(tickers)}

    shape = (len(dates), len(tickers))
    features = np.full((len(dates), len(tickers), len(feature_columns)), np.nan, dtype=np.float32)
    log_returns = np.full(shape, np.nan, dtype=np.float32)
    sigma = np.full(shape, np.nan, dtype=np.float32)
    close = np.full(shape, np.nan, dtype=np.float32)
    tradable = np.zeros(shape, dtype=bool)

    for row in df.itertuples(index=False):
        i = date_index[getattr(row, "date")]
        j = ticker_index[getattr(row, "ticker")]
        for k, col in enumerate(feature_columns):
            features[i, j, k] = getattr(row, col, np.nan)
        log_returns[i, j] = getattr(row, "log_return_raw", getattr(row, "log_return", np.nan))
        sigma[i, j] = getattr(row, "sigma_raw", getattr(row, "sigma", np.nan))
        close[i, j] = getattr(row, "close_raw", getattr(row, "close", np.nan))
        tradable[i, j] = bool(getattr(row, "tradable", False))

    return MarketArrays(dates, tickers, features, log_returns, sigma, close, tradable, feature_columns)


class MultiAssetJEPADataset(Dataset):
    def __init__(
        self,
        arrays: MarketArrays,
        split_dates: pd.DatetimeIndex,
        lookback: int,
        horizons: list[int],
    ) -> None:
        self.arrays = arrays
        self.lookback = lookback
        self.horizons = horizons
        allowed = set(pd.DatetimeIndex(split_dates))
        self.samples: list[tuple[int, int]] = []
        max_h = max(horizons)
        for end_idx, date in enumerate(arrays.dates):
            if date not in allowed:
                continue
            if end_idx < lookback - 1 or end_idx + max_h >= len(arrays.dates):
                continue
            for h in horizons:
                self.samples.append((end_idx, h))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end_idx, horizon = self.samples[idx]
        ctx = slice(end_idx - self.lookback + 1, end_idx + 1)
        target_end = end_idx + horizon
        tgt = slice(target_end - self.lookback + 1, target_end + 1)

        x = self.arrays.features[ctx]
        y = self.arrays.features[tgt]
        valid_x = self.arrays.tradable[ctx]
        valid_y = self.arrays.tradable[tgt]
        tradable_now = self.arrays.tradable[end_idx]

        # Torch convention: assets x time x features.
        x = np.nan_to_num(x, nan=0.0).transpose(1, 0, 2)
        y = np.nan_to_num(y, nan=0.0).transpose(1, 0, 2)
        future_returns = self.arrays.log_returns[end_idx + 1 : target_end + 1]
        future_returns = np.nan_to_num(future_returns, nan=0.0)
        cum_return = future_returns.sum(axis=0)
        realized_vol = np.nan_to_num(self.arrays.sigma[target_end], nan=0.0)
        future_path = np.cumsum(future_returns, axis=0)
        max_drawdown = future_path - np.maximum.accumulate(future_path, axis=0)

        return {
            "context": torch.tensor(x, dtype=torch.float32),
            "target": torch.tensor(y, dtype=torch.float32),
            "context_mask": torch.tensor(valid_x.T, dtype=torch.bool),
            "target_mask": torch.tensor(valid_y.T, dtype=torch.bool),
            "tradable_mask": torch.tensor(tradable_now, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "future_return": torch.tensor(cum_return, dtype=torch.float32),
            "future_sigma": torch.tensor(realized_vol, dtype=torch.float32),
            "future_drawdown": torch.tensor(max_drawdown.min(axis=0), dtype=torch.float32),
        }
