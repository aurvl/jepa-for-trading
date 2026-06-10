from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    signal = (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()
    return (ema_fast - ema_slow) - signal


def add_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    parts: list[pd.DataFrame] = []
    for ticker, g in df.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        g = g.copy()
        price = g["adj_close"].where(g["adj_close"].notna(), g["close"])
        g["log_price"] = np.log(price.replace(0, np.nan))
        g["log_return"] = g["log_price"].diff()
        g["overnight_gap"] = np.log(g["open"] / g["close"].shift(1))
        g["intraday_range"] = np.log(g["high"] / g["low"])
        g["volume_log"] = np.log1p(g["volume"])
        vol_mean = g["volume_log"].rolling(63, min_periods=20).mean()
        vol_std = g["volume_log"].rolling(63, min_periods=20).std()
        g["volume_z"] = (g["volume_log"] - vol_mean) / vol_std.replace(0, np.nan)
        g["rsi_scaled"] = (_rsi(price) - 50.0) / 50.0
        g["macd"] = _macd(price)
        g["rolling_vol_21"] = g["log_return"].rolling(21, min_periods=10).std() * np.sqrt(252)
        g["rolling_vol_63"] = g["log_return"].rolling(63, min_periods=20).std() * np.sqrt(252)
        # Per-asset sigma. This is intentionally not the old global sigma CSV.
        g["sigma"] = g["rolling_vol_21"]
        g["tradable"] = price.notna() & g["log_return"].notna()
        parts.append(g)

    out = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"])
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "log_return",
        "overnight_gap",
        "intraday_range",
        "volume_z",
        "rsi_scaled",
        "macd",
        "rolling_vol_21",
        "rolling_vol_63",
        "sigma_log",
    ]
    forbidden = {"ticker", "tradable", "date", "split"}
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    macro = [
        c
        for c in df.columns
        if c not in forbidden
        and c in numeric_cols
        and (
            c.startswith("t")
            or c.startswith("theme_")
            or c.endswith("__score")
            or c.endswith("__raw_factor")
        )
    ]
    return [c for c in base + macro if c in df.columns]
