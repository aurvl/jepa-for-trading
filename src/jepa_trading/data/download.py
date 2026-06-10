from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


PRICE_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_path: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists() and not force:
        return pd.read_parquet(cache)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("No price data downloaded. Check tickers/date range/network.")

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            part = raw[ticker].rename(columns=PRICE_COLUMNS)
            part = part[[c for c in PRICE_COLUMNS.values() if c in part.columns]]
            part["ticker"] = ticker
            frames.append(part.reset_index().rename(columns={"Date": "date"}))
    else:
        part = raw.rename(columns=PRICE_COLUMNS)
        part = part[[c for c in PRICE_COLUMNS.values() if c in part.columns]]
        part["ticker"] = tickers[0]
        frames.append(part.reset_index().rename(columns={"Date": "date"}))

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices = prices.dropna(subset=["date", "ticker", "close"]).sort_values(["date", "ticker"])

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache, index=False)
    return prices

