from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_macro(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Macro file not found at {path}. On Kaggle, upload macro_data.parquet "
            "and set config['data']['macro_path'] to that file."
        )
    if path.suffix.lower() == ".parquet":
        macro = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        macro = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported macro file extension: {path.suffix}. Use .parquet or .csv.")
    if "date" not in macro.columns and "Date" in macro.columns:
        macro = macro.rename(columns={"Date": "date"})
    if "date" not in macro.columns:
        raise ValueError("Macro file must contain a 'date' or 'Date' column.")
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
    # If the user supplies estimated_volatility_with_macro.csv, keep only macro
    # factors here. Per-asset sigma is recomputed from each asset's own history.
    drop_cols = [
        c
        for c in ["sample_split", "is_development_sample", "price", "sigma"]
        if c in macro.columns
    ]
    macro = macro.drop(columns=drop_cols)
    numeric_cols = [c for c in macro.columns if c != "date"]
    macro[numeric_cols] = macro[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return macro.sort_values("date").drop_duplicates("date")


def merge_macro(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    out = prices.merge(macro, on="date", how="left")
    macro_cols = [c for c in macro.columns if c != "date"]
    out[macro_cols] = out.groupby("ticker", group_keys=False)[macro_cols].ffill()
    return out
