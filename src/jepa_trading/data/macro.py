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
    macro = pd.read_parquet(path)
    if "date" not in macro.columns:
        raise ValueError("macro_data.parquet must contain a 'date' column.")
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
    drop_cols = [c for c in ["sample_split", "is_development_sample"] if c in macro.columns]
    macro = macro.drop(columns=drop_cols)
    numeric_cols = [c for c in macro.columns if c != "date"]
    macro[numeric_cols] = macro[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return macro.sort_values("date").drop_duplicates("date")


def merge_macro(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    out = prices.merge(macro, on="date", how="left")
    macro_cols = [c for c in macro.columns if c != "date"]
    out[macro_cols] = out.groupby("ticker", group_keys=False)[macro_cols].ffill()
    return out

