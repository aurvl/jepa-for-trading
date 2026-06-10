from __future__ import annotations

import pandas as pd


def add_time_split(df: pd.DataFrame, train_end: str, val_end: str) -> pd.DataFrame:
    out = df.copy()
    date = pd.to_datetime(out["date"])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    out["split"] = "test"
    out.loc[date <= train_end_ts, "split"] = "train"
    out.loc[(date > train_end_ts) & (date <= val_end_ts), "split"] = "val"
    return out


def split_dates(df: pd.DataFrame, split: str) -> pd.DatetimeIndex:
    dates = pd.to_datetime(df.loc[df["split"] == split, "date"]).drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)

