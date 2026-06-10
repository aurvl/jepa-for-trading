from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class MarketScaler:
    robust_columns: list[str]
    sigma_column: str = "sigma_log"

    def __post_init__(self) -> None:
        self.robust = RobustScaler()
        self.sigma_scaler = StandardScaler()
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "MarketScaler":
        cols = [c for c in self.robust_columns if c in df.columns and c != self.sigma_column]
        self.robust_columns = cols
        self.robust.fit(df[cols].replace([np.inf, -np.inf], np.nan).dropna()) if cols else None
        if self.sigma_column in df.columns:
            sigma_values = df[[self.sigma_column]].replace([np.inf, -np.inf], np.nan).dropna()
            self.sigma_scaler.fit(sigma_values)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("MarketScaler must be fitted before transform.")
        out = df.copy()
        if self.robust_columns:
            values = out[self.robust_columns].replace([np.inf, -np.inf], np.nan)
            mask = values.notna().all(axis=1)
            out.loc[mask, self.robust_columns] = self.robust.transform(values.loc[mask])
        if self.sigma_column in out.columns:
            values = out[[self.sigma_column]].replace([np.inf, -np.inf], np.nan)
            mask = values[self.sigma_column].notna()
            out.loc[mask, self.sigma_column] = self.sigma_scaler.transform(values.loc[mask])
        return out


def prepare_scaling_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sigma_log"] = np.log(out["sigma"].clip(lower=1e-8))
    return out


def fit_transform_train_only(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_split: str = "train",
) -> tuple[pd.DataFrame, MarketScaler]:
    prepared = prepare_scaling_columns(df)
    train = prepared[prepared["split"] == train_split]
    scaler = MarketScaler(robust_columns=feature_columns).fit(train)
    return scaler.transform(prepared), scaler

