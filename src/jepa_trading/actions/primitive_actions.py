from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrimitiveAction:
    asset_scores: np.ndarray
    gross_exposure_delta: float
    net_exposure_target: float
    cash_target_delta: float
    risk_budget: float
    rebalance_intensity: float
    horizon: int
    long_short_bias: float = 1.0

    def as_vector(self) -> np.ndarray:
        meta = np.array(
            [
                self.gross_exposure_delta,
                self.net_exposure_target,
                self.cash_target_delta,
                self.risk_budget,
                self.rebalance_intensity,
                float(self.horizon),
                self.long_short_bias,
            ],
            dtype=np.float32,
        )
        return np.concatenate([np.asarray(self.asset_scores, dtype=np.float32), meta])


@dataclass
class PrimitiveActionBatch:
    vectors: np.ndarray
    horizons: np.ndarray
    names: list[str]


def primitive_action_dim(n_assets: int) -> int:
    return n_assets + 7


def primitive_from_vector(vector: np.ndarray, n_assets: int) -> PrimitiveAction:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape[-1] != primitive_action_dim(n_assets):
        raise ValueError(f"Expected primitive action dim {primitive_action_dim(n_assets)}, got {vector.shape[-1]}")
    meta = vector[n_assets:]
    return PrimitiveAction(
        asset_scores=vector[:n_assets],
        gross_exposure_delta=float(meta[0]),
        net_exposure_target=float(meta[1]),
        cash_target_delta=float(meta[2]),
        risk_budget=float(meta[3]),
        rebalance_intensity=float(meta[4]),
        horizon=int(round(float(meta[5]))),
        long_short_bias=float(meta[6]),
    )
