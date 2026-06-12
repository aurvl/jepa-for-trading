from __future__ import annotations

import numpy as np

from jepa_trading.actions.primitive_actions import PrimitiveAction, PrimitiveActionBatch


def sample_primitive_actions(
    rng: np.random.Generator,
    n_assets: int,
    horizons: list[int],
    n_actions: int,
    score_mean: np.ndarray | None = None,
    score_scale: float = 1.0,
    gross_delta_mean: float = 0.0,
    gross_delta_scale: float = 0.25,
    risk_budget_mean: float = 0.65,
    risk_budget_scale: float = 0.25,
) -> PrimitiveActionBatch:
    vectors = []
    names = []
    horizon_values = []
    score_mean = np.zeros(n_assets, dtype=np.float32) if score_mean is None else np.asarray(score_mean, dtype=np.float32)
    for i in range(n_actions):
        horizon = int(rng.choice(horizons))
        action = PrimitiveAction(
            asset_scores=score_mean + rng.normal(0.0, score_scale, size=n_assets).astype(np.float32),
            gross_exposure_delta=float(np.clip(rng.normal(gross_delta_mean, gross_delta_scale), -1.0, 1.0)),
            net_exposure_target=float(np.clip(rng.normal(0.6, 0.35), -1.0, 1.0)),
            cash_target_delta=float(np.clip(rng.normal(0.0, 0.25), -1.0, 1.0)),
            risk_budget=float(np.clip(rng.normal(risk_budget_mean, risk_budget_scale), 0.0, 1.0)),
            rebalance_intensity=float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0)),
            horizon=horizon,
            long_short_bias=float(np.clip(rng.normal(1.0, 0.25), -1.0, 1.0)),
        )
        vectors.append(action.as_vector())
        names.append(f"primitive_{i}")
        horizon_values.append(horizon)
    return PrimitiveActionBatch(
        vectors=np.asarray(vectors, dtype=np.float32),
        horizons=np.asarray(horizon_values, dtype=np.int64),
        names=names,
    )
