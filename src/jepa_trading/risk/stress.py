from __future__ import annotations

import numpy as np


def stress_scenarios(returns_window: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    returns = np.nan_to_num(np.asarray(returns_window, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.asarray(weights, dtype=np.float32)
    asset_weights = weights[:-1] if weights.shape[0] == returns.shape[1] + 1 else weights
    if returns.ndim != 2 or returns.shape[0] == 0:
        return {"one_day_shock": 0.0, "vol_spike": 0.0, "correlation_shock": 0.0}
    portfolio_returns = returns @ asset_weights
    one_day_shock = float(np.quantile(portfolio_returns, 0.01))
    vol_spike = float(portfolio_returns.mean() - 3.0 * portfolio_returns.std())
    market_shock = np.quantile(returns, 0.05, axis=0)
    correlation_shock = float(market_shock @ asset_weights)
    return {
        "one_day_shock": one_day_shock,
        "vol_spike": vol_spike,
        "correlation_shock": correlation_shock,
    }
