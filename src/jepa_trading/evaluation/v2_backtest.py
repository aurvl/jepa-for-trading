from __future__ import annotations

import numpy as np
import pandas as pd

from jepa_trading.data.dataset import MarketArrays
from jepa_trading.planning.v2_planner import V2ImaginationPlanner
from jepa_trading.rl.env import TradingEnv
from jepa_trading.rl.observer import RawMarketObserver


def _portfolio_state_from_env(env: TradingEnv) -> np.ndarray:
    weights = env.state.weights.astype(np.float32)
    return np.concatenate(
        [
            weights,
            np.array(
                [
                    weights[-1],
                    np.abs(weights[:-1]).sum(),
                    np.maximum(weights[:-1], 0.0).sum(),
                    np.abs(np.minimum(weights[:-1], 0.0)).sum(),
                    env.state.last_turnover,
                ],
                dtype=np.float32,
            ),
        ]
    )


def _context_from_arrays(arrays: MarketArrays, end_idx: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    ctx = slice(end_idx - lookback + 1, end_idx + 1)
    x = np.nan_to_num(arrays.features[ctx], nan=0.0).transpose(1, 0, 2)
    mask = arrays.tradable[ctx].T
    return x.astype(np.float32), mask.astype(bool)


def run_v2_planner_backtest(
    arrays: MarketArrays,
    planner: V2ImaginationPlanner,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lookback: int,
    cash_initial: float,
    transaction_cost_bps: float,
    max_weight_per_asset: float,
    max_turnover: float,
    mode: str = "long_only",
    max_long_weight: float | None = None,
    max_short_weight: float = 0.05,
    max_gross_exposure: float = 1.0,
    max_net_exposure: float = 1.0,
    borrow_cost_bps: float = 2.0,
) -> pd.DataFrame:
    env = TradingEnv(
        arrays=arrays,
        observer=RawMarketObserver(arrays, lookback),
        start_date=start_date,
        end_date=end_date,
        lookback=lookback,
        cash_initial=cash_initial,
        transaction_cost_bps=transaction_cost_bps,
        max_weight_per_asset=max_weight_per_asset,
        max_turnover=max_turnover,
        mode=mode,
        max_long_weight=max_long_weight,
        max_short_weight=max_short_weight,
        max_gross_exposure=max_gross_exposure,
        max_net_exposure=max_net_exposure,
        borrow_cost_bps=borrow_cost_bps,
    )
    _, mask = env.reset()
    done = False
    while not done:
        context, context_mask = _context_from_arrays(arrays, env.idx, lookback)
        portfolio_state = _portfolio_state_from_env(env)
        result = planner.choose_action(context, context_mask, portfolio_state, mask)
        _, _, done, info, mask = env.step(result.action)
        info["planned_horizon"] = result.horizon
        info["planner_score"] = result.score
        info["predicted_log_return"] = float(result.predicted_outcome[0])
        info["predicted_drawdown"] = float(result.predicted_outcome[1])
        info["predicted_vol"] = float(result.predicted_outcome[2])
    return pd.DataFrame(env.history)
