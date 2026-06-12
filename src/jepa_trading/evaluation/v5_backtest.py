from __future__ import annotations

import numpy as np
import pandas as pd

from jepa_trading.data.dataset import MarketArrays
from jepa_trading.planning.v5_planner import V5ActionWorldModelPlanner, portfolio_state_from_weights
from jepa_trading.rl.env import TradingEnv
from jepa_trading.rl.observer import RawMarketObserver


def _context_from_arrays(arrays: MarketArrays, end_idx: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    ctx = slice(end_idx - lookback + 1, end_idx + 1)
    x = np.nan_to_num(arrays.features[ctx], nan=0.0).transpose(1, 0, 2)
    mask = arrays.tradable[ctx].T
    return x.astype(np.float32), mask.astype(bool)


def run_v5_planner_backtest(
    arrays: MarketArrays,
    planner: V5ActionWorldModelPlanner,
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
        portfolio_state = portfolio_state_from_weights(
            env.state.weights.astype(np.float32),
            equity=env.state.equity / max(cash_initial, 1e-8),
            peak_equity=env.state.peak_equity / max(cash_initial, 1e-8),
            last_turnover=env.state.last_turnover,
        )
        sigma_now = arrays.sigma[env.idx]
        recent_returns = np.nan_to_num(arrays.log_returns[max(0, env.idx - 20) : env.idx + 1], nan=0.0).sum(axis=0)
        result = planner.choose_action(
            context,
            context_mask,
            portfolio_state,
            mask,
            sigma_now,
            recent_returns,
        )
        _, _, done, info, mask = env.step(result.action)
        weights = env.state.weights.astype(float)
        info["selected_action_name"] = result.selected_name
        info["planned_horizon"] = result.horizon
        info["predicted_goal_cost"] = result.predicted_cost
        info["predicted_log_return"] = float(result.predicted_outcome[0])
        info["predicted_drawdown"] = float(result.predicted_outcome[1])
        info["predicted_vol"] = float(result.predicted_outcome[2])
        info["gross_exposure"] = float(np.abs(weights[:-1]).sum())
        info["n_candidate_actions"] = float(len(result.candidate_names))
        for ticker, weight in zip(arrays.tickers, weights[:-1]):
            info[f"weight_{ticker}"] = float(weight)
    return pd.DataFrame(env.history)
