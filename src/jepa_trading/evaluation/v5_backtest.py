from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from jepa_trading.data.dataset import MarketArrays
from jepa_trading.data.v5_dataset import V5_OUTCOME_KEYS, realized_goal_cost, realized_outcome_from_action
from jepa_trading.memory import ExperienceBuffer, ExperienceRecord
from jepa_trading.planning.v5_planner import V5ActionWorldModelPlanner, portfolio_state_from_weights
from jepa_trading.rl.env import TradingEnv
from jepa_trading.rl.observer import RawMarketObserver


def _context_from_arrays(arrays: MarketArrays, end_idx: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    ctx = slice(end_idx - lookback + 1, end_idx + 1)
    x = np.nan_to_num(arrays.features[ctx], nan=0.0).transpose(1, 0, 2)
    mask = arrays.tradable[ctx].T
    return x.astype(np.float32), mask.astype(bool)


def validate_v5_backtest(history: pd.DataFrame, name: str = "v5_agent") -> pd.DataFrame:
    reasons = []
    valid = True
    if history.empty:
        return pd.DataFrame([{"model": name, "valid_backtest": False, "reason": "empty history"}])
    numeric = history.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        valid = False
        reasons.append("non-finite numeric values")
    if (history["equity"] <= 0).any():
        valid = False
        reasons.append("negative or zero equity")
    if "gross_exposure" in history and history["gross_exposure"].mean() < 0.03:
        valid = False
        reasons.append("near-zero exposure / cash collapse")
    if "turnover" in history and history["turnover"].sum() <= 1e-8:
        valid = False
        reasons.append("zero-trade strategy")
    if "planned_horizon" in history:
        horizon_share = history["planned_horizon"].value_counts(normalize=True).max()
        if horizon_share > 0.95:
            reasons.append("horizon collapse warning")
    if "turnover" in history and history["turnover"].mean() > 0.80:
        reasons.append("turnover saturation warning")
    return pd.DataFrame(
        [{"model": name, "valid_backtest": bool(valid), "reason": "; ".join(reasons) if reasons else "ok"}]
    )


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
    output_dir: str | Path | None = None,
    return_buffer: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, ExperienceBuffer]:
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
    buffer = ExperienceBuffer()
    planner_rows = []
    while not done:
        idx_before = env.idx
        context, context_mask = _context_from_arrays(arrays, env.idx, lookback)
        portfolio_state = portfolio_state_from_weights(
            env.state.weights.astype(np.float32),
            equity=env.state.equity / max(cash_initial, 1e-8),
            peak_equity=env.state.peak_equity / max(cash_initial, 1e-8),
            last_turnover=env.state.last_turnover,
        )
        result = planner.choose_action(
            context,
            context_mask,
            portfolio_state,
            mask,
            arrays.sigma[env.idx],
            np.nan_to_num(arrays.log_returns[max(0, env.idx - 20) : env.idx + 1], nan=0.0).sum(axis=0),
        )
        current_weights = env.state.weights.astype(np.float32).copy()
        _, _, done, info, mask = env.step(result.executable_action)
        weights = env.state.weights.astype(float)

        target_end = min(idx_before + result.horizon, len(arrays.log_returns) - 1)
        future_returns = arrays.log_returns[idx_before + 1 : target_end + 1]
        realized = realized_outcome_from_action(
            future_returns,
            current_weights,
            result.executable_action,
            planner.execution_layer,
            drawdown_breach=planner.cost_config.max_drawdown,
        )
        realized_cost = realized_goal_cost(realized, result.executable_action, planner.cost_config, result.horizon)
        predicted_return = result.predicted_outcome.get("return_q50", np.nan)
        prediction_error = float(predicted_return - realized[V5_OUTCOME_KEYS.index("realized_return")])

        info["selected_action_name"] = "primitive_cem"
        info["planned_horizon"] = result.horizon
        info["predicted_goal_cost"] = result.predicted_cost
        info["realized_goal_cost"] = realized_cost
        info["prediction_error"] = prediction_error
        info["gross_exposure"] = float(np.abs(weights[:-1]).sum())
        info["cash_weight"] = float(weights[-1])
        info.update({f"pred_{k}": v for k, v in result.predicted_outcome.items()})
        info.update({f"diag_{k}": v for k, v in result.diagnostics.items()})
        for i, value in enumerate(result.primitive_action):
            info[f"primitive_{i}"] = float(value)
        for ticker, weight in zip(arrays.tickers, weights[:-1]):
            info[f"weight_{ticker}"] = float(weight)

        planner_rows.append(
            {
                "date": str(arrays.dates[idx_before]),
                "horizon": result.horizon,
                "predicted_cost": result.predicted_cost,
                "cost_mean": result.diagnostics.get("cost_mean", np.nan),
                "cost_std": result.diagnostics.get("cost_std", np.nan),
                "cash_weight": float(weights[-1]),
                "gross_exposure": float(np.abs(weights[:-1]).sum()),
                "realized_cost": realized_cost,
                "prediction_error": prediction_error,
            }
        )
        buffer.append(
            ExperienceRecord(
                date=str(arrays.dates[idx_before]),
                horizon=result.horizon,
                predicted_cost=result.predicted_cost,
                realized_cost=realized_cost,
                prediction_error=prediction_error,
                portfolio_equity=float(env.state.equity),
                primitive_action={f"p{i}": float(v) for i, v in enumerate(result.primitive_action)},
                predicted_outcomes=result.predicted_outcome,
                realized_outcomes={k: float(v) for k, v in zip(V5_OUTCOME_KEYS, realized)},
                portfolio_state_t={"cash": float(current_weights[-1]), "gross": float(np.abs(current_weights[:-1]).sum())},
                portfolio_state_tp1={"cash": float(weights[-1]), "gross": float(np.abs(weights[:-1]).sum())},
            )
        )

    history = pd.DataFrame(env.history)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        history.to_csv(output / "agent_history.csv", index=False)
        pd.DataFrame(planner_rows).to_csv(output / "planner_diagnostics.csv", index=False)
        validate_v5_backtest(history).to_csv(output / "validity.csv", index=False)
        buffer.save_csv(output / "experience_buffer.csv")
        try:
            buffer.save_parquet(output / "experience_buffer.parquet")
        except Exception:
            pass
        weight_cols = [c for c in history.columns if c.startswith("weight_")]
        if weight_cols:
            history[["date", *weight_cols]].to_csv(output / "portfolio_weights.csv", index=False)
        if "planned_horizon" in history:
            history["planned_horizon"].value_counts().rename_axis("horizon").reset_index(name="count").to_csv(
                output / "horizon_distribution.csv", index=False
            )
        primitive_cols = [c for c in history.columns if c.startswith("primitive_")]
        if primitive_cols:
            history[primitive_cols].describe().T.to_csv(output / "primitive_action_stats.csv")
    if return_buffer:
        return history, buffer
    return history
