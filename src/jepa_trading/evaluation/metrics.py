from __future__ import annotations

import numpy as np
import pandas as pd


def validate_backtest_history(history: pd.DataFrame, name: str = "strategy") -> dict[str, object]:
    required = ["equity"]
    missing = [col for col in required if col not in history.columns]
    if history.empty:
        return {"model": name, "valid_backtest": False, "invalid_reason": "empty history"}
    if missing:
        return {"model": name, "valid_backtest": False, "invalid_reason": f"missing columns: {missing}"}
    check_cols = [c for c in ["equity", "reward", "log_return", "turnover", "cost"] if c in history.columns]
    values = history[check_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return {"model": name, "valid_backtest": False, "invalid_reason": "non-finite values in backtest"}
    if (history["equity"].astype(float) <= 0).any():
        return {"model": name, "valid_backtest": False, "invalid_reason": "non-positive equity"}
    return {"model": name, "valid_backtest": True, "invalid_reason": ""}


def validate_backtest_histories(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([validate_backtest_history(history, name) for name, history in histories.items()])


def equity_metrics(history: pd.DataFrame, periods_per_year: int = 252) -> dict[str, float]:
    if history.empty:
        return {"valid_backtest": False, "invalid_reason": "empty history"}
    validity = validate_backtest_history(history)
    if not validity["valid_backtest"]:
        return {
            "valid_backtest": False,
            "invalid_reason": validity["invalid_reason"],
            "total_return": np.nan,
            "cagr": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "avg_turnover": np.nan,
            "total_cost": np.nan,
            "avg_cash_weight": np.nan,
        }
    equity = history["equity"].astype(float)
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = max(len(equity) / periods_per_year, 1e-8)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = returns.mean() / (returns.std() + 1e-12) * np.sqrt(periods_per_year)
    downside = returns[returns < 0].std()
    sortino = returns.mean() / (downside + 1e-12) * np.sqrt(periods_per_year)
    drawdown = equity / equity.cummax() - 1.0
    return {
        "valid_backtest": True,
        "invalid_reason": "",
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "avg_turnover": float(history.get("turnover", pd.Series(0.0, index=history.index)).mean()),
        "total_cost": float(history.get("cost", pd.Series(0.0, index=history.index)).sum()),
        "avg_cash_weight": float(history.get("cash_weight", pd.Series(np.nan, index=history.index)).mean()),
    }


def metrics_table(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, history in histories.items():
        row = {"model": name}
        row.update(equity_metrics(history))
        rows.append(row)
    table = pd.DataFrame(rows)
    if "valid_backtest" in table.columns and (~table["valid_backtest"].astype(bool)).any():
        invalid = table.loc[~table["valid_backtest"].astype(bool), ["model", "invalid_reason"]]
        print("INVALID BACKTEST DETECTED. Do not interpret performance positively.")
        print(invalid.to_string(index=False))
    return table.sort_values("sharpe", ascending=False, na_position="last")
