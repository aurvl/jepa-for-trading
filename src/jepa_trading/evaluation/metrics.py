from __future__ import annotations

import numpy as np
import pandas as pd


def equity_metrics(history: pd.DataFrame, periods_per_year: int = 252) -> dict[str, float]:
    if history.empty:
        return {}
    equity = history["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = max(len(equity) / periods_per_year, 1e-8)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = returns.mean() / (returns.std() + 1e-12) * np.sqrt(periods_per_year)
    downside = returns[returns < 0].std()
    sortino = returns.mean() / (downside + 1e-12) * np.sqrt(periods_per_year)
    drawdown = equity / equity.cummax() - 1.0
    return {
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
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)

