from __future__ import annotations

import numpy as np
import pandas as pd

from jepa_trading.evaluation.metrics import validate_backtest_history


def _invalid_result(reason: str) -> dict[str, float | str | bool]:
    return {
        "valid_test": False,
        "invalid_reason": reason,
        "agent_total_return": np.nan,
        "random_mean_total_return": np.nan,
        "random_best_total_return": np.nan,
        "p_value_random_beats_agent": np.nan,
    }


def randomization_p_value(agent_history: pd.DataFrame, random_histories: list[pd.DataFrame]) -> dict[str, float]:
    agent_valid = validate_backtest_history(agent_history, "agent")
    if not agent_valid["valid_backtest"]:
        return _invalid_result(str(agent_valid["invalid_reason"]))
    invalid_random = [
        validate_backtest_history(history, f"random_{i}")
        for i, history in enumerate(random_histories)
        if not validate_backtest_history(history, f"random_{i}")["valid_backtest"]
    ]
    if invalid_random:
        return _invalid_result("one or more random histories contain non-finite values")
    agent_total = agent_history["equity"].iloc[-1] / agent_history["equity"].iloc[0] - 1.0
    random_totals = np.array([h["equity"].iloc[-1] / h["equity"].iloc[0] - 1.0 for h in random_histories])
    if not np.isfinite(agent_total) or not np.isfinite(random_totals).all():
        return _invalid_result("non-finite randomization totals")
    return {
        "valid_test": True,
        "invalid_reason": "",
        "agent_total_return": float(agent_total),
        "random_mean_total_return": float(random_totals.mean()),
        "random_best_total_return": float(random_totals.max()),
        "p_value_random_beats_agent": float((random_totals >= agent_total).mean()),
    }


def bootstrap_mean_return_p_value(
    agent_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    agent_valid = validate_backtest_history(agent_history, "agent")
    bench_valid = validate_backtest_history(benchmark_history, "benchmark")
    if not agent_valid["valid_backtest"] or not bench_valid["valid_backtest"]:
        return {
            "valid_test": False,
            "invalid_reason": f"agent={agent_valid['invalid_reason']}; benchmark={bench_valid['invalid_reason']}",
            "mean_daily_excess_return": np.nan,
            "bootstrap_p_value_leq_zero": np.nan,
        }
    rng = np.random.default_rng(seed)
    agent = agent_history["equity"].pct_change(fill_method=None).dropna().to_numpy()
    bench = benchmark_history["equity"].pct_change(fill_method=None).dropna().to_numpy()
    n = min(len(agent), len(bench))
    if n == 0:
        return {
            "valid_test": False,
            "invalid_reason": "no overlapping returns",
            "mean_daily_excess_return": np.nan,
            "bootstrap_p_value_leq_zero": np.nan,
        }
    diff = agent[:n] - bench[:n]
    observed = diff.mean()
    samples = [rng.choice(diff, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    samples = np.array(samples)
    return {
        "valid_test": True,
        "invalid_reason": "",
        "mean_daily_excess_return": float(observed),
        "bootstrap_p_value_leq_zero": float((samples <= 0).mean()),
    }
