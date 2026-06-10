from __future__ import annotations

import numpy as np
import pandas as pd


def randomization_p_value(agent_history: pd.DataFrame, random_histories: list[pd.DataFrame]) -> dict[str, float]:
    agent_total = agent_history["equity"].iloc[-1] / agent_history["equity"].iloc[0] - 1.0
    random_totals = np.array([h["equity"].iloc[-1] / h["equity"].iloc[0] - 1.0 for h in random_histories])
    return {
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
    rng = np.random.default_rng(seed)
    agent = agent_history["equity"].pct_change().dropna().to_numpy()
    bench = benchmark_history["equity"].pct_change().dropna().to_numpy()
    n = min(len(agent), len(bench))
    diff = agent[:n] - bench[:n]
    observed = diff.mean()
    samples = [rng.choice(diff, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    samples = np.array(samples)
    return {
        "mean_daily_excess_return": float(observed),
        "bootstrap_p_value_leq_zero": float((samples <= 0).mean()),
    }

