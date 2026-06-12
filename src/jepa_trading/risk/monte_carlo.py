from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonteCarloResult:
    path_returns: np.ndarray
    path_equity: np.ndarray
    summary: dict[str, float]


def _max_drawdown(path: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(path, axis=1)
    return np.min(path / np.maximum(peak, 1e-8) - 1.0, axis=1)


def simulate_portfolio_returns(
    returns_window: np.ndarray,
    weights: np.ndarray,
    horizon: int,
    n_paths: int = 256,
    mode: str = "block_bootstrap",
    block_size: int = 5,
    seed: int | None = None,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    returns = np.nan_to_num(np.asarray(returns_window, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.asarray(weights, dtype=np.float32)
    asset_weights = weights[:-1] if weights.shape[0] == returns.shape[1] + 1 else weights
    if returns.ndim != 2 or returns.shape[0] == 0:
        path_returns = np.zeros((n_paths, horizon), dtype=np.float32)
    elif mode == "gaussian":
        mu = returns.mean(axis=0)
        cov = np.cov(returns, rowvar=False) + np.eye(returns.shape[1]) * 1e-8
        draws = rng.multivariate_normal(mu, cov, size=(n_paths, horizon))
        path_returns = draws @ asset_weights
    elif mode == "student_t":
        mu = returns.mean(axis=0)
        cov = np.cov(returns, rowvar=False) + np.eye(returns.shape[1]) * 1e-8
        draws = rng.multivariate_normal(np.zeros_like(mu), cov, size=(n_paths, horizon))
        scale = np.sqrt(rng.chisquare(df=5, size=(n_paths, horizon, 1)) / 5)
        path_returns = (mu + draws / np.maximum(scale, 1e-8)) @ asset_weights
    else:
        path_returns = np.zeros((n_paths, horizon), dtype=np.float32)
        for p in range(n_paths):
            chunks = []
            while sum(len(c) for c in chunks) < horizon:
                start = int(rng.integers(0, max(1, returns.shape[0] - block_size + 1)))
                chunks.append(returns[start : start + block_size])
            sampled = np.concatenate(chunks, axis=0)[:horizon]
            path_returns[p] = sampled @ asset_weights

    path_returns = np.nan_to_num(path_returns, nan=0.0, posinf=0.0, neginf=0.0)
    path_equity = np.exp(np.cumsum(path_returns, axis=1))
    terminal = np.log(np.maximum(path_equity[:, -1], 1e-8))
    drawdowns = _max_drawdown(path_equity)
    tail_n = max(1, int(np.ceil(0.05 * len(terminal))))
    summary = {
        "mean_return": float(np.mean(terminal)),
        "q05": float(np.quantile(terminal, 0.05)),
        "q25": float(np.quantile(terminal, 0.25)),
        "q50": float(np.quantile(terminal, 0.50)),
        "q75": float(np.quantile(terminal, 0.75)),
        "q95": float(np.quantile(terminal, 0.95)),
        "drawdown_q05": float(np.quantile(drawdowns, 0.05)),
        "drawdown_q25": float(np.quantile(drawdowns, 0.25)),
        "drawdown_q50": float(np.quantile(drawdowns, 0.50)),
        "drawdown_q75": float(np.quantile(drawdowns, 0.75)),
        "drawdown_q95": float(np.quantile(drawdowns, 0.95)),
        "volatility": float(np.std(path_returns) * np.sqrt(252.0)),
        "prob_loss": float(np.mean(terminal < 0.0)),
        "prob_drawdown_breach": float(np.mean(drawdowns < -0.12)),
        "cvar": float(np.mean(np.sort(terminal)[:tail_n])),
    }
    return MonteCarloResult(path_returns=path_returns, path_equity=path_equity, summary=summary)
