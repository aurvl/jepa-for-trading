from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskSnapshot:
    ewma_vol: np.ndarray
    rolling_vol: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    shrinkage_covariance: np.ndarray
    beta_to_market: np.ndarray
    var_95: np.ndarray
    cvar_95: np.ndarray
    drawdown_state: np.ndarray
    avg_correlation: float

    def feature_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.ewma_vol,
                self.rolling_vol,
                self.beta_to_market,
                self.var_95,
                self.cvar_95,
                self.drawdown_state,
                np.array([self.avg_correlation], dtype=np.float32),
            ]
        ).astype(np.float32)


def _safe_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.maximum(np.outer(std, std), 1e-12)
    return np.nan_to_num(np.clip(corr, -1.0, 1.0), nan=0.0)


def estimate_risk_snapshot(
    returns_window: np.ndarray,
    close_window: np.ndarray | None = None,
    ewma_lambda: float = 0.94,
) -> RiskSnapshot:
    r = np.nan_to_num(np.asarray(returns_window, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if r.ndim != 2:
        raise ValueError("returns_window must be time x assets")
    n_assets = r.shape[1]
    rolling_vol = r.std(axis=0) * np.sqrt(252)
    weights = np.array([(1 - ewma_lambda) * ewma_lambda ** i for i in range(len(r) - 1, -1, -1)], dtype=np.float64)
    weights /= max(weights.sum(), 1e-12)
    ewma_var = (weights[:, None] * r**2).sum(axis=0) * 252
    ewma_vol = np.sqrt(np.maximum(ewma_var, 0.0))
    cov = np.cov(r.T) * 252 if len(r) > 1 else np.eye(n_assets) * 1e-6
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.diag(np.diag(cov))
    shrink = 0.85 * cov + 0.15 * diag
    corr = _safe_corr(cov)
    avg_corr = float((corr.sum() - np.trace(corr)) / max(n_assets * (n_assets - 1), 1))
    market = r.mean(axis=1)
    market_var = float(np.var(market) + 1e-12)
    beta = np.array([np.cov(r[:, i], market)[0, 1] / market_var if len(r) > 1 else 0.0 for i in range(n_assets)])
    var_95 = np.quantile(r, 0.05, axis=0)
    cvar_95 = np.array([r[r[:, i] <= var_95[i], i].mean() if np.any(r[:, i] <= var_95[i]) else var_95[i] for i in range(n_assets)])
    if close_window is not None:
        close = np.nan_to_num(np.asarray(close_window, dtype=np.float64), nan=np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = close / np.nanmax(close, axis=0, keepdims=True) - 1.0
        drawdown = np.nan_to_num(dd[-1], nan=0.0, posinf=0.0, neginf=0.0)
    else:
        path = np.exp(np.cumsum(r, axis=0))
        drawdown = path[-1] / np.maximum(np.maximum.accumulate(path, axis=0)[-1], 1e-12) - 1.0
    return RiskSnapshot(
        ewma_vol=ewma_vol.astype(np.float32),
        rolling_vol=rolling_vol.astype(np.float32),
        covariance=shrink.astype(np.float32),
        correlation=corr.astype(np.float32),
        shrinkage_covariance=shrink.astype(np.float32),
        beta_to_market=beta.astype(np.float32),
        var_95=var_95.astype(np.float32),
        cvar_95=cvar_95.astype(np.float32),
        drawdown_state=drawdown.astype(np.float32),
        avg_correlation=avg_corr,
    )
