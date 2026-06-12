from __future__ import annotations

import numpy as np

from jepa_trading.risk.risk_estimator import RiskSnapshot


def regime_features(snapshot: RiskSnapshot) -> dict[str, float]:
    vol = float(np.nanmean(snapshot.ewma_vol))
    corr = float(snapshot.avg_correlation)
    drawdown = float(snapshot.drawdown_state)
    cvar = float(np.nanmean(snapshot.cvar_95))
    return {
        "avg_ewma_vol": vol,
        "avg_correlation": corr,
        "drawdown_state": drawdown,
        "avg_cvar_95": cvar,
        "high_vol_regime": float(vol > 0.30),
        "high_corr_regime": float(corr > 0.60),
        "risk_off_regime": float(drawdown < -0.08 or vol > 0.35 or corr > 0.70),
    }


def regime_vector(snapshot: RiskSnapshot) -> np.ndarray:
    features = regime_features(snapshot)
    return np.asarray(list(features.values()), dtype=np.float32)
