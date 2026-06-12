from __future__ import annotations

import numpy as np


def project_portfolio_action(
    current_weights: np.ndarray,
    proposed_action: np.ndarray,
    tradable_mask: np.ndarray,
    *,
    mode: str = "long_only",
    max_long_weight: float = 0.15,
    max_short_weight: float = 0.05,
    max_gross_exposure: float = 1.0,
    max_net_exposure: float = 1.0,
    max_turnover: float | None = None,
) -> np.ndarray:
    """Project a proposed portfolio action into the executable constraint set."""
    current = np.asarray(current_weights, dtype=np.float32)
    action = np.asarray(proposed_action, dtype=np.float32).copy()
    valid = np.asarray(tradable_mask, dtype=bool)
    n_assets = len(valid)
    if action.shape[0] != n_assets + 1:
        raise ValueError(f"Expected action size {n_assets + 1}, got {action.shape[0]}")
    if current.shape[0] != n_assets + 1:
        raise ValueError(f"Expected current_weights size {n_assets + 1}, got {current.shape[0]}")

    if mode == "long_only":
        action[:-1] = np.where(valid, np.clip(action[:-1], 0.0, max_long_weight), 0.0)
        action[-1] = max(float(action[-1]), 0.0)
        total = float(action.sum())
        if total <= 1e-8:
            action[:] = 0.0
            action[-1] = 1.0
        else:
            action /= total
    elif mode in {"long_short", "market_neutral"}:
        action[:-1] = np.where(valid, np.clip(action[:-1], -max_short_weight, max_long_weight), 0.0)
        if mode == "market_neutral":
            tradable_active = valid & (np.abs(action[:-1]) > 0)
            if tradable_active.any():
                action[:-1][tradable_active] -= action[:-1][tradable_active].mean()
        gross = float(np.abs(action[:-1]).sum())
        if gross > max_gross_exposure:
            action[:-1] *= max_gross_exposure / max(gross, 1e-8)
        net = float(action[:-1].sum())
        if abs(net) > max_net_exposure:
            action[:-1] *= max_net_exposure / max(abs(net), 1e-8)
        action[-1] = max(1.0 - float(np.abs(action[:-1]).sum()), 0.0)
    else:
        raise ValueError(f"Unknown portfolio mode: {mode}")

    if max_turnover is not None:
        turnover = float(np.abs(action - current).sum())
        if turnover > max_turnover:
            alpha = max_turnover / max(turnover, 1e-8)
            action = current + alpha * (action - current)
            if mode == "long_only":
                action = np.clip(action, 0.0, None)
                action /= max(float(action.sum()), 1e-8)
            else:
                action = project_portfolio_action(
                    current,
                    action,
                    valid,
                    mode=mode,
                    max_long_weight=max_long_weight,
                    max_short_weight=max_short_weight,
                    max_gross_exposure=max_gross_exposure,
                    max_net_exposure=max_net_exposure,
                    max_turnover=None,
                )
    return action.astype(np.float32)


def portfolio_state_features(
    weights: np.ndarray,
    equity: float = 1.0,
    peak_equity: float = 1.0,
    last_turnover: float = 0.0,
    recent_log_return: float = 0.0,
    recent_vol: float = 0.0,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    drawdown = float(equity / max(peak_equity, 1e-8) - 1.0)
    extras = np.array(
        [
            float(equity),
            drawdown,
            float(last_turnover),
            float(weights[-1]),
            float(np.abs(weights[:-1]).sum()),
            float(np.maximum(weights[:-1], 0.0).sum()),
            float(np.abs(np.minimum(weights[:-1], 0.0)).sum()),
            float(recent_log_return),
            float(recent_vol),
        ],
        dtype=np.float32,
    )
    return np.concatenate([weights.astype(np.float32), extras])
