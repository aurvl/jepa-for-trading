from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def run_policy_backtest(env, policy, device: torch.device) -> pd.DataFrame:
    obs, mask = env.reset()
    done = False
    while not done:
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask[None], dtype=torch.bool, device=device)
        action, _, _, _ = policy.get_action_and_value(obs_t, mask_t)
        obs, _, done, _, mask = env.step(action.squeeze(0).cpu().numpy())
    return pd.DataFrame(env.history)


def run_weight_strategy(env, weight_fn) -> pd.DataFrame:
    obs, mask = env.reset()
    done = False
    while not done:
        action = weight_fn(env, mask)
        obs, _, done, _, mask = env.step(action)
    return pd.DataFrame(env.history)


def buy_and_hold_weight(env, mask) -> np.ndarray:
    if len(env.history) == 0:
        weights = np.zeros(env.n_assets + 1, dtype=np.float32)
        valid = np.where(mask)[0]
        if len(valid):
            weights[valid] = min(1.0 / len(valid), env.max_weight)
            weights[-1] = max(1.0 - weights[:-1].sum(), 0.0)
        else:
            weights[-1] = 1.0
        return weights
    return env.state.weights


def equal_weight(env, mask) -> np.ndarray:
    weights = np.zeros(env.n_assets + 1, dtype=np.float32)
    valid = np.where(mask)[0]
    if len(valid):
        weights[valid] = min(1.0 / len(valid), env.max_weight)
        weights[-1] = max(1.0 - weights[:-1].sum(), 0.0)
    else:
        weights[-1] = 1.0
    return weights


def random_long_only_weight(env, mask, rng: np.random.Generator) -> np.ndarray:
    weights = np.zeros(env.n_assets + 1, dtype=np.float32)
    valid = np.where(mask)[0]
    if len(valid):
        raw = rng.dirichlet(np.ones(len(valid)))
        asset_weights = np.minimum(raw, env.max_weight)
        scale = min(1.0, 1.0 / max(asset_weights.sum(), 1e-8))
        weights[valid] = asset_weights * scale
    weights[-1] = max(1.0 - weights[:-1].sum(), 0.0)
    return weights


def momentum_weight(env, mask, lookback: int = 20) -> np.ndarray:
    start = max(0, env.idx - lookback)
    mom = np.nansum(env.arrays.log_returns[start : env.idx + 1], axis=0)
    mom = np.where(mask, mom, -np.inf)
    weights = np.zeros(env.n_assets + 1, dtype=np.float32)
    selected = np.argsort(mom)[-5:]
    selected = [i for i in selected if np.isfinite(mom[i]) and mom[i] > 0]
    if selected:
        weights[selected] = min(1.0 / len(selected), env.max_weight)
    weights[-1] = max(1.0 - weights[:-1].sum(), 0.0)
    return weights


def volatility_target_weight(env, mask) -> np.ndarray:
    sigma = np.nan_to_num(env.arrays.sigma[env.idx], nan=np.inf)
    inv = np.where(mask, 1.0 / np.maximum(sigma, 1e-6), 0.0)
    weights = np.zeros(env.n_assets + 1, dtype=np.float32)
    if inv.sum() > 0:
        raw = inv / inv.sum()
        weights[:-1] = np.minimum(raw, env.max_weight)
    weights[-1] = max(1.0 - weights[:-1].sum(), 0.0)
    return weights

