from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from jepa_trading.data.dataset import MultiAssetJEPADataset, build_market_arrays
from jepa_trading.data.features import add_asset_features, infer_feature_columns
from jepa_trading.data.scaling import fit_transform_train_only
from jepa_trading.data.splits import add_time_split
from jepa_trading.models.heads import MarketHeads
from jepa_trading.models.jepa import MarketJEPA
from jepa_trading.models.policy import PortfolioPolicy
from jepa_trading.rl.env import TradingEnv
from jepa_trading.rl.observer import RawMarketObserver
from jepa_trading.evaluation.metrics import equity_metrics, validate_backtest_history
from jepa_trading.evaluation.statistical_tests import randomization_p_value


def _synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=180)
    rows = []
    for k, ticker in enumerate(["AAA", "BBB", "CCC"]):
        start = k * 20
        price = 100 + 10 * k
        for date in dates[start:]:
            ret = rng.normal(0.0005, 0.015)
            prev = price
            price = price * np.exp(ret)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": prev,
                    "high": max(prev, price) * 1.01,
                    "low": min(prev, price) * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000 + rng.integers(0, 50_000),
                    "theme_growth_score": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


def _arrays():
    df = add_asset_features(_synthetic_prices())
    df = add_time_split(df, "2020-05-31", "2020-07-31")
    df["sigma_log"] = 0.0
    cols = infer_feature_columns(df)
    scaled, _ = fit_transform_train_only(df, cols)
    cols = infer_feature_columns(scaled)
    return build_market_arrays(scaled, cols), cols


def test_dataset_and_jepa_shapes():
    arrays, cols = _arrays()
    ds = MultiAssetJEPADataset(arrays, arrays.dates[:120], lookback=30, horizons=[5, 15])
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
    model = MarketJEPA(n_features=len(cols), max_assets=len(arrays.tickers), d_model=32, latent_dim=32, n_heads=4, n_layers=1)
    z_hat, z_target, z_context = model(
        batch["context"],
        batch["target"],
        batch["horizon"],
        batch["context_mask"],
        batch["target_mask"],
    )
    assert z_hat.shape == z_target.shape == z_context.shape == (4, len(arrays.tickers), 32)


def test_policy_env_step_keeps_weights_valid():
    arrays, cols = _arrays()
    observer = RawMarketObserver(arrays, lookback=30)
    env = TradingEnv(
        arrays,
        observer,
        start_date=arrays.dates[40],
        end_date=arrays.dates[80],
        lookback=30,
        max_weight_per_asset=0.4,
    )
    obs, mask = env.reset()
    policy = PortfolioPolicy(obs_dim=len(obs), n_assets=len(arrays.tickers), hidden_dim=32, max_weight=0.4)
    action, _, _, value = policy.get_action_and_value(
        torch.tensor(obs[None], dtype=torch.float32),
        torch.tensor(mask[None], dtype=torch.bool),
    )
    next_obs, reward, done, info, next_mask = env.step(action.squeeze(0).detach().numpy())
    assert np.isfinite(reward)
    assert np.isclose(env.state.weights.sum(), 1.0)
    assert np.all(env.state.weights[:-1] <= 0.4001)
    assert next_obs.shape == obs.shape


def test_invalid_backtest_is_not_interpretable():
    bad = pd.DataFrame({"equity": [1000.0, np.nan, 1200.0], "cost": [0.0, 0.0, np.inf]})
    valid = validate_backtest_history(bad, "bad")
    metrics = equity_metrics(bad)
    test = randomization_p_value(bad, [bad])
    assert valid["valid_backtest"] is False
    assert metrics["valid_backtest"] is False
    assert test["valid_test"] is False
