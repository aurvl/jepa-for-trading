from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from jepa_trading.data.dataset import build_market_arrays
from jepa_trading.data.features import add_asset_features, infer_feature_columns
from jepa_trading.data.scaling import fit_transform_train_only
from jepa_trading.data.splits import add_time_split
from jepa_trading.data.v2_dataset import PortfolioActionConfig, V2WorldModelDataset
from jepa_trading.models.world_model_v2 import V2WorldModel
from jepa_trading.planning.v2_planner import V2ImaginationPlanner
from jepa_trading.training.train_v2 import v2_world_model_loss


def _synthetic_arrays():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=180)
    rows = []
    for k, ticker in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        price = 100 + 5 * k
        for date in dates[k * 10 :]:
            ret = rng.normal(0.0002, 0.012)
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
                    "volume": 1_000_000 + rng.integers(0, 100_000),
                    "theme_growth_score": rng.normal(),
                    "theme_stress_score": rng.normal(),
                }
            )
    df = add_asset_features(pd.DataFrame(rows))
    df = add_time_split(df, "2020-05-31", "2020-07-31")
    df["sigma_log"] = 0.0
    cols = infer_feature_columns(df)
    scaled, _ = fit_transform_train_only(df, cols)
    cols = infer_feature_columns(scaled)
    return build_market_arrays(scaled, cols), cols


def test_v2_forward_loss_and_planner():
    arrays, cols = _synthetic_arrays()
    action_cfg = PortfolioActionConfig(n_action_samples=2, max_long_weight=0.4)
    ds = V2WorldModelDataset(arrays, arrays.dates[:120], 30, [5, 10], action_cfg)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=3)))
    model = V2WorldModel(
        n_features=len(cols),
        max_assets=len(arrays.tickers),
        portfolio_state_dim=batch["portfolio_state"].shape[-1],
        action_dim=batch["action"].shape[-1],
        d_model=32,
        latent_dim=32,
        n_heads=4,
        n_layers=1,
        hidden_dim=64,
        max_abs_weight=0.4,
    )
    outputs = model(batch)
    losses = v2_world_model_loss(
        outputs,
        batch,
        step=10,
        warmup_steps=5,
        weights={"jepa": 1.0, "vicreg": 0.05, "outcome": 1.0, "energy": 0.5, "policy": 0.25},
    )
    assert outputs["outcome_hat"].shape == (3, 6)
    assert outputs["policy_action"].shape == batch["action"].shape
    assert torch.isfinite(losses["loss"])

    planner = V2ImaginationPlanner(model, action_cfg, [5, 10], n_action_samples=4, device="cpu")
    sample = ds[0]
    result = planner.choose_action(
        sample["context"].numpy(),
        sample["context_mask"].numpy(),
        sample["portfolio_state"].numpy(),
        sample["tradable_mask"].numpy(),
    )
    assert result.action.shape[0] == len(arrays.tickers) + 1
    assert result.horizon in [5, 10]
    assert np.isfinite(result.score)

