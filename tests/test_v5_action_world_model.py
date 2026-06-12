from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from jepa_trading.data.actions import project_portfolio_action
from jepa_trading.data.v2_dataset import PortfolioActionConfig
from jepa_trading.data.v5_dataset import V5ActionWorldModelDataset, V5CostConfig, summarize_v5_batch
from jepa_trading.evaluation.metrics import validate_backtest_history
from jepa_trading.models.world_model_v5 import V5ActionConditionedWorldModel
from jepa_trading.planning.v5_planner import V5ActionWorldModelPlanner
from jepa_trading.training.train_v5 import v5_world_model_loss
from tests.test_v2_world_model import _synthetic_arrays


def test_v5_dataset_loss_and_planner_are_finite():
    arrays, cols = _synthetic_arrays()
    action_cfg = PortfolioActionConfig(n_action_samples=3, max_long_weight=0.4)
    cost_cfg = V5CostConfig(annual_return_target=0.10)
    ds = V5ActionWorldModelDataset(
        arrays,
        arrays.dates[:120],
        lookback=30,
        horizons=[5, 10],
        action_config=action_cfg,
        cost_config=cost_cfg,
        n_sampled_actions=3,
        max_turnover=0.4,
    )
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=3)))
    summary = summarize_v5_batch(batch)
    assert summary["outcomes_finite"]
    assert summary["costs_finite"]
    assert summary["n_candidates"] >= 7

    model = V5ActionConditionedWorldModel(
        n_features=len(cols),
        max_assets=len(arrays.tickers),
        portfolio_state_dim=batch["portfolio_state"].shape[-1],
        action_dim=batch["actions"].shape[-1],
        goal_dim=batch["goal"].shape[-1],
        d_model=32,
        latent_dim=32,
        n_heads=4,
        n_layers=1,
        hidden_dim=64,
    )
    outputs = model.forward_candidates(batch)
    losses = v5_world_model_loss(
        outputs,
        batch,
        step=10,
        warmup_steps=5,
        weights={
            "market_jepa": 1.0,
            "portfolio_jepa": 1.0,
            "vicreg": 0.05,
            "outcome": 1.0,
            "cost": 0.5,
            "rank": 0.7,
        },
    )
    assert outputs["outcome_hat"].shape[:2] == batch["actions"].shape[:2]
    assert torch.isfinite(losses["loss"])

    sample = ds[0]
    planner = V5ActionWorldModelPlanner(
        model,
        action_cfg,
        cost_cfg,
        horizons=[5, 10],
        n_sampled_actions=3,
        max_turnover=0.4,
        device="cpu",
    )
    result = planner.choose_action(
        sample["context"].numpy(),
        sample["context_mask"].numpy(),
        sample["portfolio_state"].numpy(),
        sample["tradable_mask"].numpy(),
        arrays.sigma[29],
        arrays.log_returns[:30].sum(axis=0),
    )
    assert result.action.shape[0] == len(arrays.tickers) + 1
    assert result.horizon in [5, 10]
    assert np.isfinite(result.predicted_cost)


def test_v5_projection_and_degenerate_backtest_guard():
    current = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    proposed = np.array([0.8, 0.8, 0.8, 0.0], dtype=np.float32)
    valid = np.array([True, True, False])
    projected = project_portfolio_action(
        current,
        proposed,
        valid,
        max_long_weight=0.4,
        max_turnover=0.4,
    )
    assert projected[2] == 0.0
    assert np.abs(projected - current).sum() <= 0.4001
    assert np.isclose(projected.sum(), 1.0)

    dead = pd.DataFrame(
        {
            "equity": [1000.0, 1000.0, 1000.0],
            "turnover": [0.0, 0.0, 0.0],
            "gross_exposure": [0.0, 0.0, 0.0],
            "selected_action_name": ["cash", "cash", "cash"],
        }
    )
    validity = validate_backtest_history(dead, "dead_agent")
    assert validity["valid_backtest"] is False
