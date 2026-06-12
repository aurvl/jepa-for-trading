from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from jepa_trading.actions import ExecutionConstraints, PrimitiveExecutionLayer, primitive_action_dim, sample_primitive_actions
from jepa_trading.data.v2_dataset import PortfolioActionConfig
from jepa_trading.data.v5_dataset import V5ActionWorldModelDataset, V5CostConfig, summarize_v5_batch
from jepa_trading.evaluation.v5_backtest import validate_v5_backtest
from jepa_trading.models.world_model_v5 import V5ActionConditionedWorldModel
from jepa_trading.planning.v5_planner import V5ActionWorldModelPlanner
from jepa_trading.risk import estimate_risk_snapshot, simulate_portfolio_returns, stress_scenarios
from jepa_trading.training.train_v5 import v5_world_model_loss
from tests.test_v2_world_model import _synthetic_arrays


def test_primitive_execution_respects_constraints():
    rng = np.random.default_rng(123)
    valid = np.array([True, True, False, True])
    current = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    primitives = sample_primitive_actions(rng, n_assets=4, horizons=[5, 20], n_actions=8)
    layer = PrimitiveExecutionLayer(ExecutionConstraints(max_long_weight=0.35, max_turnover=0.4))
    for primitive in primitives.vectors:
        weights = layer.execute(primitive, current, valid)
        assert np.isfinite(weights).all()
        assert weights[2] == 0.0
        assert np.abs(weights - current).sum() <= 0.4001
        assert weights[:-1].max() <= 0.3501
        assert np.isclose(weights.sum(), 1.0)


def test_risk_tools_are_finite():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0002, 0.01, size=(90, 5)).astype(np.float32)
    weights = np.array([0.15, 0.10, 0.05, 0.20, 0.10, 0.40], dtype=np.float32)
    snap = estimate_risk_snapshot(returns)
    mc = simulate_portfolio_returns(returns, weights, horizon=20, n_paths=64, seed=7)
    stress = stress_scenarios(returns, weights)
    assert np.isfinite(snap.ewma_vol).all()
    assert np.isfinite(mc.path_equity).all()
    assert all(np.isfinite(v) for v in mc.summary.values())
    assert all(np.isfinite(v) for v in stress.values())


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
        n_sampled_actions=5,
        max_turnover=0.4,
    )
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=3)))
    summary = summarize_v5_batch(batch)
    assert summary["outcomes_finite"]
    assert summary["costs_finite"]
    assert summary["primitive_actions_finite"]
    assert summary["executable_actions_finite"]
    assert summary["n_candidates"] == 5

    model = V5ActionConditionedWorldModel(
        n_features=len(cols),
        max_assets=len(arrays.tickers),
        portfolio_state_dim=batch["portfolio_state"].shape[-1],
        primitive_action_dim=primitive_action_dim(len(arrays.tickers)),
        executable_action_dim=len(arrays.tickers) + 1,
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
            "return_quantile": 1.0,
            "drawdown_quantile": 1.0,
            "outcome_aux": 0.5,
            "probability": 0.5,
            "cost": 0.5,
            "rank": 0.7,
        },
    )
    assert outputs["outcome_hat"]["return_quantiles"].shape[:2] == batch["primitive_actions"].shape[:2]
    assert torch.isfinite(losses["loss"])

    sample = ds[0]
    planner = V5ActionWorldModelPlanner(
        model,
        action_cfg,
        cost_cfg,
        horizons=[5, 10],
        n_sampled_actions=12,
        cem_iters=2,
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
    assert result.primitive_action.shape[0] == primitive_action_dim(len(arrays.tickers))
    assert result.executable_action.shape[0] == len(arrays.tickers) + 1
    assert result.horizon in [5, 10]
    assert np.isfinite(result.predicted_cost)


def test_v5_degenerate_backtest_guard():
    dead = pd.DataFrame(
        {
            "equity": [1000.0, 1000.0, 1000.0],
            "turnover": [0.0, 0.0, 0.0],
            "gross_exposure": [0.0, 0.0, 0.0],
            "selected_action_name": ["primitive_cem", "primitive_cem", "primitive_cem"],
        }
    )
    validity = validate_v5_backtest(dead, "dead_agent")
    assert validity["valid_backtest"].iloc[0] is False or validity["valid_backtest"].iloc[0] == False
