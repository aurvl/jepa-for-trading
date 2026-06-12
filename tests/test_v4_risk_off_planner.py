from __future__ import annotations

import numpy as np
import torch

from jepa_trading.data.v2_dataset import PortfolioActionConfig
from jepa_trading.data.v4_dataset import V4RiskUtilityConfig, V4WorldModelDataset, summarize_v4_batch
from jepa_trading.models.world_model_v2 import V2WorldModel
from jepa_trading.planning.v4_planner import V4RiskOffPlanner
from jepa_trading.training.train_v3 import v3_forward_candidates
from jepa_trading.training.train_v4 import v4_world_model_loss
from tests.test_v2_world_model import _synthetic_arrays


def test_v4_dataset_loss_and_risk_off_planner_are_finite():
    arrays, cols = _synthetic_arrays()
    action_cfg = PortfolioActionConfig(n_action_samples=2, max_long_weight=0.4, derisk_fraction=0.5)
    utility_cfg = V4RiskUtilityConfig(risk_off_drawdown=-0.01, defensive_bonus=0.2)
    ds = V4WorldModelDataset(
        arrays,
        arrays.dates[:120],
        lookback=30,
        horizons=[5, 10],
        action_config=action_cfg,
        utility_config=utility_cfg,
        n_sampled_actions=3,
    )
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=3)))
    summary = summarize_v4_batch(batch)
    assert summary["outcomes_finite"]
    assert summary["utilities_finite"]
    assert 0.0 <= summary["risk_off_label_rate"] <= 1.0

    model = V2WorldModel(
        n_features=len(cols),
        max_assets=len(arrays.tickers),
        portfolio_state_dim=batch["portfolio_state"].shape[-1],
        action_dim=batch["actions"].shape[-1],
        d_model=32,
        latent_dim=32,
        n_heads=4,
        n_layers=1,
        hidden_dim=64,
        max_abs_weight=0.4,
    )
    outputs = v3_forward_candidates(model, batch)
    losses = v4_world_model_loss(
        outputs,
        batch,
        step=10,
        warmup_steps=5,
        weights={"jepa": 1.0, "vicreg": 0.05, "outcome": 1.0, "energy": 0.5, "policy": 0.25, "rank": 0.7},
    )
    assert torch.isfinite(losses["loss"])

    planner = V4RiskOffPlanner(
        model,
        action_cfg,
        [5, 10],
        n_sampled_actions=3,
        hard_risk_off_drawdown=0.01,
        chunk_size=4,
        device="cpu",
    )
    sample = ds[0]
    result = planner.choose_action(
        sample["context"].numpy(),
        sample["context_mask"].numpy(),
        sample["portfolio_state"].numpy(),
        sample["tradable_mask"].numpy(),
        arrays.sigma[29],
    )
    assert result.action.shape[0] == len(arrays.tickers) + 1
    assert result.selected_name in {"cash", "derisk"}
    assert result.selected_reason == "hard_risk_off_defensive"
    assert np.isfinite(result.score)
