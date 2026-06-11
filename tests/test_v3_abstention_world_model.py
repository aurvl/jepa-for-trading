from __future__ import annotations

import numpy as np
import torch

from jepa_trading.data.v2_dataset import PortfolioActionConfig
from jepa_trading.data.v3_dataset import V3WorldModelDataset, summarize_v3_batch
from jepa_trading.models.world_model_v2 import V2WorldModel
from jepa_trading.planning.v3_planner import V3AbstentionPlanner
from jepa_trading.training.train_v3 import v3_forward_candidates, v3_world_model_loss
from tests.test_v2_world_model import _synthetic_arrays


def test_v3_dataset_has_defensive_candidates():
    arrays, _ = _synthetic_arrays()
    action_cfg = PortfolioActionConfig(n_action_samples=2, max_long_weight=0.4, derisk_fraction=0.5)
    ds = V3WorldModelDataset(
        arrays,
        arrays.dates[:120],
        lookback=30,
        horizons=[5, 10],
        action_config=action_cfg,
        n_sampled_actions=3,
    )
    sample = ds[0]
    assert sample["actions"].shape[0] == 8
    assert int(sample["hold_action_index"]) == 0
    assert int(sample["cash_action_index"]) == 1
    assert int(sample["derisk_action_index"]) == 2
    assert torch.isfinite(sample["outcomes"]).all()
    assert torch.isfinite(sample["utilities"]).all()


def test_v3_forward_loss_and_planner_are_finite():
    arrays, cols = _synthetic_arrays()
    action_cfg = PortfolioActionConfig(n_action_samples=2, max_long_weight=0.4, derisk_fraction=0.5)
    ds = V3WorldModelDataset(
        arrays,
        arrays.dates[:120],
        lookback=30,
        horizons=[5, 10],
        action_config=action_cfg,
        n_sampled_actions=3,
    )
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=3)))
    summary = summarize_v3_batch(batch)
    assert summary["outcomes_finite"]
    assert summary["utilities_finite"]

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
    losses = v3_world_model_loss(
        outputs,
        batch,
        step=10,
        warmup_steps=5,
        weights={"jepa": 1.0, "vicreg": 0.05, "outcome": 1.0, "energy": 0.5, "policy": 0.25, "rank": 0.5},
    )
    assert outputs["outcome_hat"].shape == batch["outcomes"].shape
    assert outputs["energy_hat"].shape == batch["utilities"].shape
    assert torch.isfinite(losses["loss"])

    planner = V3AbstentionPlanner(
        model,
        action_cfg,
        [5, 10],
        n_sampled_actions=3,
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
    assert result.selected_name in {"hold", "cash", "derisk", "equal_weight", "vol_target", "sampled_0", "sampled_1", "sampled_2"}
    assert np.isfinite(result.score)
