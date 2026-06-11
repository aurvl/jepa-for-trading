from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from jepa_trading.data.dataset import MultiAssetJEPADataset, build_market_arrays
from jepa_trading.data.v2_dataset import PortfolioActionConfig, V2WorldModelDataset
from jepa_trading.data.v3_dataset import V3WorldModelDataset
from jepa_trading.data.download import download_prices
from jepa_trading.data.features import add_asset_features, infer_feature_columns
from jepa_trading.data.macro import load_macro, merge_macro
from jepa_trading.data.scaling import fit_transform_train_only
from jepa_trading.data.splits import add_time_split, split_dates


def prepare_market_data(config: dict, force_download: bool = False) -> tuple[pd.DataFrame, object, list[str]]:
    data_cfg = config["data"]
    cache_path = Path(data_cfg["cache_dir"]) / "prices.parquet"
    prices = download_prices(
        data_cfg["tickers"],
        start=data_cfg["start"],
        end=data_cfg.get("end"),
        cache_path=cache_path,
        force=force_download,
    )
    macro = load_macro(data_cfg["macro_path"])
    merged = merge_macro(prices, macro)
    featured = add_asset_features(merged)
    split = add_time_split(featured, data_cfg["train_end"], data_cfg["val_end"])
    split["sigma_log"] = 0.0
    candidate_cols = infer_feature_columns(split)
    scaled, _ = fit_transform_train_only(split, candidate_cols)
    feature_columns = infer_feature_columns(scaled)
    arrays = build_market_arrays(scaled, feature_columns)
    return scaled, arrays, feature_columns


def create_jepa_dataloaders(config: dict, arrays, batch_size: int | None = None) -> dict[str, DataLoader]:
    batch = batch_size or config["training"]["batch_size"]
    lookback = config["data"]["lookback"]
    horizons = config["data"]["horizons"]
    # Split by the global calendar dates already marked in the prepared frame.
    dates = arrays.dates
    train_end = pd.Timestamp(config["data"]["train_end"])
    val_end = pd.Timestamp(config["data"]["val_end"])
    train_dates = dates[dates <= train_end]
    val_dates = dates[(dates > train_end) & (dates <= val_end)]
    test_dates = dates[dates > val_end]
    datasets = {
        "train": MultiAssetJEPADataset(arrays, train_dates, lookback, horizons),
        "val": MultiAssetJEPADataset(arrays, val_dates, lookback, horizons),
        "test": MultiAssetJEPADataset(arrays, test_dates, lookback, horizons),
    }
    return {
        split: DataLoader(ds, batch_size=batch, shuffle=(split == "train"), drop_last=(split == "train"))
        for split, ds in datasets.items()
    }


def create_v2_dataloaders(config: dict, arrays, batch_size: int | None = None) -> dict[str, DataLoader]:
    batch = batch_size or config["training"]["batch_size"]
    lookback = config["data"]["lookback"]
    horizons = config["data"]["horizons"]
    dates = arrays.dates
    train_end = pd.Timestamp(config["data"]["train_end"])
    val_end = pd.Timestamp(config["data"]["val_end"])
    train_dates = dates[dates <= train_end]
    val_dates = dates[(dates > train_end) & (dates <= val_end)]
    test_dates = dates[dates > val_end]
    portfolio_cfg = config["portfolio"]
    action_cfg = PortfolioActionConfig(
        mode=portfolio_cfg.get("mode", "long_only"),
        cash_initial=portfolio_cfg["cash_initial"],
        transaction_cost_bps=portfolio_cfg["transaction_cost_bps"],
        max_long_weight=portfolio_cfg.get("max_long_weight", portfolio_cfg.get("max_weight_per_asset", 0.15)),
        max_short_weight=portfolio_cfg.get("max_short_weight", 0.05),
        max_gross_exposure=portfolio_cfg.get("max_gross_exposure", 1.0),
        max_net_exposure=portfolio_cfg.get("max_net_exposure", 1.0),
        borrow_cost_bps=portfolio_cfg.get("borrow_cost_bps", 2.0),
        n_action_samples=config["v2"]["n_action_samples"],
        derisk_fraction=portfolio_cfg.get("derisk_fraction", 0.50),
    )
    datasets = {
        "train": V2WorldModelDataset(arrays, train_dates, lookback, horizons, action_cfg, seed=config["seed"]),
        "val": V2WorldModelDataset(arrays, val_dates, lookback, horizons, action_cfg, seed=config["seed"] + 1),
        "test": V2WorldModelDataset(arrays, test_dates, lookback, horizons, action_cfg, seed=config["seed"] + 2),
    }
    return {
        split: DataLoader(ds, batch_size=batch, shuffle=(split == "train"), drop_last=(split == "train"))
        for split, ds in datasets.items()
    }


def create_v3_dataloaders(config: dict, arrays, batch_size: int | None = None) -> dict[str, DataLoader]:
    batch = batch_size or config["training"]["batch_size"]
    lookback = config["data"]["lookback"]
    horizons = config["data"]["horizons"]
    dates = arrays.dates
    train_end = pd.Timestamp(config["data"]["train_end"])
    val_end = pd.Timestamp(config["data"]["val_end"])
    train_dates = dates[dates <= train_end]
    val_dates = dates[(dates > train_end) & (dates <= val_end)]
    test_dates = dates[dates > val_end]
    portfolio_cfg = config["portfolio"]
    action_cfg = PortfolioActionConfig(
        mode=portfolio_cfg.get("mode", "long_only"),
        cash_initial=portfolio_cfg["cash_initial"],
        transaction_cost_bps=portfolio_cfg["transaction_cost_bps"],
        max_long_weight=portfolio_cfg.get("max_long_weight", portfolio_cfg.get("max_weight_per_asset", 0.15)),
        max_short_weight=portfolio_cfg.get("max_short_weight", 0.05),
        max_gross_exposure=portfolio_cfg.get("max_gross_exposure", 1.0),
        max_net_exposure=portfolio_cfg.get("max_net_exposure", 1.0),
        borrow_cost_bps=portfolio_cfg.get("borrow_cost_bps", 2.0),
        n_action_samples=config["v3"]["n_sampled_actions"],
        derisk_fraction=portfolio_cfg.get("derisk_fraction", 0.50),
    )
    datasets = {
        "train": V3WorldModelDataset(
            arrays, train_dates, lookback, horizons, action_cfg, seed=config["seed"], n_sampled_actions=config["v3"]["n_sampled_actions"]
        ),
        "val": V3WorldModelDataset(
            arrays, val_dates, lookback, horizons, action_cfg, seed=config["seed"] + 1, n_sampled_actions=config["v3"]["n_sampled_actions"]
        ),
        "test": V3WorldModelDataset(
            arrays, test_dates, lookback, horizons, action_cfg, seed=config["seed"] + 2, n_sampled_actions=config["v3"]["n_sampled_actions"]
        ),
    }
    return {
        split: DataLoader(ds, batch_size=batch, shuffle=(split == "train"), drop_last=(split == "train"))
        for split, ds in datasets.items()
    }
