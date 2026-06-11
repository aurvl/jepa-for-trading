from __future__ import annotations

from pathlib import Path

from jepa_trading.config import ensure_dirs, load_config
from jepa_trading.data.pipeline import create_v2_dataloaders, prepare_market_data
from jepa_trading.models.world_model_v2 import V2WorldModel
from jepa_trading.training.train_v2 import train_v2_world_model
from jepa_trading.utils.device import get_device
from jepa_trading.utils.seed import seed_everything


def main() -> None:
    config = load_config()
    ensure_dirs(config)
    seed_everything(config["seed"])
    device = get_device(config["device"])
    _, arrays, feature_columns = prepare_market_data(config)
    loaders = create_v2_dataloaders(config, arrays)
    sample = next(iter(loaders["train"]))
    model = V2WorldModel(
        n_features=len(feature_columns),
        max_assets=len(arrays.tickers),
        portfolio_state_dim=sample["portfolio_state"].shape[-1],
        action_dim=sample["action"].shape[-1],
        **config["model"],
        ema_decay=config["training"]["ema_decay"],
        hidden_dim=config["v2"]["hidden_dim"],
        policy_mode=config["portfolio"]["mode"],
        max_abs_weight=config["portfolio"]["max_long_weight"],
    )
    hist = train_v2_world_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        max_steps=config["v2"]["world_max_steps"],
        warmup_steps=config["v2"]["world_warmup_steps"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        checkpoint_path=Path(config["training"]["checkpoint_dir"]) / "v2_world_model.pt",
        weights=config["v2"]["loss_weights"],
        eval_every=config["training"]["eval_every"],
        log_every=config["training"]["log_every"],
    )
    hist.to_csv("logs/v2_world_history.csv", index=False)


if __name__ == "__main__":
    main()

