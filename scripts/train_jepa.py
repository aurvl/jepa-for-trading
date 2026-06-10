from __future__ import annotations

from pathlib import Path

from jepa_trading.config import ensure_dirs, load_config
from jepa_trading.data.pipeline import create_jepa_dataloaders, prepare_market_data
from jepa_trading.models.jepa import MarketJEPA
from jepa_trading.training.train_jepa import train_jepa
from jepa_trading.utils.device import get_device
from jepa_trading.utils.seed import seed_everything


def main() -> None:
    config = load_config()
    ensure_dirs(config)
    seed_everything(config["seed"])
    device = get_device(config["device"])
    _, arrays, feature_columns = prepare_market_data(config)
    loaders = create_jepa_dataloaders(config, arrays)
    model = MarketJEPA(
        n_features=len(feature_columns),
        max_assets=len(arrays.tickers),
        **config["model"],
        ema_decay=config["training"]["ema_decay"],
    )
    hist = train_jepa(
        model,
        loaders["train"],
        loaders["val"],
        device,
        max_steps=config["training"]["jepa_max_steps"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        checkpoint_path=Path(config["training"]["checkpoint_dir"]) / "market_jepa.pt",
        eval_every=config["training"]["eval_every"],
        log_every=config["training"]["log_every"],
    )
    hist.to_csv("logs/jepa_history.csv", index=False)


if __name__ == "__main__":
    main()

