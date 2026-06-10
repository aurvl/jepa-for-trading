# JEPA for Trading

Multi-asset JEPA market representation learning plus a direct PPO portfolio
agent.

This branch implements V1:

- prices from `yfinance`;
- macro factors from `macro_data.parquet`;
- per-asset sigma estimation from each asset history;
- 70-day market windows;
- JEPA self-supervised latent prediction for horizons `5, 15, 20, 45, 60`;
- supervised market heads for returns, volatility, drawdown and quantiles;
- long-only PPO portfolio agent with cash, transaction costs, max weights and
  tradable-asset masks;
- evaluation against Buy & Hold, equal weight, momentum, volatility targeting
  and random strategies.

The action does not pretend to move the market. Market latents are predicted
from market context; actions only affect portfolio exposure, costs, PnL,
turnover and drawdown.

## Local Setup

```bash
pip install -e ".[dev]"
pytest -q
```

Place `macro_data.parquet` at:

```text
data/raw/macro_data.parquet
```

Then run:

```bash
python scripts/download_data.py
python scripts/train_jepa.py
```

## Kaggle

Use:

```text
notebooks/kaggle_run_jepa_trading.ipynb
```

Upload `macro_data.parquet` as a Kaggle dataset. The notebook auto-discovers it
under `/kaggle/input`.
