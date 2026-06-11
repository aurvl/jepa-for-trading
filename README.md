# JEPA for Trading

Multi-asset JEPA market representation learning plus a direct PPO portfolio
agent.

This repository keeps the experimental path explicit:

- `version1`: JEPA representation learning + market heads + PPO policy.
- `version2`: unified JEPA world model + VICReg + action-conditioned portfolio
  imagination + MLP planner policy.

The current branch implements V2 while keeping V1 notebooks executable.

V1 includes:

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

V2 adds:

- one-shot world model training;
- VICReg anti-collapse regularization;
- counterfactual portfolio actions sampled per training window;
- action-conditioned outcome prediction;
- energy/utility scoring;
- imagination planner over candidate actions and horizons.

## Kaggle

Use V1:

```text
notebooks/01_kaggle_v1_jepa_ppo.ipynb
```

Use V2:

```text
notebooks/02_kaggle_v2_world_model_planner.ipynb
```

Upload `macro_data.parquet` as a Kaggle dataset. The notebook auto-discovers it
under `/kaggle/input`.
