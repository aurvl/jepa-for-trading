# The Quant Financial Agent: A smart world model for trading

Multi-asset JEPA market representation learning plus action-conditioned
portfolio world-model planning.

This repository keeps the experimental path explicit:

- `version1`: JEPA representation learning + market heads + PPO policy.
- `version2`: unified JEPA world model + VICReg + action-conditioned portfolio
  imagination + MLP planner policy.
- `version3`: V2 plus abstention/risk-off planning with hold, cash and
  de-risk candidates trained by a ranking objective.
- `version4`: V3 plus drawdown-aware utility labels, hard risk-off planner
  overrides, scheduled rechecks, and per-asset weight diagnostics.
- `version5`: action-primitive JEPA world quant agent. Primitive trading
  intentions condition portfolio transitions and costs; the market transition
  remains action-independent.

The current branch implements V5 while keeping V1, V2, V3 and V4 notebooks
executable.

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

## V1 Result

The executed V1 notebook is kept as a baseline/failure record, not as a
successful strategy. The data pipeline ran and the JEPA latent loss decreased,
but the downstream decision stack became numerically invalid:

- final JEPA train loss was around `0.011`, with validation around `0.049`;
- market-head validation loss was around `106`, far above train loss;
- PPO produced `NaN` reward/loss/equity;
- backtest metrics contained `NaN`/`inf`, so the positive statistical
  interpretation printed by the notebook is invalid.

This motivates V2: unified world-model training with VICReg,
action-conditioned portfolio outcomes, and planner-first decisions.

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

## V2 Result

The executed V2 notebook is valid: no `NaN`/`inf` backtest failure was detected.
It fixes the V1 numerical failure, but it is not yet a robust winning trading
strategy.

Key test-period results:

- V2 JEPA Planner total return: `0.843`, Sharpe: `1.520`, max drawdown:
  `-0.146`;
- Buy & Hold total return: `0.171`, Sharpe: `1.832`, max drawdown: `-0.034`;
- Equal Weight total return: `1.144`, Sharpe: `1.828`, max drawdown: `-0.161`;
- randomization test: `p_value_random_beats_agent = 0.26`;
- bootstrap vs Buy & Hold: `p = 0.003`.

Interpretation: V2 is a valid world-model/planner baseline and beats Buy &
Hold on total return, but it does not beat the strongest simple baselines.
The planner saturates the turnover constraint (`avg_turnover = 0.40`) and pays
high transaction costs. Next work should focus on turnover-aware energy,
action smoothness, and horizon diversification.

## V3 Direction

V3 targets the main V2 trading failure: the planner knows how to choose an
allocation, but it does not know strongly enough when to avoid trading. The
new dataset builds, for every market window and horizon, a candidate set that
always includes:

- hold current portfolio;
- move to cash;
- de-risk existing exposure;
- equal weight;
- volatility target;
- sampled portfolio actions.

The world model is still trained in one loop, but now also learns to rank
candidates by realized future utility. At inference, the planner compares the
best imagined trade against hold/cash/de-risk and refuses to trade unless the
score advantage clears a margin. This is meant to reduce forced trading during
bad regimes, lower turnover, and make cash a real action rather than a passive
leftover weight.

## V5 Direction

V5 removes the strategy-selector mistake. The planner no longer chooses among
internal candidates like `momentum`, `equal_weight`, `vol_target` or
`risk_parity`. Those remain evaluation baselines only.

The V5 internal action is a primitive intent:

```text
asset_scores
gross_exposure_delta
net_exposure_target
cash_target_delta
risk_budget
rebalance_intensity
horizon
long_short_bias
```

A single execution layer projects that primitive into executable portfolio
weights under the same constraints in training, planning and backtesting. The
market world model predicts future market latents from market state and
horizon only. The portfolio world model predicts future portfolio latents and
probabilistic outcomes from market latents, portfolio state, primitive action,
executable action and horizon. The goal enters only the cost/energy model; it
does not modify the transition dynamics.

V5 saves diagnostics under `outputs/v5_latest/`, including:

- `agent_history.csv`;
- `planner_diagnostics.csv`;
- `experience_buffer.csv` / `experience_buffer.parquet`;
- `primitive_action_stats.csv`;
- `horizon_distribution.csv`;
- `prediction_error.csv`;
- `calibration_report.csv`;
- `validity.csv`.

Any backtest with non-finite values, near-zero exposure, zero trades, negative
equity or cash collapse must be treated as invalid, not as a successful
defensive strategy.

## Kaggle

Use V1:

```text
notebooks/01_v1_runned_jepa_ppo_failure_analysis.ipynb
```

Use V2:

```text
notebooks/02_v2_world_model_planner.ipynb
```

Use V3:

```text
notebooks/03_v3_world_model_abstention_planner.ipynb
```

Use V4:

```text
notebooks/04_v4_risk_off_world_model_planner.ipynb
```

Use V5:

```text
notebooks/05_v5_action_conditioned_world_model.ipynb
```

Upload `macro_data.parquet` as a Kaggle dataset. The notebook auto-discovers it
under `/kaggle/input`.
