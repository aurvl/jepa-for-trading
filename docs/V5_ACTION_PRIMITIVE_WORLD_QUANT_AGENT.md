# V5 Action-Primitive World Quant Agent

V5 is an action-conditioned JEPA world model for portfolio decisions. It is not
a strategy selector. The planner does not choose `momentum`, `equal_weight`,
`risk_parity`, `vol_target` or any handcrafted strategy as an internal action.
Those methods are only baselines used after the run.

## Core Separation

```text
market history + horizon
        |
        v
market world model
        |
        v
predicted future market latent

portfolio state + primitive action + executable action + market latents + horizon
        |
        v
portfolio world model
        |
        v
future portfolio latent + probabilistic outcomes

predicted outcomes + goal
        |
        v
cost / energy model
```

The action never changes the future market latent. In this project there is no
market impact model. The action only changes exposure, turnover, costs, PnL and
risk.

## Primitive Action Space

The model receives abstract trading intentions:

```text
PrimitiveAction:
    asset_scores: vector[n_assets]
    gross_exposure_delta: float
    net_exposure_target: float
    cash_target_delta: float
    risk_budget: float
    rebalance_intensity: float
    horizon: int
    long_short_bias: float
```

These are not strategies. They are controls. A single execution layer converts
them into feasible portfolio weights.

```text
primitive action
        |
        v
proposed target weights
        |
        v
constraint projection:
    tradable mask
    max asset weight
    gross / net exposure
    turnover cap
    long-only / long-short mode
    cash consistency
        |
        v
executable weights
```

The same execution layer is used in:

- dataset construction;
- model training;
- planner inference;
- backtest execution.

That removes the train/backtest action mismatch that polluted earlier versions.

## Probabilistic Outcomes

The world model predicts distributions, not only averages:

```text
return quantiles: q05, q25, q50, q75, q95
drawdown quantiles
volatility
turnover
transaction cost
CVaR / expected shortfall proxy
probability of loss
probability of drawdown breach
future equity ratio
```

Training uses:

- JEPA market latent loss;
- JEPA portfolio latent loss;
- VICReg anti-collapse regularization;
- pinball loss for quantiles;
- BCE for probability heads;
- SmoothL1 for risk and cost heads;
- pairwise ranking loss on realized action cost.

## MPC / CEM Planner

At each date the planner performs a local search over primitive actions:

```text
encode current market window
encode current portfolio state

for cem_iter:
    sample primitive actions
    project to executable weights
    predict outcomes with world model
    score outcomes with cost model
    keep elites
    update primitive sampling distribution

execute the best action only
replan tomorrow
```

This is model-based planning. It does not require the agent to invent portfolio
weights from scratch without constraints, but it also does not spoon-feed it a
strategy rule.

## Risk Tools

Risk tools are explicit modules:

- `risk_estimator.py`: EWMA vol, rolling vol, covariance, shrinkage covariance,
  correlation, beta, VaR, CVaR and drawdown state.
- `monte_carlo.py`: block bootstrap, gaussian and simple student-t portfolio
  simulations.
- `stress.py`: one-day, volatility and correlation shock summaries.
- `regime.py`: compact risk-regime features.

They do not choose actions. They provide diagnostics and extensible perception
or validation hooks.

## Experience Memory

V5 writes an experience buffer:

```text
state_t
primitive_action_t
executable_weights_t
predicted outcomes
predicted cost
realized outcomes
realized cost
prediction error
portfolio state t / t+1
```

This makes failure analysis concrete. If the agent collapses to cash, saturates
turnover, chooses one horizon, or mispredicts risk, the output files expose it.

## Degenerate Backtests

The run is invalid if it contains:

- NaN or infinite values;
- zero or negative equity;
- near-zero exposure across the run;
- zero-trade behavior;
- cash collapse presented as success;
- one-horizon collapse without explicit interpretation;
- turnover saturation.

The notebook must not print a positive interpretation when `validity.csv`
marks the backtest invalid.
