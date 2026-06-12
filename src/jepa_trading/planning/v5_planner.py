from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jepa_trading.actions import ExecutionConstraints, PrimitiveExecutionLayer, primitive_from_vector, sample_primitive_actions
from jepa_trading.data.actions import portfolio_state_features
from jepa_trading.data.v2_dataset import PortfolioActionConfig
from jepa_trading.data.v5_dataset import V5CostConfig, V5_QUANTILES, goal_vector
from jepa_trading.models.world_model_v5 import V5ActionConditionedWorldModel


@dataclass
class V5PlannerResult:
    primitive_action: np.ndarray
    executable_action: np.ndarray
    horizon: int
    predicted_cost: float
    predicted_outcome: dict[str, float]
    candidate_costs: np.ndarray
    candidate_horizons: np.ndarray
    diagnostics: dict[str, float]

    @property
    def action(self) -> np.ndarray:
        return self.executable_action

    @property
    def selected_name(self) -> str:
        return "primitive_cem"


class V5ActionWorldModelPlanner:
    CODE_VERSION = "v5.1-action-primitive-cem-world-model"

    def __init__(
        self,
        model: V5ActionConditionedWorldModel,
        action_config: PortfolioActionConfig,
        cost_config: V5CostConfig,
        horizons: list[int],
        n_sampled_actions: int = 128,
        max_turnover: float | None = 0.40,
        cem_iters: int = 3,
        elite_frac: float = 0.20,
        uncertainty_penalty: float = 0.15,
        turnover_penalty: float = 0.02,
        chunk_size: int = 256,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.model = model.to(device).eval()
        self.action_config = action_config
        self.cost_config = cost_config
        self.horizons = horizons
        self.n_sampled_actions = n_sampled_actions
        self.cem_iters = cem_iters
        self.elite_frac = elite_frac
        self.uncertainty_penalty = uncertainty_penalty
        self.turnover_penalty = turnover_penalty
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.execution_layer = PrimitiveExecutionLayer(
            ExecutionConstraints(
                mode=action_config.mode,
                max_long_weight=action_config.max_long_weight,
                max_short_weight=action_config.max_short_weight,
                max_gross_exposure=action_config.max_gross_exposure,
                max_net_exposure=action_config.max_net_exposure,
                max_turnover=max_turnover,
                transaction_cost_bps=action_config.transaction_cost_bps,
                borrow_cost_bps=action_config.borrow_cost_bps,
            )
        )

    @torch.no_grad()
    def _score_candidates(
        self,
        z_market_now: torch.Tensor,
        portfolio_state: np.ndarray,
        primitive_actions: np.ndarray,
        executable_actions: np.ndarray,
        horizons: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        costs: list[np.ndarray] = []
        outcome_rows: list[dict[str, float]] = []
        portfolio_np = np.repeat(portfolio_state[None].astype(np.float32), len(primitive_actions), axis=0)
        goals_np = np.asarray([goal_vector(self.cost_config, int(h)) for h in horizons], dtype=np.float32)
        for start in range(0, len(primitive_actions), self.chunk_size):
            end = min(start + self.chunk_size, len(primitive_actions))
            primitive_t = torch.tensor(primitive_actions[start:end], dtype=torch.float32, device=self.device)
            executable_t = torch.tensor(executable_actions[start:end], dtype=torch.float32, device=self.device)
            horizon_t = torch.tensor(horizons[start:end], dtype=torch.long, device=self.device)
            goal_t = torch.tensor(goals_np[start:end], dtype=torch.float32, device=self.device)
            portfolio_t = torch.tensor(portfolio_np[start:end], dtype=torch.float32, device=self.device)
            z_now_rep = z_market_now.repeat(end - start, 1, 1)
            z_market_future = self.model.market_jepa.predict_future(z_now_rep, horizon_t)
            z_portfolio_now = self.model.portfolio_encoder(portfolio_t)
            z_portfolio = self.model.transition(
                z_now_rep,
                z_market_future,
                z_portfolio_now,
                portfolio_t,
                primitive_t,
                executable_t,
                horizon_t,
            )
            outcome_hat = self.model.outcome_head(z_portfolio)
            features = torch.cat(
                [
                    outcome_hat["return_quantiles"],
                    outcome_hat["drawdown_quantiles"],
                    outcome_hat["volatility"].unsqueeze(-1),
                    outcome_hat["turnover"].unsqueeze(-1),
                    outcome_hat["cost"].unsqueeze(-1),
                    outcome_hat["cvar"].unsqueeze(-1),
                    torch.sigmoid(outcome_hat["prob_loss_logit"]).unsqueeze(-1),
                    torch.sigmoid(outcome_hat["prob_drawdown_breach_logit"]).unsqueeze(-1),
                    outcome_hat["future_equity_ratio"].unsqueeze(-1),
                ],
                dim=-1,
            )
            base_cost = self.model.cost_model(features, goal_t)
            uncertainty = (outcome_hat["return_quantiles"][:, -1] - outcome_hat["return_quantiles"][:, 0]).abs()
            turnover = outcome_hat["turnover"]
            score = base_cost + self.uncertainty_penalty * uncertainty + self.turnover_penalty * turnover
            costs.append(score.detach().cpu().numpy())
            for i in range(end - start):
                outcome_rows.append(
                    {
                        "return_q05": float(outcome_hat["return_quantiles"][i, 0].detach().cpu()),
                        "return_q25": float(outcome_hat["return_quantiles"][i, 1].detach().cpu()),
                        "return_q50": float(outcome_hat["return_quantiles"][i, 2].detach().cpu()),
                        "return_q75": float(outcome_hat["return_quantiles"][i, 3].detach().cpu()),
                        "return_q95": float(outcome_hat["return_quantiles"][i, 4].detach().cpu()),
                        "drawdown_q05": float(outcome_hat["drawdown_quantiles"][i, 0].detach().cpu()),
                        "drawdown_q50": float(outcome_hat["drawdown_quantiles"][i, 2].detach().cpu()),
                        "volatility": float(outcome_hat["volatility"][i].detach().cpu()),
                        "turnover": float(outcome_hat["turnover"][i].detach().cpu()),
                        "transaction_cost": float(outcome_hat["cost"][i].detach().cpu()),
                        "cvar": float(outcome_hat["cvar"][i].detach().cpu()),
                        "prob_loss": float(torch.sigmoid(outcome_hat["prob_loss_logit"][i]).detach().cpu()),
                        "prob_drawdown_breach": float(
                            torch.sigmoid(outcome_hat["prob_drawdown_breach_logit"][i]).detach().cpu()
                        ),
                        "future_equity_ratio": float(outcome_hat["future_equity_ratio"][i].detach().cpu()),
                    }
                )
        return np.nan_to_num(np.concatenate(costs), nan=np.inf, posinf=np.inf, neginf=np.inf), outcome_rows

    @torch.no_grad()
    def choose_action(
        self,
        context: np.ndarray,
        context_mask: np.ndarray,
        portfolio_state: np.ndarray,
        tradable_mask: np.ndarray,
        sigma_now: np.ndarray | None = None,
        recent_returns: np.ndarray | None = None,
    ) -> V5PlannerResult:
        context_t = torch.tensor(context[None], dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(context_mask[None], dtype=torch.bool, device=self.device)
        z_market_now = self.model.market_jepa.encode_context(context_t, mask_t)
        n_assets = len(tradable_mask)
        current_weights = np.asarray(portfolio_state[: n_assets + 1], dtype=np.float32)

        score_mean = np.zeros(n_assets, dtype=np.float32)
        score_scale = 1.0
        gross_mean = 0.0
        gross_scale = 0.25
        risk_mean = 0.65
        risk_scale = 0.25
        all_costs = np.array([], dtype=np.float32)
        all_horizons = np.array([], dtype=np.int64)
        best = None
        best_outcome = {}

        for iteration in range(self.cem_iters):
            primitives = sample_primitive_actions(
                self.rng,
                n_assets=n_assets,
                horizons=self.horizons,
                n_actions=self.n_sampled_actions,
                score_mean=score_mean,
                score_scale=score_scale,
                gross_delta_mean=gross_mean,
                gross_delta_scale=gross_scale,
                risk_budget_mean=risk_mean,
                risk_budget_scale=risk_scale,
            )
            executable = np.stack(
                [
                    self.execution_layer.execute(vec, current_weights, tradable_mask.astype(bool))
                    for vec in primitives.vectors
                ]
            ).astype(np.float32)
            costs, outcome_rows = self._score_candidates(
                z_market_now,
                portfolio_state,
                primitives.vectors,
                executable,
                primitives.horizons,
            )
            all_costs = costs
            all_horizons = primitives.horizons
            elite_n = max(2, int(np.ceil(self.elite_frac * len(costs))))
            elite_idx = np.argsort(costs)[:elite_n]
            elites = primitives.vectors[elite_idx]
            score_mean = elites[:, :n_assets].mean(axis=0)
            score_scale = float(np.clip(elites[:, :n_assets].std(), 0.05, 1.5))
            gross_mean = float(np.clip(elites[:, n_assets].mean(), -1.0, 1.0))
            gross_scale = float(np.clip(elites[:, n_assets].std(), 0.03, 0.5))
            risk_mean = float(np.clip(elites[:, n_assets + 3].mean(), 0.0, 1.0))
            risk_scale = float(np.clip(elites[:, n_assets + 3].std(), 0.03, 0.5))
            selected = int(np.argmin(costs))
            best = (primitives.vectors[selected], executable[selected], int(primitives.horizons[selected]), float(costs[selected]))
            best_outcome = outcome_rows[selected]

        if best is None:
            cash = current_weights.copy()
            cash[:-1] = 0.0
            cash[-1] = 1.0
            best = (np.zeros(n_assets + 7, dtype=np.float32), cash, int(self.horizons[0]), float("inf"))
        primitive, executable, horizon, predicted_cost = best
        primitive_obj = primitive_from_vector(primitive, n_assets)
        diagnostics = {
            "cem_iters": float(self.cem_iters),
            "candidate_count": float(self.n_sampled_actions),
            "best_cost": float(predicted_cost),
            "cost_mean": float(np.mean(all_costs)) if len(all_costs) else float("nan"),
            "cost_std": float(np.std(all_costs)) if len(all_costs) else float("nan"),
            "gross_exposure": float(np.abs(executable[:-1]).sum()),
            "cash_weight": float(executable[-1]),
            "rebalance_intensity": float(primitive_obj.rebalance_intensity),
            "risk_budget": float(primitive_obj.risk_budget),
            "horizon_entropy": float(len(np.unique(all_horizons)) / max(len(self.horizons), 1)) if len(all_horizons) else 0.0,
        }
        return V5PlannerResult(
            primitive_action=primitive.astype(np.float32),
            executable_action=executable.astype(np.float32),
            horizon=horizon,
            predicted_cost=predicted_cost,
            predicted_outcome=best_outcome,
            candidate_costs=all_costs,
            candidate_horizons=all_horizons,
            diagnostics=diagnostics,
        )


def portfolio_state_from_weights(weights: np.ndarray, equity: float, peak_equity: float, last_turnover: float) -> np.ndarray:
    return portfolio_state_features(weights, equity=equity, peak_equity=peak_equity, last_turnover=last_turnover)
