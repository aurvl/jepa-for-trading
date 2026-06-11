from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from jepa_trading.data.dataset import MarketArrays


@dataclass
class PortfolioState:
    equity: float
    peak_equity: float
    weights: np.ndarray
    last_turnover: float = 0.0


class TradingEnv:
    def __init__(
        self,
        arrays: MarketArrays,
        observer,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        lookback: int,
        cash_initial: float = 1000.0,
        transaction_cost_bps: float = 5.0,
        max_weight_per_asset: float = 0.15,
        max_turnover: float = 0.40,
        mode: str = "long_only",
        max_long_weight: float | None = None,
        max_short_weight: float = 0.05,
        max_gross_exposure: float = 1.0,
        max_net_exposure: float = 1.0,
        borrow_cost_bps: float = 2.0,
        drawdown_penalty: float = 0.20,
        turnover_penalty: float = 0.05,
        concentration_penalty: float = 0.02,
    ) -> None:
        self.arrays = arrays
        self.observer = observer
        self.lookback = lookback
        self.cash_initial = cash_initial
        self.cost_rate = transaction_cost_bps / 10000.0
        self.max_weight = max_weight_per_asset
        self.mode = mode
        self.max_long_weight = max_long_weight if max_long_weight is not None else max_weight_per_asset
        self.max_short_weight = max_short_weight
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.borrow_cost_rate = borrow_cost_bps / 10000.0
        self.max_turnover = max_turnover
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.concentration_penalty = concentration_penalty
        dates = arrays.dates
        self.start_idx = max(int(np.searchsorted(dates, pd.Timestamp(start_date))), lookback)
        self.end_idx = min(int(np.searchsorted(dates, pd.Timestamp(end_date), side="right")) - 2, len(dates) - 2)
        self.n_assets = len(arrays.tickers)
        self.state: PortfolioState
        self.idx = self.start_idx
        self.history: list[dict[str, float]] = []

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.idx = self.start_idx
        self.state = PortfolioState(
            equity=self.cash_initial,
            peak_equity=self.cash_initial,
            weights=np.zeros(self.n_assets + 1, dtype=np.float32),
        )
        self.state.weights[-1] = 1.0
        self.history = []
        return self._obs(), self.tradable_mask()

    def tradable_mask(self) -> np.ndarray:
        return self.arrays.tradable[self.idx].astype(bool)

    def _portfolio_features(self) -> np.ndarray:
        drawdown = self.state.equity / max(self.state.peak_equity, 1e-8) - 1.0
        extras = np.array(
            [
                self.state.equity / self.cash_initial - 1.0,
                drawdown,
                self.state.last_turnover,
                self.state.weights[-1],
            ],
            dtype=np.float32,
        )
        return np.concatenate([self.state.weights.astype(np.float32), extras])

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.observer.market_state(self.idx), self._portfolio_features()]).astype(np.float32)

    def _sanitize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[0] != self.n_assets + 1:
            raise ValueError(f"Expected action size {self.n_assets + 1}, got {action.shape[0]}")
        tradable = self.tradable_mask()
        if self.mode == "long_only":
            action[:-1] = np.where(tradable, np.clip(action[:-1], 0.0, self.max_long_weight), 0.0)
            action[-1] = max(float(action[-1]), 0.0)
            total = float(action.sum())
            if total <= 1e-8:
                action[-1] = 1.0
                total = 1.0
            action /= total
        elif self.mode in {"long_short", "market_neutral"}:
            action[:-1] = np.where(
                tradable,
                np.clip(action[:-1], -self.max_short_weight, self.max_long_weight),
                0.0,
            )
            if self.mode == "market_neutral":
                valid = tradable & (np.abs(action[:-1]) > 0)
                if valid.any():
                    asset_action = action[:-1]
                    asset_action[valid] -= asset_action[valid].mean()
                    action[:-1] = asset_action
            gross = float(np.abs(action[:-1]).sum())
            if gross > self.max_gross_exposure:
                action[:-1] *= self.max_gross_exposure / gross
            net = float(action[:-1].sum())
            if abs(net) > self.max_net_exposure:
                action[:-1] *= self.max_net_exposure / abs(net)
            action[-1] = max(1.0 - float(np.abs(action[:-1]).sum()), 0.0)
        else:
            raise ValueError(f"Unknown portfolio mode: {self.mode}")
        turnover = float(np.abs(action - self.state.weights).sum())
        if turnover > self.max_turnover:
            alpha = self.max_turnover / turnover
            action = self.state.weights + alpha * (action - self.state.weights)
            action = np.clip(action, 0.0, None)
            action /= action.sum().clip(min=1e-8)
        return action.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float], np.ndarray]:
        prev_equity = self.state.equity
        target_weights = self._sanitize_action(action)
        turnover = float(np.abs(target_weights - self.state.weights).sum())
        cost = prev_equity * turnover * self.cost_rate
        borrow_cost = prev_equity * float(np.abs(np.minimum(target_weights[:-1], 0.0)).sum()) * self.borrow_cost_rate
        investable = max(prev_equity - cost - borrow_cost, 1e-8)
        next_returns = np.nan_to_num(self.arrays.log_returns[self.idx + 1], nan=0.0)
        gross = target_weights[-1] + float(np.sum(target_weights[:-1] * np.exp(next_returns)))
        next_equity = investable * gross
        self.state.peak_equity = max(self.state.peak_equity, next_equity)
        drawdown = next_equity / max(self.state.peak_equity, 1e-8) - 1.0
        concentration = float(np.sum(target_weights[:-1] ** 2))
        log_ret = float(np.log(next_equity / max(prev_equity, 1e-8)))
        reward = (
            log_ret
            - self.drawdown_penalty * abs(min(drawdown, 0.0))
            - self.turnover_penalty * turnover
            - self.concentration_penalty * concentration
        )
        self.state.equity = float(next_equity)
        self.state.weights = target_weights
        self.state.last_turnover = turnover
        date = self.arrays.dates[self.idx + 1]
        self.history.append(
            {
                "date": date,
                "equity": self.state.equity,
                "reward": reward,
                "log_return": log_ret,
                "turnover": turnover,
                "cost": cost + borrow_cost,
                "drawdown": drawdown,
                "cash_weight": float(target_weights[-1]),
            }
        )
        self.idx += 1
        done = self.idx >= self.end_idx
        return self._obs(), float(reward), done, self.history[-1], self.tradable_mask()
