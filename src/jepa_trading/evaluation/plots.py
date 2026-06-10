from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(
    agent: pd.DataFrame,
    buy_hold: pd.DataFrame,
    random_histories: list[pd.DataFrame],
    extra: dict[str, pd.DataFrame] | None = None,
) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 7))
    for hist in random_histories:
        ax.plot(hist["date"], hist["equity"], color="gray", alpha=0.16, linewidth=0.8)
    if extra:
        for name, hist in extra.items():
            ax.plot(hist["date"], hist["equity"], linewidth=1.5, label=name)
    ax.plot(buy_hold["date"], buy_hold["equity"], color="white", linewidth=2.0, label="Buy & Hold")
    ax.plot(agent["date"], agent["equity"], color="#39ff14", linewidth=2.5, label="JEPA-PPO Agent")
    ax.set_title("Trading Strategy Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.18)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_drawdown(history: pd.DataFrame, label: str = "JEPA-PPO Agent") -> None:
    equity = history["equity"]
    dd = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(history["date"], dd, 0, color="#39ff14", alpha=0.35)
    ax.plot(history["date"], dd, color="#39ff14", label=label)
    ax.set_title("Drawdown")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_turnover(history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(history["date"], history["turnover"], color="#39ff14")
    ax.set_title("Portfolio Turnover")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

