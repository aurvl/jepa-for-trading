from __future__ import annotations

from itertools import cycle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jepa_trading.models.heads import MarketHeads
from jepa_trading.models.jepa import MarketJEPA
from jepa_trading.training.checkpoints import save_checkpoint
from jepa_trading.training.losses import market_head_loss


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_heads(
    jepa: MarketJEPA,
    heads: MarketHeads,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    jepa.eval()
    heads.eval()
    vals: list[float] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        z = jepa.encode_context(batch["context"], batch["context_mask"])
        outputs = heads(z)
        vals.append(float(market_head_loss(outputs, batch)["loss"].item()))
    return float(sum(vals) / max(len(vals), 1))


def train_market_heads(
    jepa: MarketJEPA,
    heads: MarketHeads,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_steps: int,
    lr: float,
    checkpoint_path: str | Path,
    eval_every: int = 200,
) -> pd.DataFrame:
    jepa.to(device).eval()
    heads.to(device)
    for p in jepa.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=lr)
    batches = cycle(train_loader)
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for step in tqdm(range(1, max_steps + 1), desc="train heads", dynamic_ncols=True):
        heads.train()
        batch = _to_device(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = jepa.encode_context(batch["context"], batch["context_mask"])
        outputs = heads(z)
        losses = market_head_loss(outputs, batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        optimizer.step()

        row = {"step": step, **{k: float(v.item()) for k, v in losses.items()}}
        if step % eval_every == 0 or step == 1:
            val_loss = evaluate_heads(jepa, heads, val_loader, device)
            row["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(checkpoint_path, heads, optimizer, step=step, val_loss=val_loss, kind="market_heads")
        history.append(row)
    return pd.DataFrame(history)

