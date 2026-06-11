from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jepa_trading.models.jepa import MarketJEPA, jepa_latent_loss
from jepa_trading.training.checkpoints import save_checkpoint


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_jepa(model: MarketJEPA, loader: DataLoader, device: torch.device, max_batches: int = 20) -> float:
    model.eval()
    vals: list[float] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        z_hat, z_target, _ = model(
            batch["context"],
            batch["target"],
            batch["horizon"],
            batch["context_mask"],
            batch["target_mask"],
        )
        loss = jepa_latent_loss(z_hat, z_target, batch["tradable_mask"])
        vals.append(float(loss.item()))
    return float(sum(vals) / max(len(vals), 1))


def train_jepa(
    model: MarketJEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_steps: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: str | Path,
    eval_every: int = 200,
    log_every: int = 25,
) -> pd.DataFrame:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    pbar = tqdm(range(1, max_steps + 1), desc="train JEPA", dynamic_ncols=True)
    batches = iter(train_loader)

    for step in pbar:
        model.train()
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            batch = next(batches)
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        z_hat, z_target, _ = model(
            batch["context"],
            batch["target"],
            batch["horizon"],
            batch["context_mask"],
            batch["target_mask"],
        )
        loss = jepa_latent_loss(z_hat, z_target, batch["tradable_mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target_encoder()

        row = {"step": step, "train_loss": float(loss.item())}
        if step % eval_every == 0 or step == 1:
            val_loss = evaluate_jepa(model, val_loader, device)
            row["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step=step,
                    val_loss=val_loss,
                    kind="market_jepa",
                )
        history.append(row)
        if step % log_every == 0:
            pbar.set_postfix(train_loss=f"{loss.item():.4f}", best_val=f"{best_val:.4f}")

    return pd.DataFrame(history)
