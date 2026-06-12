from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jepa_trading.models.jepa import jepa_latent_loss
from jepa_trading.models.world_model_v5 import V5ActionConditionedWorldModel
from jepa_trading.training.checkpoints import save_checkpoint
from jepa_trading.training.vicreg import vicreg_regularizer


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _normalized_latent_loss(z_hat: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    z_hat = F.normalize(z_hat, dim=-1)
    z_target = F.normalize(z_target.detach(), dim=-1)
    return (2 - 2 * (z_hat * z_target).sum(dim=-1)).mean()


def v5_world_model_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    step: int,
    warmup_steps: int,
    weights: dict[str, float],
    rank_margin: float = 0.05,
) -> dict[str, torch.Tensor]:
    market_jepa = jepa_latent_loss(outputs["z_market_hat"], outputs["z_market_target"], batch["tradable_mask"])
    portfolio_jepa = _normalized_latent_loss(outputs["z_portfolio_hat"], outputs["z_portfolio_target"])
    var_now, cov_now = vicreg_regularizer(outputs["z_market_now"])
    var_hat, cov_hat = vicreg_regularizer(outputs["z_market_hat"])
    vicreg = var_now + cov_now + 0.5 * (var_hat + cov_hat)

    outcome = F.smooth_l1_loss(outputs["outcome_hat"], batch["outcomes"])
    cost = F.smooth_l1_loss(outputs["cost_hat"], batch["goal_costs"])

    best_idx = batch["best_action_index"]
    best_cost_hat = outputs["cost_hat"].gather(1, best_idx[:, None]).squeeze(1)
    rank_raw = F.relu(rank_margin + best_cost_hat[:, None] - outputs["cost_hat"])
    rank_mask = torch.ones_like(rank_raw, dtype=torch.bool)
    rank_mask.scatter_(1, best_idx[:, None], False)
    rank = rank_raw[rank_mask].mean()

    phase = min(1.0, max(0.0, step / max(warmup_steps, 1)))
    loss = (
        weights["market_jepa"] * market_jepa
        + weights["portfolio_jepa"] * portfolio_jepa
        + weights["vicreg"] * vicreg
        + phase * weights["outcome"] * outcome
        + phase * weights["cost"] * cost
        + phase * weights["rank"] * rank
    )
    return {
        "loss": loss,
        "market_jepa_loss": market_jepa,
        "portfolio_jepa_loss": portfolio_jepa,
        "vicreg_loss": vicreg,
        "outcome_loss": outcome,
        "cost_loss": cost,
        "rank_loss": rank,
        "phase": torch.tensor(phase, device=loss.device),
    }


@torch.no_grad()
def evaluate_v5(
    model: V5ActionConditionedWorldModel,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    weights: dict[str, float],
    rank_margin: float = 0.05,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    rows = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        outputs = model.forward_candidates(batch)
        losses = v5_world_model_loss(outputs, batch, warmup_steps, warmup_steps, weights, rank_margin=rank_margin)
        rows.append({k: float(v.item()) for k, v in losses.items()})
    if not rows:
        return {"loss": float("inf")}
    return {k: float(sum(row[k] for row in rows) / len(rows)) for k in rows[0]}


def train_v5_world_model(
    model: V5ActionConditionedWorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_steps: int,
    warmup_steps: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: str | Path,
    weights: dict[str, float],
    rank_margin: float = 0.05,
    eval_every: int = 200,
    log_every: int = 25,
    history_path: str | Path | None = None,
    history_save_every: int | None = None,
) -> pd.DataFrame:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    batches = iter(train_loader)
    best_val = float("inf")
    history: list[dict[str, float]] = []
    history_save_every = history_save_every or log_every
    history_path = Path(history_path) if history_path is not None else None
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(1, max_steps + 1), desc="train V5 action world model", dynamic_ncols=True)
    for step in pbar:
        model.train()
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            batch = next(batches)
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_candidates(batch)
        losses = v5_world_model_loss(outputs, batch, step, warmup_steps, weights, rank_margin=rank_margin)
        if not torch.isfinite(losses["loss"]):
            diagnostics = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
            raise FloatingPointError(f"Non-finite V5 loss detected: {diagnostics}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target_encoders()

        row = {"step": step, **{k: float(v.item()) for k, v in losses.items()}}
        if step % eval_every == 0 or step == 1:
            val = evaluate_v5(model, val_loader, device, warmup_steps, weights, rank_margin=rank_margin)
            row.update({f"val_{k}": v for k, v in val.items()})
            if val["loss"] < best_val:
                best_val = val["loss"]
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step=step,
                    val_loss=best_val,
                    kind="v5_action_conditioned_world_model",
                )
        history.append(row)
        if history_path is not None and (
            step == 1 or step % history_save_every == 0 or step % eval_every == 0 or step == max_steps
        ):
            pd.DataFrame(history).to_csv(history_path, index=False)
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{row['loss']:.4f}", best_val=f"{best_val:.4f}")
    history_df = pd.DataFrame(history)
    if history_path is not None:
        history_df.to_csv(history_path, index=False)
    return history_df
