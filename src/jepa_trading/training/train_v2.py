from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jepa_trading.models.jepa import jepa_latent_loss
from jepa_trading.models.world_model_v2 import V2WorldModel
from jepa_trading.training.checkpoints import save_checkpoint
from jepa_trading.training.vicreg import vicreg_regularizer


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def v2_world_model_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    step: int,
    warmup_steps: int,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    jepa = jepa_latent_loss(outputs["z_hat"], outputs["z_target"], batch["tradable_mask"])
    var_now, cov_now = vicreg_regularizer(outputs["z_now"])
    var_tgt, cov_tgt = vicreg_regularizer(outputs["z_target"].detach())
    var_hat, cov_hat = vicreg_regularizer(outputs["z_hat"])
    vicreg = var_now + cov_now + var_tgt + cov_tgt + 0.5 * (var_hat + cov_hat)

    outcome = F.smooth_l1_loss(outputs["outcome_hat"], batch["outcome"])
    energy = F.smooth_l1_loss(outputs["energy_hat"], batch["utility"])
    policy_distill = F.smooth_l1_loss(outputs["policy_action"], batch["action"])

    phase = min(1.0, max(0.0, step / max(warmup_steps, 1)))
    loss = (
        weights["jepa"] * jepa
        + weights["vicreg"] * vicreg
        + phase * weights["outcome"] * outcome
        + phase * weights["energy"] * energy
        + phase * weights["policy"] * policy_distill
    )
    return {
        "loss": loss,
        "jepa_loss": jepa,
        "vicreg_loss": vicreg,
        "outcome_loss": outcome,
        "energy_loss": energy,
        "policy_distill_loss": policy_distill,
        "phase": torch.tensor(phase, device=loss.device),
    }


@torch.no_grad()
def evaluate_v2(
    model: V2WorldModel,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    weights: dict[str, float],
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        outputs = model(batch)
        losses = v2_world_model_loss(outputs, batch, warmup_steps, warmup_steps, weights)
        rows.append({k: float(v.item()) for k, v in losses.items()})
    if not rows:
        return {"loss": float("inf")}
    return {k: float(sum(row[k] for row in rows) / len(rows)) for k in rows[0]}


def train_v2_world_model(
    model: V2WorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_steps: int,
    warmup_steps: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: str | Path,
    weights: dict[str, float],
    eval_every: int = 200,
    log_every: int = 25,
) -> pd.DataFrame:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Do not use itertools.cycle here: it caches every yielded batch and can
    # silently fill Kaggle RAM during long max_steps training.
    batches = iter(train_loader)
    best_val = float("inf")
    history: list[dict[str, float]] = []

    pbar = tqdm(range(1, max_steps + 1), desc="train V2 world model", dynamic_ncols=True)
    for step in pbar:
        model.train()
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            batch = next(batches)
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = v2_world_model_loss(outputs, batch, step, warmup_steps, weights)
        if not torch.isfinite(losses["loss"]):
            diagnostics = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
            outcome_min = float(batch["outcome"].detach().cpu().min().item())
            outcome_max = float(batch["outcome"].detach().cpu().max().item())
            utility_min = float(batch["utility"].detach().cpu().min().item())
            utility_max = float(batch["utility"].detach().cpu().max().item())
            raise FloatingPointError(
                "Non-finite V2 loss detected. "
                f"losses={diagnostics}, outcome_range=({outcome_min}, {outcome_max}), "
                f"utility_range=({utility_min}, {utility_max})"
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target_encoder()

        row = {"step": step, **{k: float(v.item()) for k, v in losses.items()}}
        if step % eval_every == 0 or step == 1:
            val = evaluate_v2(model, val_loader, device, warmup_steps, weights)
            row.update({f"val_{k}": v for k, v in val.items()})
            if val["loss"] < best_val:
                best_val = val["loss"]
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step=step,
                    val_loss=best_val,
                    kind="v2_world_model",
                )
        history.append(row)
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{row['loss']:.4f}", best_val=f"{best_val:.4f}")
    return pd.DataFrame(history)

