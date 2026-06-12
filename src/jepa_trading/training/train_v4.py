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
from jepa_trading.training.train_v3 import v3_forward_candidates
from jepa_trading.training.vicreg import vicreg_regularizer


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def v4_world_model_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    step: int,
    warmup_steps: int,
    weights: dict[str, float],
    rank_margin: float = 0.05,
) -> dict[str, torch.Tensor]:
    jepa = jepa_latent_loss(outputs["z_hat"], outputs["z_target"], batch["tradable_mask"])
    var_now, cov_now = vicreg_regularizer(outputs["z_now"])
    var_tgt, cov_tgt = vicreg_regularizer(outputs["z_target"].detach())
    var_hat, cov_hat = vicreg_regularizer(outputs["z_hat"])
    vicreg = var_now + cov_now + var_tgt + cov_tgt + 0.5 * (var_hat + cov_hat)

    outcome = F.smooth_l1_loss(outputs["outcome_hat"], batch["outcomes"])
    energy = F.smooth_l1_loss(outputs["energy_hat"], batch["utilities"])
    policy_distill = F.smooth_l1_loss(outputs["policy_action"], batch["best_action"])

    best_idx = batch["best_action_index"]
    best_energy = outputs["energy_hat"].gather(1, best_idx[:, None]).squeeze(1)
    rank_raw = F.relu(rank_margin - best_energy[:, None] + outputs["energy_hat"])
    rank_mask = torch.ones_like(rank_raw, dtype=torch.bool)
    rank_mask.scatter_(1, best_idx[:, None], False)
    rank = rank_raw[rank_mask].mean()

    hold_energy = outputs["energy_hat"].gather(1, batch["hold_action_index"][:, None]).squeeze(1)
    cash_energy = outputs["energy_hat"].gather(1, batch["cash_action_index"][:, None]).squeeze(1)
    derisk_energy = outputs["energy_hat"].gather(1, batch["derisk_action_index"][:, None]).squeeze(1)
    defensive_energy = torch.maximum(cash_energy, derisk_energy)

    risk_mask = batch["risk_off_label"].float()
    if risk_mask.sum() > 0:
        risk_off_rank = (F.relu(rank_margin - defensive_energy + hold_energy) * risk_mask).sum() / risk_mask.sum()
    else:
        risk_off_rank = torch.zeros((), device=outputs["energy_hat"].device)
    risk_on_mask = 1.0 - risk_mask
    if risk_on_mask.sum() > 0:
        risk_on_rank = (F.relu(rank_margin - best_energy + defensive_energy) * risk_on_mask).sum() / risk_on_mask.sum()
    else:
        risk_on_rank = torch.zeros((), device=outputs["energy_hat"].device)
    rank = rank + 0.75 * risk_off_rank + 0.25 * risk_on_rank

    phase = min(1.0, max(0.0, step / max(warmup_steps, 1)))
    loss = (
        weights["jepa"] * jepa
        + weights["vicreg"] * vicreg
        + phase * weights["outcome"] * outcome
        + phase * weights["energy"] * energy
        + phase * weights["policy"] * policy_distill
        + phase * weights["rank"] * rank
    )
    return {
        "loss": loss,
        "jepa_loss": jepa,
        "vicreg_loss": vicreg,
        "outcome_loss": outcome,
        "energy_loss": energy,
        "policy_distill_loss": policy_distill,
        "rank_loss": rank,
        "risk_off_rank_loss": risk_off_rank,
        "risk_on_rank_loss": risk_on_rank,
        "phase": torch.tensor(phase, device=loss.device),
    }


@torch.no_grad()
def evaluate_v4(
    model: V2WorldModel,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    weights: dict[str, float],
    rank_margin: float = 0.05,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        outputs = v3_forward_candidates(model, batch)
        losses = v4_world_model_loss(outputs, batch, warmup_steps, warmup_steps, weights, rank_margin=rank_margin)
        rows.append({k: float(v.item()) for k, v in losses.items()})
    if not rows:
        return {"loss": float("inf")}
    return {k: float(sum(row[k] for row in rows) / len(rows)) for k in rows[0]}


def train_v4_world_model(
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

    pbar = tqdm(range(1, max_steps + 1), desc="train V4 risk-off world model", dynamic_ncols=True)
    for step in pbar:
        model.train()
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            batch = next(batches)
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = v3_forward_candidates(model, batch)
        losses = v4_world_model_loss(outputs, batch, step, warmup_steps, weights, rank_margin=rank_margin)
        if not torch.isfinite(losses["loss"]):
            diagnostics = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
            raise FloatingPointError(f"Non-finite V4 loss detected. losses={diagnostics}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target_encoder()

        row = {"step": step, **{k: float(v.item()) for k, v in losses.items()}}
        if step % eval_every == 0 or step == 1:
            val = evaluate_v4(model, val_loader, device, warmup_steps, weights, rank_margin=rank_margin)
            row.update({f"val_{k}": v for k, v in val.items()})
            if val["loss"] < best_val:
                best_val = val["loss"]
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step=step,
                    val_loss=best_val,
                    kind="v4_risk_off_world_model",
                )
        history.append(row)
        if history_path is not None and (
            step == 1
            or step % history_save_every == 0
            or step % eval_every == 0
            or step == max_steps
        ):
            pd.DataFrame(history).to_csv(history_path, index=False)
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{row['loss']:.4f}", best_val=f"{best_val:.4f}")
    history_df = pd.DataFrame(history)
    if history_path is not None:
        history_df.to_csv(history_path, index=False)
    return history_df
