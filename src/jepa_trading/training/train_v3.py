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


def v3_forward_candidates(model: V2WorldModel, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    z_hat, z_target, z_now = model.market_jepa(
        batch["context"],
        batch["target"],
        batch["horizon"],
        batch["context_mask"],
        batch["target_mask"],
    )

    batch_size, n_candidates, action_dim = batch["actions"].shape
    n_assets, latent_dim = z_now.shape[1], z_now.shape[2]
    z_now_flat = z_now[:, None].expand(batch_size, n_candidates, n_assets, latent_dim).reshape(
        batch_size * n_candidates, n_assets, latent_dim
    )
    z_hat_flat = z_hat[:, None].expand(batch_size, n_candidates, n_assets, latent_dim).reshape(
        batch_size * n_candidates, n_assets, latent_dim
    )
    portfolio_flat = batch["portfolio_state"][:, None].expand(batch_size, n_candidates, -1).reshape(
        batch_size * n_candidates, -1
    )
    action_flat = batch["actions"].reshape(batch_size * n_candidates, action_dim)
    horizon_flat = batch["horizon"][:, None].expand(batch_size, n_candidates).reshape(batch_size * n_candidates)

    outcome_hat = model.outcome_model(z_now_flat, z_hat_flat, portfolio_flat, action_flat, horizon_flat).reshape(
        batch_size, n_candidates, -1
    )
    energy_hat = model.energy_model(
        outcome_hat.reshape(batch_size * n_candidates, -1),
        portfolio_flat,
        action_flat,
    ).reshape(batch_size, n_candidates)
    obs = torch.cat(
        [
            z_now.reshape(batch_size, -1),
            batch["tradable_mask"].float(),
            batch["portfolio_state"],
        ],
        dim=-1,
    )
    policy_action = model.policy(obs, batch["tradable_mask"])
    return {
        "z_hat": z_hat,
        "z_target": z_target,
        "z_now": z_now,
        "outcome_hat": outcome_hat,
        "energy_hat": energy_hat,
        "policy_action": policy_action,
    }


def v3_world_model_loss(
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
    advantage_signal = torch.maximum(batch["advantage_vs_hold"], batch["advantage_vs_cash"])
    abstention_rank = F.relu(rank_margin - best_energy + torch.maximum(hold_energy, cash_energy))
    abstention_rank = (abstention_rank * (advantage_signal > 0).float()).mean()
    rank = rank + 0.5 * abstention_rank

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
        "phase": torch.tensor(phase, device=loss.device),
    }


@torch.no_grad()
def evaluate_v3(
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
        losses = v3_world_model_loss(outputs, batch, warmup_steps, warmup_steps, weights, rank_margin=rank_margin)
        rows.append({k: float(v.item()) for k, v in losses.items()})
    if not rows:
        return {"loss": float("inf")}
    return {k: float(sum(row[k] for row in rows) / len(rows)) for k in rows[0]}


def train_v3_world_model(
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
) -> pd.DataFrame:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    batches = iter(train_loader)
    best_val = float("inf")
    history: list[dict[str, float]] = []

    pbar = tqdm(range(1, max_steps + 1), desc="train V3 abstention world model", dynamic_ncols=True)
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
        losses = v3_world_model_loss(outputs, batch, step, warmup_steps, weights, rank_margin=rank_margin)
        if not torch.isfinite(losses["loss"]):
            diagnostics = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
            utility_min = float(batch["utilities"].detach().cpu().min().item())
            utility_max = float(batch["utilities"].detach().cpu().max().item())
            raise FloatingPointError(
                "Non-finite V3 loss detected. "
                f"losses={diagnostics}, utility_range=({utility_min}, {utility_max})"
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target_encoder()

        row = {"step": step, **{k: float(v.item()) for k, v in losses.items()}}
        if step % eval_every == 0 or step == 1:
            val = evaluate_v3(model, val_loader, device, warmup_steps, weights, rank_margin=rank_margin)
            row.update({f"val_{k}": v for k, v in val.items()})
            if val["loss"] < best_val:
                best_val = val["loss"]
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step=step,
                    val_loss=best_val,
                    kind="v3_abstention_world_model",
                )
        history.append(row)
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{row['loss']:.4f}", best_val=f"{best_val:.4f}")
    return pd.DataFrame(history)
