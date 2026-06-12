from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jepa_trading.data.v5_dataset import V5_OUTCOME_KEYS, V5_QUANTILES
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


def pinball_loss(pred_quantiles: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...] = V5_QUANTILES) -> torch.Tensor:
    q = torch.tensor(quantiles, dtype=pred_quantiles.dtype, device=pred_quantiles.device)
    error = target.unsqueeze(-1) - pred_quantiles
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def pairwise_ranking_loss(cost_hat: torch.Tensor, realized_costs: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    best_idx = torch.argmin(realized_costs, dim=1)
    best_hat = cost_hat.gather(1, best_idx[:, None]).squeeze(1)
    raw = F.relu(margin + best_hat[:, None] - cost_hat)
    mask = torch.ones_like(raw, dtype=torch.bool)
    mask.scatter_(1, best_idx[:, None], False)
    return raw[mask].mean()


def v5_world_model_loss(
    outputs: dict,
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

    outcomes = batch["realized_outcomes"]
    out_hat = outputs["outcome_hat"]
    idx = {name: i for i, name in enumerate(V5_OUTCOME_KEYS)}
    return_q = pinball_loss(out_hat["return_quantiles"], outcomes[..., idx["realized_return"]])
    drawdown_q = pinball_loss(out_hat["drawdown_quantiles"], outcomes[..., idx["drawdown"]])
    volatility = F.smooth_l1_loss(out_hat["volatility"], outcomes[..., idx["volatility"]])
    turnover = F.smooth_l1_loss(out_hat["turnover"], outcomes[..., idx["turnover"]])
    transaction_cost = F.smooth_l1_loss(out_hat["cost"], outcomes[..., idx["transaction_cost"]])
    cvar = F.smooth_l1_loss(out_hat["cvar"], outcomes[..., idx["cvar"]])
    equity = F.smooth_l1_loss(out_hat["future_equity_ratio"], outcomes[..., idx["future_equity_ratio"]])
    prob_loss = F.binary_cross_entropy_with_logits(out_hat["prob_loss_logit"], outcomes[..., idx["prob_loss"]])
    prob_drawdown = F.binary_cross_entropy_with_logits(
        out_hat["prob_drawdown_breach_logit"], outcomes[..., idx["prob_drawdown_breach"]]
    )
    outcome = return_q + drawdown_q + volatility + turnover + transaction_cost + cvar + equity + prob_loss + prob_drawdown
    cost = F.smooth_l1_loss(outputs["cost_hat"], batch["realized_costs"])
    rank = pairwise_ranking_loss(outputs["cost_hat"], batch["realized_costs"], margin=rank_margin)

    phase = min(1.0, max(0.0, step / max(warmup_steps, 1)))
    loss = (
        weights.get("market_jepa", 1.0) * market_jepa
        + weights.get("portfolio_jepa", 1.0) * portfolio_jepa
        + weights.get("vicreg", 0.05) * vicreg
        + phase * weights.get("return_quantile", weights.get("outcome", 1.0)) * return_q
        + phase * weights.get("drawdown_quantile", weights.get("outcome", 1.0)) * drawdown_q
        + phase * weights.get("outcome_aux", weights.get("outcome", 1.0)) * (volatility + turnover + transaction_cost + cvar + equity)
        + phase * weights.get("probability", 0.5) * (prob_loss + prob_drawdown)
        + phase * weights.get("cost", 0.5) * cost
        + phase * weights.get("rank", 0.7) * rank
    )
    return {
        "loss": loss,
        "market_jepa_loss": market_jepa,
        "portfolio_jepa_loss": portfolio_jepa,
        "vicreg_loss": vicreg,
        "return_quantile_loss": return_q,
        "drawdown_quantile_loss": drawdown_q,
        "volatility_loss": volatility,
        "turnover_loss": turnover,
        "transaction_cost_loss": transaction_cost,
        "cvar_loss": cvar,
        "future_equity_loss": equity,
        "prob_loss_bce": prob_loss,
        "drawdown_breach_bce": prob_drawdown,
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
                    kind="v5_action_primitive_world_model",
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
