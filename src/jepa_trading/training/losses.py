from __future__ import annotations

import torch
import torch.nn.functional as F

from jepa_trading.models.heads import pinball_loss


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred - target).pow(2)
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def market_head_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mask = batch["tradable_mask"].float()
    ret_loss = pinball_loss(outputs["return_quantiles"], batch["future_return"], mask=mask)
    sigma_loss = masked_mse(outputs["sigma"], batch["future_sigma"], mask)
    dd_loss = masked_mse(outputs["drawdown"], batch["future_drawdown"], mask)
    positive = (batch["future_return"] > 0).float()
    pos_loss = F.binary_cross_entropy_with_logits(outputs["positive_logit"], positive, reduction="none")
    pos_loss = (pos_loss * mask).sum() / mask.sum().clamp_min(1.0)
    total = ret_loss + 0.5 * sigma_loss + 0.5 * dd_loss + 0.25 * pos_loss
    return {
        "loss": total,
        "return_quantile_loss": ret_loss,
        "sigma_loss": sigma_loss,
        "drawdown_loss": dd_loss,
        "positive_loss": pos_loss,
    }
