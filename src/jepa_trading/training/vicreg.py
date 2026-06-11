from __future__ import annotations

import torch
import torch.nn.functional as F


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n == 0 or m == 0:
        return x.new_zeros(0)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_regularizer(
    z: torch.Tensor,
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    # z may be batch x assets x dim. Flatten batch/assets so the covariance is
    # estimated across all available asset examples.
    z = z.reshape(-1, z.shape[-1])
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(variance_target - std).mean()
    if z.shape[0] <= 1:
        cov_loss = z.new_tensor(0.0)
    else:
        cov = (z.T @ z) / (z.shape[0] - 1)
        cov_loss = off_diagonal(cov).pow(2).sum() / z.shape[-1]
    return var_loss, cov_loss

