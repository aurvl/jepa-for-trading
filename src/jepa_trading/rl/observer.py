from __future__ import annotations

import numpy as np
import torch

from jepa_trading.data.dataset import MarketArrays
from jepa_trading.models.heads import MarketHeads
from jepa_trading.models.jepa import MarketJEPA


class JEPAMarketObserver:
    def __init__(
        self,
        arrays: MarketArrays,
        jepa: MarketJEPA,
        heads: MarketHeads,
        lookback: int,
        horizons: list[int],
        device: torch.device,
    ) -> None:
        self.arrays = arrays
        self.jepa = jepa.to(device).eval()
        self.heads = heads.to(device).eval()
        self.lookback = lookback
        self.horizons = horizons
        self.device = device

    @torch.no_grad()
    def market_state(self, end_idx: int) -> np.ndarray:
        ctx = slice(end_idx - self.lookback + 1, end_idx + 1)
        x = np.nan_to_num(self.arrays.features[ctx], nan=0.0).transpose(1, 0, 2)
        mask = self.arrays.tradable[ctx].T
        x_t = torch.tensor(x[None], dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask[None], dtype=torch.bool, device=self.device)
        z = self.jepa.encode_context(x_t, mask_t)

        chunks = [z]
        for horizon in self.horizons:
            h = torch.full((1,), horizon, dtype=torch.long, device=self.device)
            z_h = self.jepa.predict_future(z, h)
            out = self.heads(z_h)
            chunks.extend(
                [
                    out["return_quantiles"],
                    out["sigma"].unsqueeze(-1),
                    out["drawdown"].unsqueeze(-1),
                    torch.sigmoid(out["positive_logit"]).unsqueeze(-1),
                ]
            )
        state = torch.cat(chunks, dim=-1).squeeze(0).detach().cpu().numpy()
        tradable = self.arrays.tradable[end_idx].astype(np.float32)[:, None]
        return np.concatenate([state, tradable], axis=-1).reshape(-1).astype(np.float32)


class RawMarketObserver:
    def __init__(self, arrays: MarketArrays, lookback: int) -> None:
        self.arrays = arrays
        self.lookback = lookback

    def market_state(self, end_idx: int) -> np.ndarray:
        ctx = slice(end_idx - self.lookback + 1, end_idx + 1)
        x = self.arrays.features[ctx]
        means = np.nanmean(x, axis=0)
        last = x[-1]
        state = np.concatenate([np.nan_to_num(last), np.nan_to_num(means)], axis=-1)
        tradable = self.arrays.tradable[end_idx].astype(np.float32)[:, None]
        return np.concatenate([state, tradable], axis=-1).reshape(-1).astype(np.float32)

