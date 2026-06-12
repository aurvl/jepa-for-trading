from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ExperienceRecord:
    date: str
    horizon: int
    predicted_cost: float
    realized_cost: float | None = None
    prediction_error: float | None = None
    regime: str = "unknown"
    portfolio_equity: float | None = None
    primitive_action: dict[str, float] = field(default_factory=dict)
    predicted_outcomes: dict[str, float] = field(default_factory=dict)
    realized_outcomes: dict[str, float] = field(default_factory=dict)
    portfolio_state_t: dict[str, float] = field(default_factory=dict)
    portfolio_state_tp1: dict[str, float] = field(default_factory=dict)


class ExperienceBuffer:
    def __init__(self) -> None:
        self.records: list[ExperienceRecord] = []

    def append(self, record: ExperienceRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for record in self.records:
            base = asdict(record)
            row = {k: v for k, v in base.items() if not isinstance(v, dict)}
            for prefix in ("primitive_action", "predicted_outcomes", "realized_outcomes", "portfolio_state_t", "portfolio_state_tp1"):
                for key, value in base[prefix].items():
                    if isinstance(value, (int, float, np.floating)):
                        row[f"{prefix}_{key}"] = float(value)
            rows.append(row)
        return pd.DataFrame(rows)

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)

    def save_parquet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_parquet(path, index=False)
