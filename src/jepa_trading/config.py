from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("cache_dir", "processed_dir"):
        Path(config["data"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

