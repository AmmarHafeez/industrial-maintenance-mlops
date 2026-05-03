from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must contain a mapping: {config_path}")
    return payload


def load_config_dir(config_dir: str | Path) -> dict[str, dict[str, Any]]:
    directory = Path(config_dir)
    required_files = {
        "data": directory / "data.yaml",
        "model": directory / "model.yaml",
        "training": directory / "training.yaml",
        "api": directory / "api.yaml",
    }
    return {name: load_yaml(path) for name, path in required_files.items()}
