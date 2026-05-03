from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    version: str
    model_type: str
    feature_columns: list[str]
    window_size: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_count(self) -> int:
        return len(self.feature_columns)


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> None:
    """Save a model bundle with inference metadata."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": bundle.model,
        "version": bundle.version,
        "model_type": bundle.model_type,
        "feature_columns": bundle.feature_columns,
        "window_size": bundle.window_size,
        "metadata": bundle.metadata,
    }
    joblib.dump(payload, model_path)


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load a model bundle saved by save_model_bundle."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    payload = joblib.load(model_path)
    if isinstance(payload, ModelBundle):
        return payload
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unsupported model bundle format: {model_path}")

    return ModelBundle(
        model=payload["model"],
        version=str(payload.get("version", "unknown")),
        model_type=str(payload.get("model_type", "unknown")),
        feature_columns=list(payload.get("feature_columns", [])),
        window_size=int(payload.get("window_size", 0)),
        metadata=dict(payload.get("metadata", {})),
    )
