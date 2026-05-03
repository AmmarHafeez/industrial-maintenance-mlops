from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferenceStats:
    mean: list[float]
    std: list[float]
    feature_columns: list[str]
    window_size: int


@dataclass(frozen=True)
class DriftResult:
    is_drifted: bool
    max_abs_z_score: float
    drifted_features: list[str]
    z_scores: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DriftDetector:
    def __init__(self, reference_stats: ReferenceStats | None, z_threshold: float = 3.0) -> None:
        self.reference_stats = reference_stats
        self.z_threshold = z_threshold

    def compare(self, window: list[list[float]] | np.ndarray) -> DriftResult | None:
        if self.reference_stats is None:
            return None
        return compare_window_to_reference(window, self.reference_stats, self.z_threshold)


def compute_reference_stats(
    windows: np.ndarray,
    feature_columns: list[str],
    window_size: int | None = None,
) -> ReferenceStats:
    """Compute reference mean and standard deviation from training windows."""
    array = np.asarray(windows, dtype=float)
    if array.ndim == 3:
        flattened = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
        inferred_window_size = array.shape[1]
    elif array.ndim == 2:
        flattened = array
        inferred_window_size = window_size or 1
    else:
        raise ValueError("windows must be a 2D or 3D array")
    if flattened.shape[1] != len(feature_columns):
        raise ValueError("Feature column count does not match window feature count")

    return ReferenceStats(
        mean=flattened.mean(axis=0).astype(float).tolist(),
        std=flattened.std(axis=0).astype(float).tolist(),
        feature_columns=feature_columns,
        window_size=window_size or inferred_window_size,
    )


def compare_window_to_reference(
    window: list[list[float]] | np.ndarray,
    reference_stats: ReferenceStats,
    z_threshold: float = 3.0,
    epsilon: float = 1e-8,
) -> DriftResult:
    """Compare a single incoming window to reference feature statistics."""
    array = np.asarray(window, dtype=float)
    if array.ndim != 2:
        raise ValueError("window must be a 2D array")
    if array.shape[1] != len(reference_stats.feature_columns):
        raise ValueError("Window feature count does not match reference statistics")

    incoming_mean = array.mean(axis=0)
    reference_mean = np.asarray(reference_stats.mean, dtype=float)
    reference_std = np.asarray(reference_stats.std, dtype=float)
    safe_std = np.where(reference_std < epsilon, epsilon, reference_std)
    z_scores = (incoming_mean - reference_mean) / safe_std
    abs_scores = np.abs(z_scores)

    drifted_indices = np.where(abs_scores > z_threshold)[0]
    drifted_features = [reference_stats.feature_columns[index] for index in drifted_indices]
    z_score_map = {
        feature: float(score)
        for feature, score in zip(reference_stats.feature_columns, z_scores, strict=True)
    }

    return DriftResult(
        is_drifted=bool(len(drifted_features)),
        max_abs_z_score=float(abs_scores.max()) if len(abs_scores) else 0.0,
        drifted_features=drifted_features,
        z_scores=z_score_map,
    )


def save_reference_stats(stats: ReferenceStats, path: str | Path) -> None:
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(stats), file, indent=2)


def load_reference_stats(path: str | Path) -> ReferenceStats:
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Reference statistics file not found: {stats_path}")
    with stats_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return ReferenceStats(
        mean=list(payload["mean"]),
        std=list(payload["std"]),
        feature_columns=list(payload["feature_columns"]),
        window_size=int(payload["window_size"]),
    )
