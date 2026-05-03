from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from industrial_maintenance_mlops.data.parser import (
    OPERATIONAL_SETTING_COLUMNS,
    SENSOR_COLUMNS,
)


@dataclass(frozen=True)
class WindowedDataset:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    feature_columns: list[str]
    target_column: str


def select_feature_columns(
    frame: pd.DataFrame,
    include_operational_settings: bool = True,
    include_sensors: bool = True,
) -> list[str]:
    """Select configured feature columns that are present in the frame."""
    columns: list[str] = []
    if include_operational_settings:
        columns.extend(column for column in OPERATIONAL_SETTING_COLUMNS if column in frame.columns)
    if include_sensors:
        columns.extend(column for column in SENSOR_COLUMNS if column in frame.columns)
    if not columns:
        raise ValueError("No feature columns selected")
    return columns


def build_windows(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "rul",
    window_size: int = 30,
    stride: int = 1,
    group_column: str = "unit_number",
    time_column: str = "time_in_cycles",
) -> WindowedDataset:
    """Build fixed-length windows from engine trajectories."""
    feature_list = list(feature_columns)
    _validate_window_inputs(frame, feature_list, target_column, window_size, stride, group_column, time_column)

    windows: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, int]] = []

    for unit_number, group in frame.sort_values([group_column, time_column]).groupby(group_column):
        ordered = group.sort_values(time_column)
        if len(ordered) < window_size:
            continue

        values = ordered[feature_list].to_numpy(dtype=float)
        labels = ordered[target_column].to_numpy(dtype=float)
        times = ordered[time_column].to_numpy(dtype=int)

        for start in range(0, len(ordered) - window_size + 1, stride):
            end = start + window_size
            windows.append(values[start:end])
            targets.append(float(labels[end - 1]))
            metadata.append(
                {
                    "unit_number": int(unit_number),
                    "start_cycle": int(times[start]),
                    "end_cycle": int(times[end - 1]),
                }
            )

    if not windows:
        raise ValueError("No windows generated; check window_size and trajectory lengths")

    return WindowedDataset(
        X=np.stack(windows).astype(float),
        y=np.asarray(targets, dtype=float),
        metadata=pd.DataFrame(metadata),
        feature_columns=feature_list,
        target_column=target_column,
    )


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """Flatten a 3D window tensor for sklearn estimators."""
    array = np.asarray(windows, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array shaped (samples, window, features), found {array.shape}")
    return array.reshape(array.shape[0], array.shape[1] * array.shape[2])


def build_failure_risk_targets(rul_values: np.ndarray, threshold: int = 30) -> np.ndarray:
    """Convert RUL values into binary failure-risk labels."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return (np.asarray(rul_values, dtype=float) <= threshold).astype(int)


def _validate_window_inputs(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    stride: int,
    group_column: str,
    time_column: str,
) -> None:
    if frame.empty:
        raise ValueError("Input frame is empty")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    required = set(feature_columns + [target_column, group_column, time_column])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
