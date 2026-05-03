"""Feature engineering utilities."""

from industrial_maintenance_mlops.features.windowing import (
    WindowedDataset,
    build_failure_risk_targets,
    build_windows,
    flatten_windows,
    select_feature_columns,
)

__all__ = [
    "WindowedDataset",
    "build_failure_risk_targets",
    "build_windows",
    "flatten_windows",
    "select_feature_columns",
]
