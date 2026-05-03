from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_rul_targets(
    frame: pd.DataFrame,
    max_rul: int | None = 125,
    target_column: str = "rul",
    raw_target_column: str = "rul_raw",
) -> pd.DataFrame:
    """Generate RUL labels for complete training trajectories."""
    _validate_engine_frame(frame)
    if max_rul is not None and max_rul <= 0:
        raise ValueError("max_rul must be positive when provided")

    labeled = frame.copy()
    max_cycles = labeled.groupby("unit_number")["time_in_cycles"].transform("max")
    labeled[raw_target_column] = max_cycles - labeled["time_in_cycles"]
    labeled[target_column] = labeled[raw_target_column]
    if max_rul is not None:
        labeled[target_column] = labeled[target_column].clip(upper=max_rul)
    return labeled


def add_test_rul_targets(
    test_frame: pd.DataFrame,
    final_rul: pd.Series | Sequence[int],
    max_rul: int | None = 125,
    target_column: str = "rul",
    raw_target_column: str = "rul_raw",
) -> pd.DataFrame:
    """Generate RUL labels for partial test trajectories using final RUL values."""
    _validate_engine_frame(test_frame)
    if max_rul is not None and max_rul <= 0:
        raise ValueError("max_rul must be positive when provided")

    final_rul_by_unit = _coerce_final_rul(final_rul)
    units = set(test_frame["unit_number"].unique())
    missing_units = sorted(units.difference(final_rul_by_unit.index))
    if missing_units:
        raise ValueError(f"Missing final RUL values for unit numbers: {missing_units}")

    labeled = test_frame.copy()
    observed_end = labeled.groupby("unit_number")["time_in_cycles"].transform("max")
    final_values = labeled["unit_number"].map(final_rul_by_unit)
    labeled[raw_target_column] = final_values + (observed_end - labeled["time_in_cycles"])
    labeled[target_column] = labeled[raw_target_column]
    if max_rul is not None:
        labeled[target_column] = labeled[target_column].clip(upper=max_rul)
    return labeled


def _coerce_final_rul(final_rul: pd.Series | Sequence[int]) -> pd.Series:
    if isinstance(final_rul, pd.Series):
        values = final_rul.copy()
        if values.index.name != "unit_number":
            values.index.name = "unit_number"
        return values.astype(int)

    values = pd.Series(list(final_rul), dtype=int, name="final_rul")
    values.index = range(1, len(values) + 1)
    values.index.name = "unit_number"
    return values


def _validate_engine_frame(frame: pd.DataFrame) -> None:
    required = {"unit_number", "time_in_cycles"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Engine time-series frame is empty")
