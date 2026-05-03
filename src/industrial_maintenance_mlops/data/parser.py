from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
OPERATIONAL_SETTING_COLUMNS = ["operational_setting_1", "operational_setting_2", "operational_setting_3"]
SENSOR_COLUMNS = [f"sensor_{idx}" for idx in range(1, 22)]
CMAPSS_COLUMNS = INDEX_COLUMNS + OPERATIONAL_SETTING_COLUMNS + SENSOR_COLUMNS


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read a C-MAPSS train or test whitespace-delimited text file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"C-MAPSS file not found: {file_path}")

    LOGGER.info("Reading C-MAPSS file from %s", file_path)
    frame = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    if frame.empty:
        raise ValueError(f"C-MAPSS file is empty: {file_path}")
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"Expected {len(CMAPSS_COLUMNS)} columns in {file_path}, found {frame.shape[1]}"
        )

    frame.columns = CMAPSS_COLUMNS
    for column in CMAPSS_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["unit_number"] = frame["unit_number"].astype(int)
    frame["time_in_cycles"] = frame["time_in_cycles"].astype(int)
    return frame


def read_rul_file(path: str | Path) -> pd.Series:
    """Read a C-MAPSS RUL file and return values indexed by unit number."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"RUL file not found: {file_path}")

    LOGGER.info("Reading C-MAPSS RUL file from %s", file_path)
    frame = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    if frame.empty:
        raise ValueError(f"RUL file is empty: {file_path}")
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one column in RUL file {file_path}, found {frame.shape[1]}")

    values = pd.to_numeric(frame.iloc[:, 0], errors="raise").astype(int)
    values.index = range(1, len(values) + 1)
    values.index.name = "unit_number"
    values.name = "final_rul"
    return values


def load_cmapss_subset(
    raw_data_dir: str | Path,
    subset: str = "FD001",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load train, test, and RUL files for a C-MAPSS subset."""
    raw_dir = Path(raw_data_dir)
    train_path = raw_dir / f"train_{subset}.txt"
    test_path = raw_dir / f"test_{subset}.txt"
    rul_path = raw_dir / f"RUL_{subset}.txt"
    return read_cmapss_file(train_path), read_cmapss_file(test_path), read_rul_file(rul_path)
