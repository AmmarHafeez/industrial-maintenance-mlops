from __future__ import annotations

from pathlib import Path

import pytest


def make_cmapss_row(unit_number: int, cycle: int, sensor_offset: float = 0.0) -> str:
    settings = [0.1, 0.2, 0.3]
    sensors = [sensor_offset + cycle + sensor_index for sensor_index in range(1, 22)]
    values = [unit_number, cycle, *settings, *sensors]
    return " ".join(str(value) for value in values)


@pytest.fixture
def tiny_cmapss_file(tmp_path: Path) -> Path:
    rows = [
        make_cmapss_row(1, 1),
        make_cmapss_row(1, 2),
        make_cmapss_row(1, 3),
        make_cmapss_row(2, 1, sensor_offset=10.0),
        make_cmapss_row(2, 2, sensor_offset=10.0),
        make_cmapss_row(2, 3, sensor_offset=10.0),
    ]
    path = tmp_path / "train_FD001.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
