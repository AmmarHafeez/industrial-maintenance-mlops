from __future__ import annotations

from industrial_maintenance_mlops.data.parser import read_cmapss_file
from industrial_maintenance_mlops.data.rul import add_rul_targets, add_test_rul_targets
from industrial_maintenance_mlops.features.windowing import (
    build_failure_risk_targets,
    build_windows,
    flatten_windows,
)


def test_rul_target_generation_clips_training_trajectories(tiny_cmapss_file):
    frame = read_cmapss_file(tiny_cmapss_file)
    labeled = add_rul_targets(frame, max_rul=1)

    unit_one = labeled[labeled["unit_number"] == 1]
    assert unit_one["rul_raw"].tolist() == [2, 1, 0]
    assert unit_one["rul"].tolist() == [1, 1, 0]


def test_test_rul_generation_uses_final_rul(tiny_cmapss_file):
    frame = read_cmapss_file(tiny_cmapss_file)
    labeled = add_test_rul_targets(frame, final_rul=[5, 8], max_rul=None)

    unit_two = labeled[labeled["unit_number"] == 2]
    assert unit_two["rul"].tolist() == [10, 9, 8]


def test_window_generation_returns_fixed_length_windows(tiny_cmapss_file):
    frame = read_cmapss_file(tiny_cmapss_file)
    labeled = add_rul_targets(frame, max_rul=125)
    feature_columns = ["sensor_1", "sensor_2"]

    windows = build_windows(labeled, feature_columns, window_size=2, stride=1)

    assert windows.X.shape == (4, 2, 2)
    assert windows.y.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert windows.metadata["end_cycle"].tolist() == [2, 3, 2, 3]
    assert flatten_windows(windows.X).shape == (4, 4)


def test_failure_risk_targets_use_threshold():
    labels = build_failure_risk_targets([40, 30, 5], threshold=30)

    assert labels.tolist() == [0, 1, 1]
