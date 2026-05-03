from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_maintenance_mlops.monitoring.drift import ReferenceStats, save_reference_stats
from industrial_maintenance_mlops.monitoring.drift_report import (
    DriftReportConfig,
    run_drift_report,
)


def write_reference_stats(path: Path) -> None:
    save_reference_stats(
        ReferenceStats(
            mean=[0.0, 0.0],
            std=[1.0, 1.0],
            feature_columns=["sensor_1", "sensor_2"],
            window_size=2,
        ),
        path,
    )


def test_drift_report_json_is_written_with_expected_sections(tiny_cmapss_dataset_dir, tmp_path):
    reference_path = tmp_path / "reference_stats.json"
    output_path = tmp_path / "metrics" / "drift_report_FD001.json"
    write_reference_stats(reference_path)
    config = DriftReportConfig(
        raw_dir=tiny_cmapss_dataset_dir,
        reference_stats_path=reference_path,
        output_path=output_path,
        window_size=2,
        mean_z_threshold=1.0,
    )

    report = run_drift_report(config)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert report["dataset_id"] == "FD001"
    assert "summary" in payload
    assert "per_feature" in payload
    assert payload["summary"]["feature_count"] == 2
    assert payload["summary"]["window_count"] == 5
    assert set(payload["per_feature"]) == {"sensor_1", "sensor_2"}


def test_missing_reference_stats_raises_clear_error(tiny_cmapss_dataset_dir, tmp_path):
    config = DriftReportConfig(
        raw_dir=tiny_cmapss_dataset_dir,
        reference_stats_path=tmp_path / "missing_reference_stats.json",
        output_path=tmp_path / "drift_report.json",
        window_size=2,
    )

    with pytest.raises(FileNotFoundError, match="Missing required drift report files"):
        run_drift_report(config)


def test_missing_test_file_raises_clear_error(tmp_path):
    reference_path = tmp_path / "reference_stats.json"
    write_reference_stats(reference_path)
    config = DriftReportConfig(
        raw_dir=tmp_path / "missing_raw",
        reference_stats_path=reference_path,
        output_path=tmp_path / "drift_report.json",
        window_size=2,
    )

    with pytest.raises(FileNotFoundError, match="test_FD001.txt"):
        run_drift_report(config)


def test_drift_report_produces_feature_flags(tiny_cmapss_dataset_dir, tmp_path):
    reference_path = tmp_path / "reference_stats.json"
    output_path = tmp_path / "drift_report.json"
    write_reference_stats(reference_path)
    config = DriftReportConfig(
        raw_dir=tiny_cmapss_dataset_dir,
        reference_stats_path=reference_path,
        output_path=output_path,
        window_size=2,
        mean_z_threshold=1.0,
        std_ratio_threshold=2.0,
    )

    payload = run_drift_report(config)

    assert payload["summary"]["number_of_features_flagged"] >= 1
    assert any(values["flagged"] for values in payload["per_feature"].values())
