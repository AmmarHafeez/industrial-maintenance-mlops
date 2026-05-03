from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from industrial_maintenance_mlops.data.parser import read_cmapss_file, read_rul_file
from industrial_maintenance_mlops.evaluation.evaluate_cmapss import (
    EvaluationConfig,
    extract_final_windows,
    run_evaluation,
)
from industrial_maintenance_mlops.models.persistence import ModelBundle, save_model_bundle


def write_test_model_bundles(models_dir: Path, version: str = "test-version") -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = ["sensor_1", "sensor_2"]
    X = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )
    y_rul = np.array([80.0, 50.0, 10.0, 0.0])
    y_risk = np.array([0, 0, 1, 1])
    regressor = LinearRegression().fit(X, y_rul)
    classifier = RandomForestClassifier(n_estimators=2, random_state=42).fit(X, y_risk)

    save_model_bundle(
        ModelBundle(
            model=regressor,
            version=version,
            model_type="rul_regression",
            feature_columns=feature_columns,
            window_size=2,
        ),
        models_dir / "rul_regressor.joblib",
    )
    save_model_bundle(
        ModelBundle(
            model=classifier,
            version=version,
            model_type="failure_risk_classification",
            feature_columns=feature_columns,
            window_size=2,
        ),
        models_dir / "failure_risk_classifier.joblib",
    )


def test_missing_model_artifacts_raise_clear_error(tiny_cmapss_dataset_dir, tmp_path):
    config = EvaluationConfig(
        raw_dir=tiny_cmapss_dataset_dir,
        models_dir=tmp_path / "missing_models",
        metrics_dir=tmp_path / "metrics",
        window_size=2,
    )

    with pytest.raises(FileNotFoundError, match="Missing required evaluation files"):
        run_evaluation(config)


def test_missing_test_and_rul_files_raise_clear_error(tmp_path):
    models_dir = tmp_path / "models"
    write_test_model_bundles(models_dir)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = EvaluationConfig(
        raw_dir=raw_dir,
        models_dir=models_dir,
        metrics_dir=tmp_path / "metrics",
        window_size=2,
    )

    with pytest.raises(FileNotFoundError, match="test_FD001.txt"):
        run_evaluation(config)


def test_final_window_extraction_per_engine(tiny_cmapss_dataset_dir):
    test_frame = read_cmapss_file(tiny_cmapss_dataset_dir / "test_FD001.txt")
    final_rul = read_rul_file(tiny_cmapss_dataset_dir / "RUL_FD001.txt")

    dataset = extract_final_windows(
        test_frame=test_frame,
        final_rul=final_rul,
        feature_columns=["sensor_1", "sensor_2"],
        window_size=2,
        max_rul=125,
    )

    assert dataset.X.shape == (5, 2, 2)
    assert dataset.engine_ids == [1, 2, 3, 4, 5]
    assert dataset.X[0].tolist() == [[13.0, 14.0], [14.0, 15.0]]
    assert dataset.y_rul.tolist() == [2.0, 2.0, 2.0, 2.0, 2.0]


def test_evaluation_writes_metrics_json_with_expected_sections(
    tiny_cmapss_dataset_dir,
    tmp_path,
):
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    write_test_model_bundles(models_dir)
    (tiny_cmapss_dataset_dir / "RUL_FD001.txt").write_text("4\n3\n2\n1\n0\n", encoding="utf-8")
    config = EvaluationConfig(
        raw_dir=tiny_cmapss_dataset_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        window_size=2,
        max_rul=125,
        risk_threshold=2,
    )

    report = run_evaluation(config)
    metrics_payload = json.loads(config.metrics_path.read_text(encoding="utf-8"))

    assert config.metrics_path.exists()
    assert report["dataset_id"] == "FD001"
    assert metrics_payload["model_version"] == "test-version"
    assert metrics_payload["data"]["engine_count"] == 5
    assert set(metrics_payload["regression"]) == {"mae", "rmse", "r2"}
    assert {
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "precision",
        "recall",
        "confusion_matrix",
    }.issubset(metrics_payload["classification"])
