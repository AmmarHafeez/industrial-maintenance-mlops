from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_maintenance_mlops.data.parser import read_cmapss_file
from industrial_maintenance_mlops.training.pipeline import (
    TrainingPipelineConfig,
    build_model_version,
    run_training_with_config,
    split_engine_ids,
)


def test_engine_level_split_has_no_engine_id_overlap(tiny_cmapss_file):
    frame = read_cmapss_file(tiny_cmapss_file)

    train_ids, validation_ids = split_engine_ids(frame, test_size=0.5, random_state=42)

    assert train_ids
    assert validation_ids
    assert train_ids.isdisjoint(validation_ids)


def test_training_pipeline_runs_on_synthetic_data_and_writes_artifacts(
    tiny_cmapss_dataset_dir,
    tmp_path,
):
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    config = TrainingPipelineConfig(
        dataset_id="FD001",
        raw_dir=tiny_cmapss_dataset_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        window_size=2,
        stride=1,
        max_rul=4,
        risk_threshold=1,
        test_size=0.4,
        random_state=42,
        classifier_estimator="random_forest",
        regressor_n_estimators=2,
        classifier_n_estimators=2,
    )

    report = run_training_with_config(config)
    metrics_payload = json.loads(config.metrics_path.read_text(encoding="utf-8"))

    assert (models_dir / "rul_regressor.joblib").exists()
    assert (models_dir / "failure_risk_classifier.joblib").exists()
    assert (models_dir / "reference_stats.json").exists()
    assert config.metrics_path.exists()
    assert "regression" in report
    assert "classification" in report
    assert "model_version" in metrics_payload
    assert metrics_payload["configuration"]["classifier_estimator"] == "random_forest"
    assert metrics_payload["configuration"]["classifier_max_iter"] == 2000
    assert set(metrics_payload["regression"]) == {"mae", "rmse", "r2"}
    assert {
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "precision",
        "recall",
        "confusion_matrix",
    }.issubset(metrics_payload["classification"])


def test_missing_raw_files_raise_clear_error(tmp_path):
    config = TrainingPipelineConfig(
        dataset_id="FD001",
        raw_dir=tmp_path / "missing",
        models_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        window_size=2,
    )

    with pytest.raises(FileNotFoundError, match="Missing required C-MAPSS files"):
        run_training_with_config(config)


def test_generated_artifact_paths_are_ignored_by_git():
    ignore_text = Path(".gitignore").read_text(encoding="utf-8")

    assert "models/" in ignore_text
    assert "reports/metrics/" in ignore_text
    assert "data/raw/" in ignore_text


def test_model_version_names_scaled_logistic_regression():
    config = TrainingPipelineConfig(classifier_estimator="logistic_regression")

    assert "scaled-logistic-regression" in build_model_version(config)
