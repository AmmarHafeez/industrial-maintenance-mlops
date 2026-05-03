from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from industrial_maintenance_mlops.data.parser import read_cmapss_file
from industrial_maintenance_mlops.data.rul import add_rul_targets
from industrial_maintenance_mlops.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
)
from industrial_maintenance_mlops.features.windowing import (
    build_failure_risk_targets,
    build_windows,
    flatten_windows,
    select_feature_columns,
)
from industrial_maintenance_mlops.models.baselines import (
    train_failure_classifier,
    train_random_forest_regressor,
)
from industrial_maintenance_mlops.models.persistence import ModelBundle, save_model_bundle
from industrial_maintenance_mlops.monitoring.drift import (
    compute_reference_stats,
    save_reference_stats,
)
from industrial_maintenance_mlops.utils.config import load_config_dir
from industrial_maintenance_mlops.utils.io import write_json
from industrial_maintenance_mlops.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def run_training(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Run the baseline C-MAPSS training pipeline."""
    configure_logging()
    configs = load_config_dir(config_dir)
    data_config = configs["data"]
    model_config = configs["model"]
    training_config = configs["training"]

    raw_dir = Path(data_config["raw_data_dir"])
    train_path = raw_dir / str(data_config["train_file"])
    max_rul = int(data_config.get("max_rul", 125))

    LOGGER.info("Loading training data from %s", train_path)
    train_frame = read_cmapss_file(train_path)
    labeled_frame = add_rul_targets(train_frame, max_rul=max_rul)
    feature_columns = select_feature_columns(
        labeled_frame,
        include_operational_settings=bool(data_config.get("include_operational_settings", True)),
        include_sensors=bool(data_config.get("include_sensors", True)),
    )

    windows = build_windows(
        labeled_frame,
        feature_columns=feature_columns,
        window_size=int(training_config.get("window_size", 30)),
        stride=int(training_config.get("stride", 1)),
    )
    X = flatten_windows(windows.X)
    y_rul = windows.y
    y_risk = build_failure_risk_targets(
        y_rul,
        threshold=int(model_config["classification"].get("failure_risk_threshold", 30)),
    )

    train_index, validation_index = _split_indices(
        sample_count=len(y_rul),
        validation_size=float(training_config.get("validation_size", 0.2)),
        random_state=int(model_config.get("random_state", 42)),
    )

    regressor = train_random_forest_regressor(
        X[train_index],
        y_rul[train_index],
        n_estimators=int(model_config["regression"].get("n_estimators", 100)),
        max_depth=model_config["regression"].get("max_depth"),
        random_state=int(model_config.get("random_state", 42)),
    )
    classifier = train_failure_classifier(
        X[train_index],
        y_risk[train_index],
        estimator=str(model_config["classification"].get("estimator", "logistic_regression")),
        n_estimators=int(model_config["classification"].get("n_estimators", 100)),
        max_depth=model_config["classification"].get("max_depth"),
        max_iter=int(model_config["classification"].get("max_iter", 1000)),
        random_state=int(model_config.get("random_state", 42)),
    )

    regression_report = regression_metrics(y_rul[validation_index], regressor.predict(X[validation_index]))
    classification_report = classification_metrics(
        y_risk[validation_index],
        classifier.predict(X[validation_index]),
    )

    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    models_dir = Path(training_config.get("models_dir", "models"))
    metrics_dir = Path(training_config.get("metrics_dir", "reports/metrics"))
    reference_stats_path = Path(training_config.get("reference_stats_path", "models/reference_stats.json"))

    save_model_bundle(
        ModelBundle(
            model=regressor,
            version=version,
            model_type="rul_regression",
            feature_columns=feature_columns,
            window_size=windows.X.shape[1],
            metadata={"max_rul": max_rul},
        ),
        models_dir / "rul_regressor.joblib",
    )
    save_model_bundle(
        ModelBundle(
            model=classifier,
            version=version,
            model_type="failure_risk_classification",
            feature_columns=feature_columns,
            window_size=windows.X.shape[1],
            metadata={
                "failure_risk_threshold": int(
                    model_config["classification"].get("failure_risk_threshold", 30)
                )
            },
        ),
        models_dir / "failure_risk_classifier.joblib",
    )

    reference_stats = compute_reference_stats(windows.X[train_index], feature_columns)
    save_reference_stats(reference_stats, reference_stats_path)

    report = {
        "model_version": version,
        "subset": data_config.get("subset", "FD001"),
        "sample_count": int(len(y_rul)),
        "window_size": int(windows.X.shape[1]),
        "feature_count": int(windows.X.shape[2]),
        "max_rul": max_rul,
        "regression": regression_report,
        "classification": classification_report,
    }
    write_json(report, metrics_dir / "training_metrics.json")
    LOGGER.info("Training complete. Metrics written to %s", metrics_dir / "training_metrics.json")
    return report


def _split_indices(
    sample_count: int,
    validation_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count < 2:
        raise ValueError("At least two windows are required for training")
    indices = np.arange(sample_count)
    train_index, validation_index = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )
    return np.asarray(train_index), np.asarray(validation_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline C-MAPSS training pipeline.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing YAML configs.")
    args = parser.parse_args()
    run_training(args.config_dir)


if __name__ == "__main__":
    main()
