from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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


@dataclass(frozen=True)
class TrainingPipelineConfig:
    dataset_id: str = "FD001"
    raw_dir: Path = Path("data/raw/CMAPSSData")
    models_dir: Path = Path("models")
    metrics_dir: Path = Path("reports/metrics")
    window_size: int = 30
    stride: int = 1
    max_rul: int = 125
    risk_threshold: int = 30
    test_size: float = 0.2
    random_state: int = 42
    include_operational_settings: bool = True
    include_sensors: bool = True
    classifier_estimator: str = "logistic_regression"
    regressor_n_estimators: int = 100
    regressor_max_depth: int | None = None
    classifier_n_estimators: int = 100
    classifier_max_depth: int | None = None
    classifier_max_iter: int = 1000

    @property
    def train_file(self) -> Path:
        return self.raw_dir / f"train_{self.dataset_id}.txt"

    @property
    def test_file(self) -> Path:
        return self.raw_dir / f"test_{self.dataset_id}.txt"

    @property
    def rul_file(self) -> Path:
        return self.raw_dir / f"RUL_{self.dataset_id}.txt"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / f"training_metrics_{self.dataset_id}.json"

    @property
    def reference_stats_path(self) -> Path:
        return self.models_dir / "reference_stats.json"


def run_training(
    config_dir: str | Path = "configs",
    dataset_id: str | None = None,
    raw_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    max_rul: int | None = None,
    risk_threshold: int | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Run the baseline C-MAPSS training and validation workflow."""
    configure_logging()
    config = build_training_config(
        config_dir=config_dir,
        dataset_id=dataset_id,
        raw_dir=raw_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        window_size=window_size,
        stride=stride,
        max_rul=max_rul,
        risk_threshold=risk_threshold,
        test_size=test_size,
        random_state=random_state,
    )
    return run_training_with_config(config)


def build_training_config(
    config_dir: str | Path = "configs",
    dataset_id: str | None = None,
    raw_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    max_rul: int | None = None,
    risk_threshold: int | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> TrainingPipelineConfig:
    configs = load_config_dir(config_dir)
    data_config = configs["data"]
    model_config = configs["model"]
    training_config = configs["training"]
    classification_config = model_config.get("classification", {})
    regression_config = model_config.get("regression", {})

    return TrainingPipelineConfig(
        dataset_id=dataset_id or str(data_config.get("subset", "FD001")),
        raw_dir=Path(raw_dir or data_config.get("raw_data_dir", "data/raw/CMAPSSData")),
        models_dir=Path(models_dir or training_config.get("models_dir", "models")),
        metrics_dir=Path(metrics_dir or training_config.get("metrics_dir", "reports/metrics")),
        window_size=int(_override_or_default(window_size, training_config.get("window_size", 30))),
        stride=int(_override_or_default(stride, training_config.get("stride", 1))),
        max_rul=int(_override_or_default(max_rul, data_config.get("max_rul", 125))),
        risk_threshold=int(
            _override_or_default(
                risk_threshold,
                classification_config.get("failure_risk_threshold", 30),
            )
        ),
        test_size=float(
            _override_or_default(
                test_size,
                training_config.get("test_size", training_config.get("validation_size", 0.2)),
            )
        ),
        random_state=int(_override_or_default(random_state, model_config.get("random_state", 42))),
        include_operational_settings=bool(data_config.get("include_operational_settings", True)),
        include_sensors=bool(data_config.get("include_sensors", True)),
        classifier_estimator=str(classification_config.get("estimator", "logistic_regression")),
        regressor_n_estimators=int(regression_config.get("n_estimators", 100)),
        regressor_max_depth=regression_config.get("max_depth"),
        classifier_n_estimators=int(classification_config.get("n_estimators", 100)),
        classifier_max_depth=classification_config.get("max_depth"),
        classifier_max_iter=int(classification_config.get("max_iter", 1000)),
    )


def _override_or_default(value: Any | None, default: Any) -> Any:
    return default if value is None else value


def run_training_with_config(config: TrainingPipelineConfig) -> dict[str, Any]:
    validate_training_config(config)
    validate_required_raw_files(config)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading C-MAPSS training data from %s", config.train_file)
    train_frame = read_cmapss_file(config.train_file)
    labeled_frame = add_rul_targets(train_frame, max_rul=config.max_rul)
    feature_columns = select_feature_columns(
        labeled_frame,
        include_operational_settings=config.include_operational_settings,
        include_sensors=config.include_sensors,
    )

    train_engine_ids, validation_engine_ids = split_engine_ids(
        labeled_frame,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    LOGGER.info(
        "Split %s engines into %s training and %s validation engines",
        labeled_frame["unit_number"].nunique(),
        len(train_engine_ids),
        len(validation_engine_ids),
    )

    LOGGER.info("Building windows with window_size=%s and stride=%s", config.window_size, config.stride)
    windows = build_windows(
        labeled_frame,
        feature_columns=feature_columns,
        window_size=config.window_size,
        stride=config.stride,
    )
    train_index, validation_index = window_indices_for_engine_split(
        windows.metadata,
        train_engine_ids,
        validation_engine_ids,
    )
    if len(train_index) == 0 or len(validation_index) == 0:
        raise ValueError("Engine-level split produced empty train or validation windows")

    X = flatten_windows(windows.X)
    y_rul = windows.y
    y_risk = build_failure_risk_targets(y_rul, threshold=config.risk_threshold)

    LOGGER.info("Training RUL regressor on %s windows", len(train_index))
    regressor = train_random_forest_regressor(
        X[train_index],
        y_rul[train_index],
        n_estimators=config.regressor_n_estimators,
        max_depth=config.regressor_max_depth,
        random_state=config.random_state,
    )

    LOGGER.info("Training failure-risk classifier on %s windows", len(train_index))
    classifier = train_failure_classifier(
        X[train_index],
        y_risk[train_index],
        estimator=config.classifier_estimator,
        n_estimators=config.classifier_n_estimators,
        max_depth=config.classifier_max_depth,
        max_iter=config.classifier_max_iter,
        random_state=config.random_state,
    )

    regression_report = regression_metrics(
        y_rul[validation_index],
        regressor.predict(X[validation_index]),
    )
    classification_report = classification_metrics(
        y_risk[validation_index],
        classifier.predict(X[validation_index]),
    )
    model_version = build_model_version(config)

    save_model_bundle(
        ModelBundle(
            model=regressor,
            version=model_version,
            model_type="rul_regression",
            feature_columns=feature_columns,
            window_size=windows.X.shape[1],
            metadata={"dataset_id": config.dataset_id, "max_rul": config.max_rul},
        ),
        config.models_dir / "rul_regressor.joblib",
    )
    save_model_bundle(
        ModelBundle(
            model=classifier,
            version=model_version,
            model_type="failure_risk_classification",
            feature_columns=feature_columns,
            window_size=windows.X.shape[1],
            metadata={
                "dataset_id": config.dataset_id,
                "failure_risk_threshold": config.risk_threshold,
            },
        ),
        config.models_dir / "failure_risk_classifier.joblib",
    )

    reference_stats = compute_reference_stats(windows.X[train_index], feature_columns)
    save_reference_stats(reference_stats, config.reference_stats_path)

    report = build_metrics_report(
        config=config,
        model_version=model_version,
        feature_count=windows.X.shape[2],
        train_engine_ids=train_engine_ids,
        validation_engine_ids=validation_engine_ids,
        train_window_count=len(train_index),
        validation_window_count=len(validation_index),
        regression_report=regression_report,
        classification_report=classification_report,
    )
    write_json(report, config.metrics_path)
    LOGGER.info("Training metrics written to %s", config.metrics_path)
    return report


def validate_training_config(config: TrainingPipelineConfig) -> None:
    if not config.dataset_id:
        raise ValueError("dataset_id must not be empty")
    if config.window_size <= 0:
        raise ValueError("window_size must be positive")
    if config.stride <= 0:
        raise ValueError("stride must be positive")
    if config.max_rul <= 0:
        raise ValueError("max_rul must be positive")
    if config.risk_threshold < 0:
        raise ValueError("risk_threshold must be non-negative")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")


def validate_required_raw_files(config: TrainingPipelineConfig) -> None:
    required_paths = [config.train_file, config.test_file, config.rul_file]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing required C-MAPSS files for {config.dataset_id} under {config.raw_dir}: {names}"
        )


def split_engine_ids(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    engine_column: str = "unit_number",
) -> tuple[set[int], set[int]]:
    if engine_column not in frame.columns:
        raise ValueError(f"Missing engine column: {engine_column}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    engine_ids = np.asarray(sorted(frame[engine_column].unique()), dtype=int)
    if len(engine_ids) < 2:
        raise ValueError("At least two engines are required for an engine-level split")

    train_ids, validation_ids = train_test_split(
        engine_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return set(map(int, train_ids)), set(map(int, validation_ids))


def window_indices_for_engine_split(
    metadata: pd.DataFrame,
    train_engine_ids: set[int],
    validation_engine_ids: set[int],
    engine_column: str = "unit_number",
) -> tuple[np.ndarray, np.ndarray]:
    if engine_column not in metadata.columns:
        raise ValueError(f"Missing engine column in window metadata: {engine_column}")
    overlap = train_engine_ids.intersection(validation_engine_ids)
    if overlap:
        raise ValueError(f"Engine split contains overlapping engine ids: {sorted(overlap)}")

    train_mask = metadata[engine_column].isin(train_engine_ids).to_numpy()
    validation_mask = metadata[engine_column].isin(validation_engine_ids).to_numpy()
    return np.flatnonzero(train_mask), np.flatnonzero(validation_mask)


def build_model_version(config: TrainingPipelineConfig) -> str:
    classifier_name = config.classifier_estimator.replace("_", "-")
    return (
        f"{config.dataset_id}-win{config.window_size}-stride{config.stride}"
        f"-rul{config.max_rul}-risk{config.risk_threshold}"
        f"-{classifier_name}-seed{config.random_state}"
    )


def build_metrics_report(
    config: TrainingPipelineConfig,
    model_version: str,
    feature_count: int,
    train_engine_ids: set[int],
    validation_engine_ids: set[int],
    train_window_count: int,
    validation_window_count: int,
    regression_report: dict[str, float],
    classification_report: dict[str, object],
) -> dict[str, Any]:
    config_payload = asdict(config)
    config_payload["raw_dir"] = str(config.raw_dir)
    config_payload["models_dir"] = str(config.models_dir)
    config_payload["metrics_dir"] = str(config.metrics_dir)

    return {
        "dataset_id": config.dataset_id,
        "model_version": model_version,
        "configuration": config_payload,
        "data": {
            "train_file": config.train_file.name,
            "test_file": config.test_file.name,
            "rul_file": config.rul_file.name,
            "train_engine_count": len(train_engine_ids),
            "validation_engine_count": len(validation_engine_ids),
            "train_window_count": train_window_count,
            "validation_window_count": validation_window_count,
            "feature_count": feature_count,
            "window_size": config.window_size,
        },
        "regression": regression_report,
        "classification": classification_report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train C-MAPSS baseline maintenance models.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing YAML configs.")
    parser.add_argument("--dataset-id", default=None, help="C-MAPSS dataset id such as FD001.")
    parser.add_argument("--raw-dir", default=None, help="Directory containing C-MAPSS raw text files.")
    parser.add_argument("--models-dir", default=None, help="Directory for trained model files.")
    parser.add_argument("--metrics-dir", default=None, help="Directory for metrics JSON files.")
    parser.add_argument("--window-size", type=int, default=None, help="Number of cycles per window.")
    parser.add_argument("--stride", type=int, default=None, help="Window stride in cycles.")
    parser.add_argument("--max-rul", type=int, default=None, help="Maximum clipped RUL target.")
    parser.add_argument(
        "--risk-threshold",
        type=int,
        default=None,
        help="RUL threshold for failure-risk labels.",
    )
    parser.add_argument("--test-size", type=float, default=None, help="Validation engine fraction.")
    parser.add_argument("--random-state", type=int, default=None, help="Deterministic random seed.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_training(
        config_dir=args.config_dir,
        dataset_id=args.dataset_id,
        raw_dir=args.raw_dir,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        window_size=args.window_size,
        stride=args.stride,
        max_rul=args.max_rul,
        risk_threshold=args.risk_threshold,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(format_completion_summary(report))


def format_completion_summary(report: dict[str, Any]) -> str:
    data = report["data"]
    configuration = report["configuration"]
    return (
        "Training complete: "
        f"dataset={report['dataset_id']}, "
        f"train_windows={data['train_window_count']}, "
        f"validation_windows={data['validation_window_count']}, "
        f"models_dir={configuration['models_dir']}, "
        f"metrics_dir={configuration['metrics_dir']}"
    )


if __name__ == "__main__":
    main()
