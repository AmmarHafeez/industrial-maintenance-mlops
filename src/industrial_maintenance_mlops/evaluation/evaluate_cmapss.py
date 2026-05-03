from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_maintenance_mlops.data.parser import read_cmapss_file, read_rul_file
from industrial_maintenance_mlops.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
)
from industrial_maintenance_mlops.features.windowing import (
    build_failure_risk_targets,
    flatten_windows,
)
from industrial_maintenance_mlops.models.persistence import ModelBundle, load_model_bundle
from industrial_maintenance_mlops.utils.io import write_json
from industrial_maintenance_mlops.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationConfig:
    dataset_id: str = "FD001"
    raw_dir: Path = Path("data/raw/CMAPSSData")
    models_dir: Path = Path("models")
    metrics_dir: Path = Path("reports/metrics")
    window_size: int = 30
    max_rul: int = 125
    risk_threshold: int = 30

    @property
    def test_file(self) -> Path:
        return self.raw_dir / f"test_{self.dataset_id}.txt"

    @property
    def rul_file(self) -> Path:
        return self.raw_dir / f"RUL_{self.dataset_id}.txt"

    @property
    def rul_model_path(self) -> Path:
        return self.models_dir / "rul_regressor.joblib"

    @property
    def failure_risk_model_path(self) -> Path:
        return self.models_dir / "failure_risk_classifier.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / f"test_metrics_{self.dataset_id}.json"


@dataclass(frozen=True)
class FinalWindowDataset:
    X: np.ndarray
    y_rul: np.ndarray
    engine_ids: list[int]
    feature_columns: list[str]


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    configure_logging()
    validate_evaluation_config(config)
    validate_required_files(config)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model bundles from %s", config.models_dir)
    rul_bundle = load_model_bundle(config.rul_model_path)
    risk_bundle = load_model_bundle(config.failure_risk_model_path)
    validate_model_bundles(rul_bundle, risk_bundle, config)

    LOGGER.info("Loading C-MAPSS test data from %s", config.test_file)
    test_frame = read_cmapss_file(config.test_file)
    final_rul = read_rul_file(config.rul_file)
    dataset = extract_final_windows(
        test_frame=test_frame,
        final_rul=final_rul,
        feature_columns=rul_bundle.feature_columns,
        window_size=config.window_size,
        max_rul=config.max_rul,
    )
    flat_windows = flatten_windows(dataset.X)

    LOGGER.info("Scoring %s final test-engine windows", len(dataset.engine_ids))
    rul_predictions = rul_bundle.model.predict(flat_windows)
    risk_predictions = risk_bundle.model.predict(flat_windows)
    true_risk = build_failure_risk_targets(dataset.y_rul, threshold=config.risk_threshold)

    report = build_metrics_report(
        config=config,
        model_version=resolve_model_version(rul_bundle, risk_bundle),
        dataset=dataset,
        regression_report=regression_metrics(dataset.y_rul, rul_predictions),
        classification_report=classification_metrics(true_risk, risk_predictions),
    )
    write_json(report, config.metrics_path)
    LOGGER.info("Test-set metrics written to %s", config.metrics_path)
    return report


def validate_evaluation_config(config: EvaluationConfig) -> None:
    if not config.dataset_id:
        raise ValueError("dataset_id must not be empty")
    if config.window_size <= 0:
        raise ValueError("window_size must be positive")
    if config.max_rul <= 0:
        raise ValueError("max_rul must be positive")
    if config.risk_threshold < 0:
        raise ValueError("risk_threshold must be non-negative")


def validate_required_files(config: EvaluationConfig) -> None:
    required_paths = [
        config.test_file,
        config.rul_file,
        config.rul_model_path,
        config.failure_risk_model_path,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required evaluation files: {names}")


def validate_model_bundles(
    rul_bundle: ModelBundle,
    risk_bundle: ModelBundle,
    config: EvaluationConfig,
) -> None:
    if rul_bundle.window_size != config.window_size:
        raise ValueError(
            f"RUL model window_size {rul_bundle.window_size} does not match {config.window_size}"
        )
    if risk_bundle.window_size != config.window_size:
        raise ValueError(
            "Failure-risk model window_size "
            f"{risk_bundle.window_size} does not match {config.window_size}"
        )
    if rul_bundle.feature_columns != risk_bundle.feature_columns:
        raise ValueError("RUL and failure-risk model feature columns do not match")


def extract_final_windows(
    test_frame: pd.DataFrame,
    final_rul: pd.Series,
    feature_columns: list[str],
    window_size: int,
    max_rul: int,
) -> FinalWindowDataset:
    required_columns = {"unit_number", "time_in_cycles", *feature_columns}
    missing_columns = required_columns.difference(test_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required test columns: {sorted(missing_columns)}")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    engine_ids: list[int] = []
    final_rul_by_unit = final_rul.astype(float)

    for unit_number, group in test_frame.sort_values(["unit_number", "time_in_cycles"]).groupby(
        "unit_number"
    ):
        if unit_number not in final_rul_by_unit.index:
            raise ValueError(f"Missing final RUL for test engine {unit_number}")
        ordered = group.sort_values("time_in_cycles")
        if len(ordered) < window_size:
            raise ValueError(
                f"Test engine {unit_number} has {len(ordered)} cycles, "
                f"but window_size is {window_size}"
            )
        final_window = ordered.tail(window_size)[feature_columns].to_numpy(dtype=float)
        windows.append(final_window)
        targets.append(float(min(final_rul_by_unit.loc[unit_number], max_rul)))
        engine_ids.append(int(unit_number))

    if not windows:
        raise ValueError("No test-engine windows were extracted")

    return FinalWindowDataset(
        X=np.stack(windows).astype(float),
        y_rul=np.asarray(targets, dtype=float),
        engine_ids=engine_ids,
        feature_columns=feature_columns,
    )


def resolve_model_version(rul_bundle: ModelBundle, risk_bundle: ModelBundle) -> str | dict[str, str]:
    if rul_bundle.version == risk_bundle.version:
        return rul_bundle.version
    return {"rul": rul_bundle.version, "failure_risk": risk_bundle.version}


def build_metrics_report(
    config: EvaluationConfig,
    model_version: str | dict[str, str],
    dataset: FinalWindowDataset,
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
            "test_file": config.test_file.name,
            "rul_file": config.rul_file.name,
            "engine_count": len(dataset.engine_ids),
            "window_count": int(dataset.X.shape[0]),
            "window_size": int(dataset.X.shape[1]),
            "feature_count": int(dataset.X.shape[2]),
            "engine_ids": dataset.engine_ids,
        },
        "regression": regression_report,
        "classification": classification_report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate C-MAPSS models on held-out test data.")
    parser.add_argument("--dataset-id", default="FD001", help="C-MAPSS dataset id such as FD001.")
    parser.add_argument("--raw-dir", default="data/raw/CMAPSSData", help="Raw C-MAPSS data directory.")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained model bundles.")
    parser.add_argument("--metrics-dir", default="reports/metrics", help="Directory for metrics JSON.")
    parser.add_argument("--window-size", type=int, default=30, help="Number of cycles per final window.")
    parser.add_argument("--max-rul", type=int, default=125, help="Maximum clipped RUL target.")
    parser.add_argument(
        "--risk-threshold",
        type=int,
        default=30,
        help="RUL threshold for high-risk labels.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_evaluation(
        EvaluationConfig(
            dataset_id=args.dataset_id,
            raw_dir=Path(args.raw_dir),
            models_dir=Path(args.models_dir),
            metrics_dir=Path(args.metrics_dir),
            window_size=args.window_size,
            max_rul=args.max_rul,
            risk_threshold=args.risk_threshold,
        )
    )
    print(format_completion_summary(report))


def format_completion_summary(report: dict[str, Any]) -> str:
    metrics_file = (
        Path(report["configuration"]["metrics_dir"])
        / f"test_metrics_{report['dataset_id']}.json"
    )
    return (
        "Evaluation complete: "
        f"dataset={report['dataset_id']}, "
        f"engines={report['data']['engine_count']}, "
        f"metrics_path={metrics_file}"
    )


if __name__ == "__main__":
    main()
