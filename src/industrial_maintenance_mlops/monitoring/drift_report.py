from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_maintenance_mlops.data.parser import read_cmapss_file
from industrial_maintenance_mlops.monitoring.drift import ReferenceStats, load_reference_stats
from industrial_maintenance_mlops.utils.io import write_json
from industrial_maintenance_mlops.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftReportConfig:
    dataset_id: str = "FD001"
    raw_dir: Path = Path("data/raw/CMAPSSData")
    reference_stats_path: Path = Path("models/reference_stats.json")
    output_path: Path = Path("reports/metrics/drift_report_FD001.json")
    window_size: int = 30
    max_rul: int = 125
    mean_z_threshold: float = 3.0
    std_ratio_threshold: float = 2.0

    @property
    def test_file(self) -> Path:
        return self.raw_dir / f"test_{self.dataset_id}.txt"


@dataclass(frozen=True)
class FinalFeatureWindows:
    X: np.ndarray
    engine_ids: list[int]
    feature_columns: list[str]


def run_drift_report(config: DriftReportConfig) -> dict[str, Any]:
    configure_logging()
    validate_config(config)
    validate_required_files(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading reference statistics from %s", config.reference_stats_path)
    reference_stats = load_reference_stats(config.reference_stats_path)
    if reference_stats.window_size != config.window_size:
        raise ValueError(
            f"Reference window_size {reference_stats.window_size} does not match {config.window_size}"
        )

    LOGGER.info("Loading C-MAPSS test data from %s", config.test_file)
    test_frame = read_cmapss_file(config.test_file)
    windows = extract_final_feature_windows(
        test_frame=test_frame,
        feature_columns=reference_stats.feature_columns,
        window_size=config.window_size,
    )
    report = build_drift_report(config, reference_stats, windows)
    write_json(report, config.output_path)
    LOGGER.info("Drift report written to %s", config.output_path)
    return report


def validate_config(config: DriftReportConfig) -> None:
    if not config.dataset_id:
        raise ValueError("dataset_id must not be empty")
    if config.window_size <= 0:
        raise ValueError("window_size must be positive")
    if config.max_rul <= 0:
        raise ValueError("max_rul must be positive")
    if config.mean_z_threshold <= 0:
        raise ValueError("mean_z_threshold must be positive")
    if config.std_ratio_threshold <= 1:
        raise ValueError("std_ratio_threshold must be greater than 1")


def validate_required_files(config: DriftReportConfig) -> None:
    missing = [path for path in [config.test_file, config.reference_stats_path] if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required drift report files: {names}")


def extract_final_feature_windows(
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> FinalFeatureWindows:
    required_columns = {"unit_number", "time_in_cycles", *feature_columns}
    missing_columns = required_columns.difference(test_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required test columns: {sorted(missing_columns)}")

    windows: list[np.ndarray] = []
    engine_ids: list[int] = []
    for unit_number, group in test_frame.sort_values(["unit_number", "time_in_cycles"]).groupby(
        "unit_number"
    ):
        ordered = group.sort_values("time_in_cycles")
        if len(ordered) < window_size:
            raise ValueError(
                f"Test engine {unit_number} has {len(ordered)} cycles, "
                f"but window_size is {window_size}"
            )
        windows.append(ordered.tail(window_size)[feature_columns].to_numpy(dtype=float))
        engine_ids.append(int(unit_number))

    if not windows:
        raise ValueError("No final test-engine windows were extracted")

    return FinalFeatureWindows(
        X=np.stack(windows).astype(float),
        engine_ids=engine_ids,
        feature_columns=feature_columns,
    )


def build_drift_report(
    config: DriftReportConfig,
    reference_stats: ReferenceStats,
    windows: FinalFeatureWindows,
) -> dict[str, Any]:
    flattened = windows.X.reshape(windows.X.shape[0] * windows.X.shape[1], windows.X.shape[2])
    current_stats = compute_current_statistics(flattened)
    per_feature = compute_feature_drift(
        reference_stats=reference_stats,
        current_stats=current_stats,
        mean_z_threshold=config.mean_z_threshold,
        std_ratio_threshold=config.std_ratio_threshold,
    )
    config_payload = asdict(config)
    config_payload["raw_dir"] = str(config.raw_dir)
    config_payload["reference_stats_path"] = str(config.reference_stats_path)
    config_payload["output_path"] = str(config.output_path)

    return {
        "dataset_id": config.dataset_id,
        "configuration": config_payload,
        "data": {
            "test_file": config.test_file.name,
            "engine_count": len(windows.engine_ids),
            "window_count": int(windows.X.shape[0]),
            "window_size": int(windows.X.shape[1]),
            "feature_count": int(windows.X.shape[2]),
            "engine_ids": windows.engine_ids,
        },
        "summary": build_summary(per_feature, window_count=int(windows.X.shape[0])),
        "per_feature": per_feature,
    }


def compute_current_statistics(flattened_windows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": flattened_windows.mean(axis=0),
        "std": flattened_windows.std(axis=0),
        "min": flattened_windows.min(axis=0),
        "max": flattened_windows.max(axis=0),
    }


def compute_feature_drift(
    reference_stats: ReferenceStats,
    current_stats: dict[str, np.ndarray],
    mean_z_threshold: float,
    std_ratio_threshold: float,
    epsilon: float = 1e-8,
) -> dict[str, dict[str, float | bool | None]]:
    reference_mean = np.asarray(reference_stats.mean, dtype=float)
    reference_std = np.asarray(reference_stats.std, dtype=float)
    current_mean = current_stats["mean"]
    current_std = current_stats["std"]
    current_min = current_stats["min"]
    current_max = current_stats["max"]

    if len(reference_stats.feature_columns) != len(current_mean):
        raise ValueError("Reference feature count does not match current window feature count")

    per_feature: dict[str, dict[str, float | bool | None]] = {}
    for index, feature in enumerate(reference_stats.feature_columns):
        mean_abs_difference = float(abs(current_mean[index] - reference_mean[index]))
        std_abs_difference = float(abs(current_std[index] - reference_std[index]))
        mean_z_shift = _safe_ratio(current_mean[index] - reference_mean[index], reference_std[index], epsilon)
        std_ratio = _safe_ratio(current_std[index], reference_std[index], epsilon)
        flagged = is_feature_flagged(
            mean_z_shift=mean_z_shift,
            std_ratio=std_ratio,
            mean_z_threshold=mean_z_threshold,
            std_ratio_threshold=std_ratio_threshold,
        )
        per_feature[feature] = {
            "reference_mean": float(reference_mean[index]),
            "reference_std": float(reference_std[index]),
            "current_mean": float(current_mean[index]),
            "current_std": float(current_std[index]),
            "current_min": float(current_min[index]),
            "current_max": float(current_max[index]),
            "mean_abs_difference": mean_abs_difference,
            "std_abs_difference": std_abs_difference,
            "mean_z_shift": mean_z_shift,
            "std_ratio": std_ratio,
            "flagged": flagged,
        }
    return per_feature


def _safe_ratio(numerator: float, denominator: float, epsilon: float) -> float | None:
    if abs(float(denominator)) < epsilon:
        return None
    return float(numerator / denominator)


def is_feature_flagged(
    mean_z_shift: float | None,
    std_ratio: float | None,
    mean_z_threshold: float,
    std_ratio_threshold: float,
) -> bool:
    mean_flagged = mean_z_shift is not None and abs(mean_z_shift) >= mean_z_threshold
    std_flagged = std_ratio is not None and (
        std_ratio >= std_ratio_threshold or std_ratio <= 1 / std_ratio_threshold
    )
    return bool(mean_flagged or std_flagged)


def build_summary(
    per_feature: dict[str, dict[str, float | bool | None]],
    window_count: int,
) -> dict[str, float | int]:
    mean_differences = np.asarray(
        [values["mean_abs_difference"] for values in per_feature.values()],
        dtype=float,
    )
    std_differences = np.asarray(
        [values["std_abs_difference"] for values in per_feature.values()],
        dtype=float,
    )
    return {
        "feature_count": int(len(per_feature)),
        "window_count": int(window_count),
        "max_mean_abs_difference": float(mean_differences.max()),
        "mean_mean_abs_difference": float(mean_differences.mean()),
        "max_std_abs_difference": float(std_differences.max()),
        "mean_std_abs_difference": float(std_differences.mean()),
        "number_of_features_flagged": int(
            sum(bool(values["flagged"]) for values in per_feature.values())
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a C-MAPSS drift report.")
    parser.add_argument("--dataset-id", default="FD001", help="C-MAPSS dataset id such as FD001.")
    parser.add_argument("--raw-dir", default="data/raw/CMAPSSData", help="Raw C-MAPSS data directory.")
    parser.add_argument(
        "--reference-stats",
        default="models/reference_stats.json",
        help="Path to training reference statistics JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output drift report JSON path.",
    )
    parser.add_argument("--window-size", type=int, default=30, help="Number of cycles per final window.")
    parser.add_argument("--max-rul", type=int, default=125, help="Included for run configuration parity.")
    parser.add_argument(
        "--mean-z-threshold",
        type=float,
        default=3.0,
        help="Absolute z-shift threshold for mean drift flags.",
    )
    parser.add_argument(
        "--std-ratio-threshold",
        type=float,
        default=2.0,
        help="Upper and reciprocal lower threshold for standard-deviation drift flags.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_drift_report(
        DriftReportConfig(
            dataset_id=args.dataset_id,
            raw_dir=Path(args.raw_dir),
            reference_stats_path=Path(args.reference_stats),
            output_path=Path(args.output or f"reports/metrics/drift_report_{args.dataset_id}.json"),
            window_size=args.window_size,
            max_rul=args.max_rul,
            mean_z_threshold=args.mean_z_threshold,
            std_ratio_threshold=args.std_ratio_threshold,
        )
    )
    print(format_completion_summary(report))


def format_completion_summary(report: dict[str, Any]) -> str:
    return (
        "Drift report complete: "
        f"dataset={report['dataset_id']}, "
        f"windows={report['data']['window_count']}, "
        f"flagged_features={report['summary']['number_of_features_flagged']}, "
        f"output={report['configuration']['output_path']}"
    )


if __name__ == "__main__":
    main()
