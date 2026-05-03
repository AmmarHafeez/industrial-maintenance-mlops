from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from industrial_maintenance_mlops.features.windowing import flatten_windows
from industrial_maintenance_mlops.models.persistence import ModelBundle, load_model_bundle


class InvalidWindowShapeError(ValueError):
    """Raised when an inference window does not match model metadata."""


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction endpoint is called without a loaded model."""


class PredictionService:
    def __init__(
        self,
        rul_model: ModelBundle | None = None,
        failure_model: ModelBundle | None = None,
        default_window_size: int = 30,
        default_feature_count: int = 24,
        model_version: str = "local",
    ) -> None:
        self.rul_model = rul_model
        self.failure_model = failure_model
        self.default_window_size = default_window_size
        self.default_feature_count = default_feature_count
        self.model_version = model_version

    @classmethod
    def from_paths(
        cls,
        rul_model_path: str | Path,
        failure_model_path: str | Path,
        default_window_size: int = 30,
        default_feature_count: int = 24,
        model_version: str = "local",
    ) -> "PredictionService":
        rul_model = _load_optional_bundle(rul_model_path)
        failure_model = _load_optional_bundle(failure_model_path)
        return cls(
            rul_model=rul_model,
            failure_model=failure_model,
            default_window_size=default_window_size,
            default_feature_count=default_feature_count,
            model_version=model_version,
        )

    def health(self) -> dict[str, object]:
        return {
            "rul_model_loaded": self.rul_model is not None,
            "failure_risk_model_loaded": self.failure_model is not None,
            "default_window_size": self.default_window_size,
            "default_feature_count": self.default_feature_count,
        }

    def predict_rul(self, sensor_window: list[list[float]] | np.ndarray) -> float:
        if self.rul_model is None:
            raise ModelNotLoadedError("RUL model is not loaded")
        array = self.validate_window(sensor_window, self.rul_model)
        prediction = self.rul_model.model.predict(flatten_windows(array[np.newaxis, :, :]))
        return float(prediction[0])

    def predict_failure_risk(self, sensor_window: list[list[float]] | np.ndarray) -> dict[str, Any]:
        if self.failure_model is None:
            raise ModelNotLoadedError("Failure-risk model is not loaded")
        array = self.validate_window(sensor_window, self.failure_model)
        flat = flatten_windows(array[np.newaxis, :, :])

        if hasattr(self.failure_model.model, "predict_proba"):
            probabilities = self.failure_model.model.predict_proba(flat)[0]
            classes = list(getattr(self.failure_model.model, "classes_", range(len(probabilities))))
            positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
            probability = float(probabilities[positive_index])
            label = int(probability >= 0.5)
        else:
            label = int(self.failure_model.model.predict(flat)[0])
            probability = None

        return {"label": label, "probability": probability}

    def predict_batch(
        self,
        sensor_windows: list[list[list[float]]],
        model_type: Literal["rul", "failure-risk"] = "rul",
    ) -> list[float | dict[str, Any]]:
        if model_type == "rul":
            return [self.predict_rul(window) for window in sensor_windows]
        if model_type == "failure-risk":
            return [self.predict_failure_risk(window) for window in sensor_windows]
        raise ValueError(f"Unsupported model_type: {model_type}")

    def validate_window(
        self,
        sensor_window: list[list[float]] | np.ndarray,
        bundle: ModelBundle | None = None,
    ) -> np.ndarray:
        try:
            array = np.asarray(sensor_window, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidWindowShapeError("sensor_window must contain numeric values") from exc

        if array.ndim != 2:
            raise InvalidWindowShapeError("sensor_window must be a 2D array")
        if not np.isfinite(array).all():
            raise InvalidWindowShapeError("sensor_window must contain only finite numeric values")

        expected_window_size = bundle.window_size if bundle else self.default_window_size
        expected_feature_count = bundle.feature_count if bundle else self.default_feature_count
        if array.shape != (expected_window_size, expected_feature_count):
            raise InvalidWindowShapeError(
                "sensor_window shape must be "
                f"({expected_window_size}, {expected_feature_count}); found {array.shape}"
            )
        return array

    def model_version_for(self, model_type: Literal["rul", "failure-risk"]) -> str:
        if model_type == "rul" and self.rul_model is not None:
            return self.rul_model.version
        if model_type == "failure-risk" and self.failure_model is not None:
            return self.failure_model.version
        return self.model_version


def _load_optional_bundle(path: str | Path) -> ModelBundle | None:
    model_path = Path(path)
    if not model_path.exists():
        return None
    return load_model_bundle(model_path)
