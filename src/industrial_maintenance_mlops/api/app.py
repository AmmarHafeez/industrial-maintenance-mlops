from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from industrial_maintenance_mlops.api.schemas import (
    BatchPredictionResponse,
    BatchPredictionRequest,
    FailureRiskPredictionResponse,
    PredictionRequest,
    RulPredictionResponse,
)
from industrial_maintenance_mlops.inference.predictor import (
    InvalidWindowShapeError,
    ModelNotLoadedError,
    PredictionService,
)
from industrial_maintenance_mlops.monitoring.drift import (
    DriftDetector,
    load_reference_stats,
)
from industrial_maintenance_mlops.monitoring.metrics import PrometheusMetrics
from industrial_maintenance_mlops.utils.config import load_yaml
from industrial_maintenance_mlops.utils.logging import configure_logging

configure_logging()


@dataclass(frozen=True)
class ApiSettings:
    model_version: str = "local"
    rul_model_path: Path = Path("models/rul_regressor.joblib")
    failure_risk_model_path: Path = Path("models/failure_risk_classifier.joblib")
    reference_stats_path: Path = Path("models/reference_stats.json")
    window_size: int = 30
    feature_count: int = 24
    drift_z_threshold: float = 3.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ApiSettings":
        return cls(
            model_version=str(values.get("model_version", cls.model_version)),
            rul_model_path=Path(values.get("rul_model_path", cls.rul_model_path)),
            failure_risk_model_path=Path(
                values.get("failure_risk_model_path", cls.failure_risk_model_path)
            ),
            reference_stats_path=Path(
                values.get("reference_stats_path", cls.reference_stats_path)
            ),
            window_size=int(values.get("window_size", cls.window_size)),
            feature_count=int(values.get("feature_count", cls.feature_count)),
            drift_z_threshold=float(values.get("drift_z_threshold", cls.drift_z_threshold)),
        )

    def with_env_overrides(self) -> "ApiSettings":
        return ApiSettings(
            model_version=os.environ.get("API_MODEL_VERSION", self.model_version),
            rul_model_path=Path(os.environ.get("RUL_MODEL_PATH", str(self.rul_model_path))),
            failure_risk_model_path=Path(
                os.environ.get("FAILURE_RISK_MODEL_PATH", str(self.failure_risk_model_path))
            ),
            reference_stats_path=Path(
                os.environ.get("REFERENCE_STATS_PATH", str(self.reference_stats_path))
            ),
            window_size=int(os.environ.get("API_WINDOW_SIZE", self.window_size)),
            feature_count=int(os.environ.get("API_FEATURE_COUNT", self.feature_count)),
            drift_z_threshold=float(os.environ.get("DRIFT_Z_THRESHOLD", self.drift_z_threshold)),
        )


def create_app(
    settings: ApiSettings | None = None,
    prediction_service: PredictionService | None = None,
    drift_detector: DriftDetector | None = None,
    metrics: PrometheusMetrics | None = None,
) -> FastAPI:
    settings = settings or load_api_settings()
    service = prediction_service or PredictionService.from_paths(
        rul_model_path=settings.rul_model_path,
        failure_model_path=settings.failure_risk_model_path,
        default_window_size=settings.window_size,
        default_feature_count=settings.feature_count,
        model_version=settings.model_version,
    )
    detector = drift_detector if drift_detector is not None else _load_drift_detector(settings)
    app_metrics = metrics or PrometheusMetrics()

    application = FastAPI(
        title="industrial-maintenance-mlops",
        version="0.1.0",
        description="Predictive maintenance inference API for C-MAPSS turbofan windows.",
    )
    application.state.prediction_service = service
    application.state.drift_detector = detector
    application.state.metrics = app_metrics
    application.state.settings = settings

    @application.middleware("http")
    async def record_http_requests(request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        app_metrics.record_request(request.method, request.url.path, response.status_code)
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        app_metrics.record_validation_error(request.url.path)
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @application.get("/health")
    def health() -> dict[str, object]:
        model_health = service.health()
        return {
            "status": "ok",
            "ready": bool(model_health["ready"]),
            "models": model_health,
            "reference_stats_loaded": detector.reference_stats is not None,
            "artifact_paths": {
                "rul_model_path": str(settings.rul_model_path),
                "failure_risk_model_path": str(settings.failure_risk_model_path),
                "reference_stats_path": str(settings.reference_stats_path),
            },
        }

    @application.post("/predict/rul", response_model=RulPredictionResponse)
    def predict_rul(payload: PredictionRequest) -> RulPredictionResponse:
        started = time.perf_counter()
        try:
            prediction = service.predict_rul(payload.sensor_window)
        except InvalidWindowShapeError as exc:
            app_metrics.record_validation_error("/predict/rul")
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelNotLoadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        elapsed = time.perf_counter() - started
        app_metrics.record_prediction("rul")
        app_metrics.record_latency("/predict/rul", elapsed)
        return RulPredictionResponse(
            model_version=service.model_version_for("rul"),
            predicted_rul=prediction,
            latency_ms=elapsed * 1000,
            input_metadata=_metadata_with_drift(payload.metadata, payload.sensor_window, detector),
        )

    @application.post("/predict/failure-risk", response_model=FailureRiskPredictionResponse)
    def predict_failure_risk(payload: PredictionRequest) -> FailureRiskPredictionResponse:
        started = time.perf_counter()
        try:
            prediction = service.predict_failure_risk(payload.sensor_window)
        except InvalidWindowShapeError as exc:
            app_metrics.record_validation_error("/predict/failure-risk")
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelNotLoadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        elapsed = time.perf_counter() - started
        app_metrics.record_prediction("failure-risk")
        app_metrics.record_latency("/predict/failure-risk", elapsed)
        return FailureRiskPredictionResponse(
            model_version=service.model_version_for("failure-risk"),
            failure_risk_probability=prediction["failure_risk_probability"],
            predicted_high_risk=prediction["predicted_high_risk"],
            latency_ms=elapsed * 1000,
            input_metadata=_metadata_with_drift(payload.metadata, payload.sensor_window, detector),
        )

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
        started = time.perf_counter()
        try:
            prediction = service.predict_batch(payload.sensor_windows, payload.model_type)
        except InvalidWindowShapeError as exc:
            app_metrics.record_validation_error("/predict/batch")
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelNotLoadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        elapsed = time.perf_counter() - started
        app_metrics.record_prediction(payload.model_type)
        app_metrics.record_latency("/predict/batch", elapsed)
        return BatchPredictionResponse(
            model_version=service.model_version_for(payload.model_type),
            model_type=payload.model_type,
            predictions=prediction,
            latency_ms=elapsed * 1000,
            input_metadata=payload.metadata,
        )

    @application.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(content=app_metrics.render(), media_type="text/plain; version=0.0.4")

    return application


def load_api_settings() -> ApiSettings:
    config_path = Path(os.environ.get("API_CONFIG_PATH", "configs/api.yaml"))
    if not config_path.exists():
        return ApiSettings().with_env_overrides()
    return ApiSettings.from_mapping(load_yaml(config_path)).with_env_overrides()


def _load_drift_detector(settings: ApiSettings) -> DriftDetector:
    reference_path = Path(settings.reference_stats_path)
    if not reference_path.exists():
        return DriftDetector(reference_stats=None, z_threshold=settings.drift_z_threshold)
    return DriftDetector(
        reference_stats=load_reference_stats(reference_path),
        z_threshold=settings.drift_z_threshold,
    )


def _metadata_with_drift(
    metadata: dict[str, Any],
    sensor_window: list[list[float]],
    detector: DriftDetector,
) -> dict[str, Any]:
    enriched = dict(metadata)
    drift_result = detector.compare(sensor_window)
    if drift_result is not None:
        enriched["drift"] = drift_result.to_dict()
    return enriched


app = create_app()
