from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from industrial_maintenance_mlops.api.app import ApiSettings, create_app
from industrial_maintenance_mlops.inference.predictor import PredictionService
from industrial_maintenance_mlops.models.persistence import ModelBundle
from industrial_maintenance_mlops.monitoring.drift import DriftDetector, ReferenceStats
from industrial_maintenance_mlops.monitoring.metrics import PrometheusMetrics


class FakeRegressor:
    def predict(self, X):
        return np.full(X.shape[0], 42.0)


class FakeClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.tile(np.array([[0.25, 0.75]]), (X.shape[0], 1))


def build_test_client() -> TestClient:
    feature_columns = ["sensor_1", "sensor_2"]
    service = PredictionService(
        rul_model=ModelBundle(
            model=FakeRegressor(),
            version="test-rul",
            model_type="rul_regression",
            feature_columns=feature_columns,
            window_size=2,
        ),
        failure_model=ModelBundle(
            model=FakeClassifier(),
            version="test-risk",
            model_type="failure_risk_classification",
            feature_columns=feature_columns,
            window_size=2,
        ),
        default_window_size=2,
        default_feature_count=2,
    )
    detector = DriftDetector(
        ReferenceStats(
            mean=[1.0, 2.0],
            std=[1.0, 1.0],
            feature_columns=feature_columns,
            window_size=2,
        ),
        z_threshold=3.0,
    )
    app = create_app(
        settings=ApiSettings(window_size=2, feature_count=2),
        prediction_service=service,
        drift_detector=detector,
        metrics=PrometheusMetrics(),
    )
    return TestClient(app)


def test_health_endpoint():
    client = build_test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models"]["rul_model_loaded"] is True


def test_rul_prediction_endpoint_with_fake_model():
    client = build_test_client()
    payload = {"sensor_window": [[1.0, 2.0], [1.5, 2.5]], "metadata": {"unit": 1}}

    response = client.post("/predict/rul", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-rul"
    assert body["prediction"] == 42.0
    assert body["metadata"]["unit"] == 1
    assert "drift" in body["metadata"]


def test_failure_risk_prediction_endpoint_with_fake_model():
    client = build_test_client()
    payload = {"sensor_window": [[1.0, 2.0], [1.5, 2.5]]}

    response = client.post("/predict/failure-risk", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == {"label": 1, "probability": 0.75}


def test_invalid_input_validation():
    client = build_test_client()
    payload = {"sensor_window": [[1.0, 2.0]]}

    response = client.post("/predict/rul", json=payload)

    assert response.status_code == 422
    assert "sensor_window shape" in response.json()["detail"]


def test_metrics_endpoint_returns_text():
    client = build_test_client()
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in response.text
