from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from industrial_maintenance_mlops.api.app import ApiSettings, create_app, load_api_settings
from industrial_maintenance_mlops.inference.predictor import PredictionService
from industrial_maintenance_mlops.models.persistence import ModelBundle, save_model_bundle
from industrial_maintenance_mlops.monitoring.drift import (
    DriftDetector,
    ReferenceStats,
    save_reference_stats,
)
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
    assert response.json()["ready"] is True
    assert response.json()["models"]["rul_model_loaded"] is True


def test_rul_prediction_endpoint_with_fake_model():
    client = build_test_client()
    payload = {"sensor_window": [[1.0, 2.0], [1.5, 2.5]], "metadata": {"unit": 1}}

    response = client.post("/predict/rul", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-rul"
    assert body["predicted_rul"] == 42.0
    assert body["input_metadata"]["unit"] == 1
    assert "drift" in body["input_metadata"]


def test_failure_risk_prediction_endpoint_with_fake_model():
    client = build_test_client()
    payload = {"sensor_window": [[1.0, 2.0], [1.5, 2.5]]}

    response = client.post("/predict/failure-risk", json=payload)

    assert response.status_code == 200
    assert response.json()["failure_risk_probability"] == 0.75
    assert response.json()["predicted_high_risk"] is True


def test_batch_prediction_endpoint_with_fake_model():
    client = build_test_client()
    payload = {
        "model_type": "rul",
        "sensor_windows": [
            [[1.0, 2.0], [1.5, 2.5]],
            [[2.0, 3.0], [2.5, 3.5]],
        ],
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "rul"
    assert body["predictions"] == [42.0, 42.0]


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


def test_health_works_without_model_files(tmp_path):
    app = create_app(
        settings=ApiSettings(
            rul_model_path=tmp_path / "missing_rul.joblib",
            failure_risk_model_path=tmp_path / "missing_risk.joblib",
            reference_stats_path=tmp_path / "missing_stats.json",
            window_size=2,
            feature_count=2,
        ),
        metrics=PrometheusMetrics(),
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["models"]["rul_model_loaded"] is False
    assert response.json()["models"]["failure_risk_model_loaded"] is False


def test_prediction_returns_clear_error_without_model_files(tmp_path):
    app = create_app(
        settings=ApiSettings(
            rul_model_path=tmp_path / "missing_rul.joblib",
            failure_risk_model_path=tmp_path / "missing_risk.joblib",
            window_size=2,
            feature_count=2,
        ),
        metrics=PrometheusMetrics(),
    )
    client = TestClient(app)

    response = client.post("/predict/rul", json={"sensor_window": [[1.0, 2.0], [3.0, 4.0]]})

    assert response.status_code == 503
    assert "artifact is unavailable" in response.json()["detail"]


def test_api_loads_temporary_joblib_models(tmp_path):
    feature_columns = ["sensor_1", "sensor_2"]
    rul_path = tmp_path / "rul_regressor.joblib"
    risk_path = tmp_path / "failure_risk_classifier.joblib"
    stats_path = tmp_path / "reference_stats.json"
    X = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )
    y_rul = np.array([80.0, 60.0, 20.0, 5.0])
    y_risk = np.array([0, 0, 1, 1])
    regressor = LinearRegression().fit(X, y_rul)
    classifier = RandomForestClassifier(n_estimators=2, random_state=42).fit(X, y_risk)

    save_model_bundle(
        ModelBundle(
            model=regressor,
            version="temp-rul",
            model_type="rul_regression",
            feature_columns=feature_columns,
            window_size=2,
        ),
        rul_path,
    )
    save_model_bundle(
        ModelBundle(
            model=classifier,
            version="temp-risk",
            model_type="failure_risk_classification",
            feature_columns=feature_columns,
            window_size=2,
        ),
        risk_path,
    )
    save_reference_stats(
        ReferenceStats(
            mean=[1.0, 1.0],
            std=[1.0, 1.0],
            feature_columns=feature_columns,
            window_size=2,
        ),
        stats_path,
    )

    app = create_app(
        settings=ApiSettings(
            rul_model_path=rul_path,
            failure_risk_model_path=risk_path,
            reference_stats_path=stats_path,
            window_size=2,
            feature_count=2,
        ),
        metrics=PrometheusMetrics(),
    )
    client = TestClient(app)

    health = client.get("/health").json()
    rul_response = client.post("/predict/rul", json={"sensor_window": [[1.0, 1.0], [2.0, 2.0]]})
    risk_response = client.post(
        "/predict/failure-risk",
        json={"sensor_window": [[1.0, 1.0], [2.0, 2.0]]},
    )
    batch_response = client.post(
        "/predict/batch",
        json={
            "model_type": "failure-risk",
            "sensor_windows": [
                [[1.0, 1.0], [2.0, 2.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
        },
    )

    assert health["ready"] is True
    assert health["reference_stats_loaded"] is True
    assert rul_response.status_code == 200
    assert isinstance(rul_response.json()["predicted_rul"], float)
    assert risk_response.status_code == 200
    assert isinstance(risk_response.json()["failure_risk_probability"], float)
    assert isinstance(risk_response.json()["predicted_high_risk"], bool)
    assert batch_response.status_code == 200
    assert len(batch_response.json()["predictions"]) == 2


def test_model_paths_can_be_configured(tmp_path, monkeypatch):
    config_path = tmp_path / "api.yaml"
    rul_path = tmp_path / "configured_rul.joblib"
    risk_path = tmp_path / "configured_risk.joblib"
    stats_path = tmp_path / "configured_stats.json"
    config_path.write_text(
        "\n".join(
            [
                f"rul_model_path: '{rul_path}'",
                f"failure_risk_model_path: '{risk_path}'",
                f"reference_stats_path: '{stats_path}'",
                "window_size: 2",
                "feature_count: 2",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("API_CONFIG_PATH", str(config_path))

    settings = load_api_settings()

    assert settings.rul_model_path == rul_path
    assert settings.failure_risk_model_path == risk_path
    assert settings.reference_stats_path == stats_path
    assert settings.window_size == 2
    assert settings.feature_count == 2
