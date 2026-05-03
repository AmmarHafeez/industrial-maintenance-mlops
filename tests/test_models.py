from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from industrial_maintenance_mlops.models.baselines import (
    train_failure_classifier,
    train_random_forest_regressor,
)
from industrial_maintenance_mlops.models.persistence import (
    ModelBundle,
    load_model_bundle,
    save_model_bundle,
)


def test_model_training_smoke():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    y_rul = np.array([80.0, 60.0, 20.0, 5.0])
    y_risk = np.array([0, 0, 1, 1])

    regressor = train_random_forest_regressor(X, y_rul, n_estimators=2, random_state=42)
    classifier = train_failure_classifier(
        X,
        y_risk,
        estimator="random_forest",
        n_estimators=2,
        random_state=42,
    )

    assert regressor.predict(X[:1]).shape == (1,)
    assert classifier.predict(X[:1]).shape == (1,)


def test_logistic_regression_classifier_uses_scaling_pipeline(tmp_path):
    X = np.array(
        [
            [0.0, 100.0],
            [1.0, 110.0],
            [2.0, 900.0],
            [3.0, 950.0],
        ]
    )
    y_risk = np.array([0, 0, 1, 1])

    classifier = train_failure_classifier(
        X,
        y_risk,
        estimator="logistic_regression",
        max_iter=2000,
        random_state=42,
    )

    assert isinstance(classifier, Pipeline)
    assert isinstance(classifier.named_steps["scaler"], StandardScaler)
    assert "classifier" in classifier.named_steps

    model_path = tmp_path / "failure_risk_classifier.joblib"
    save_model_bundle(
        ModelBundle(
            model=classifier,
            version="test-scaled-logistic-regression",
            model_type="failure_risk_classification",
            feature_columns=["sensor_1", "sensor_2"],
            window_size=1,
        ),
        model_path,
    )
    loaded = load_model_bundle(model_path)

    assert isinstance(loaded.model, Pipeline)
    assert isinstance(loaded.model.named_steps["scaler"], StandardScaler)
