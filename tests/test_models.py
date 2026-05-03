from __future__ import annotations

import numpy as np

from industrial_maintenance_mlops.models.baselines import (
    train_failure_classifier,
    train_random_forest_regressor,
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
