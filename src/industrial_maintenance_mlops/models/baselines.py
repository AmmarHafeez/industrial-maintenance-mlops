from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def train_random_forest_regressor(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a deterministic RandomForestRegressor baseline."""
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if features.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(features) != len(targets):
        raise ValueError("X and y must contain the same number of samples")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features, targets)
    return model


def train_failure_classifier(
    X: np.ndarray,
    y: np.ndarray,
    estimator: str = "logistic_regression",
    n_estimators: int = 100,
    max_depth: int | None = None,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression | RandomForestClassifier:
    """Train a baseline classifier for near-failure risk."""
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=int)
    if features.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(features) != len(targets):
        raise ValueError("X and y must contain the same number of samples")
    if len(np.unique(targets)) < 2:
        raise ValueError("Failure-risk classifier requires at least two classes")

    if estimator == "logistic_regression":
        model: LogisticRegression | RandomForestClassifier = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )
    elif estimator == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported classifier estimator: {estimator}")

    model.fit(features, targets)
    return model
