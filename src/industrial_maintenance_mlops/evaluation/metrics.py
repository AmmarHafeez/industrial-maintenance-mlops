from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute basic regression metrics."""
    truth = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(truth, predictions)))
    return {
        "mae": float(mean_absolute_error(truth, predictions)),
        "rmse": rmse,
        "r2": float(r2_score(truth, predictions)) if len(truth) > 1 else 0.0,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute basic binary classification metrics."""
    truth = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
    }
