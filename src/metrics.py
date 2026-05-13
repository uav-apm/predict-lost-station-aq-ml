"""Evaluation metrics used by the baseline station-reconstruction scaffold."""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return actual, pred


def rmse(y_true, y_pred) -> float:
    """Return Root Mean Squared Error."""
    actual, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def r2(y_true, y_pred) -> float:
    """Return coefficient of determination (R-squared)."""
    actual, pred = _as_arrays(y_true, y_pred)
    denom = float(np.sum((actual - actual.mean()) ** 2))
    if denom == 0.0:
        return 1.0 if np.allclose(actual, pred) else 0.0
    return float(1.0 - (np.sum((actual - pred) ** 2) / denom))
