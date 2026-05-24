"""Model evaluation helpers."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import root_mean_squared_error


def evaluate_rmse(model, X, y) -> float:
    """Predict and compute RMSE."""
    predictions = model.predict(X)
    return root_mean_squared_error(y, predictions)


def rmse(squared_errors):
    """RMSE statistic for bootstrap confidence intervals."""
    return np.sqrt(np.mean(squared_errors))


def rmse_confidence_interval(
    predictions,
    y_true,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for RMSE."""
    squared_errors = (predictions - y_true) ** 2
    boot_result = stats.bootstrap(
        [squared_errors],
        rmse,
        confidence_level=confidence,
        random_state=random_state,
    )
    interval = boot_result.confidence_interval
    return interval.low, interval.high
