"""Model evaluation helpers."""

from __future__ import annotations

import numpy as np
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
    n_resamples: int = 10_000,
) -> tuple[float, float]:
    """Compute a percentile bootstrap confidence interval for RMSE.

    This avoids relying on scipy.stats.bootstrap, which can break across some
    SciPy/NumPy version combinations.
    """
    squared_errors = np.asarray((predictions - y_true) ** 2, dtype=float)
    rng = np.random.default_rng(random_state)
    bootstrap_indices = rng.integers(
        0,
        len(squared_errors),
        size=(n_resamples, len(squared_errors)),
    )
    bootstrap_rmses = np.sqrt(np.mean(squared_errors[bootstrap_indices], axis=1))
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(bootstrap_rmses, [alpha, 1 - alpha])
    return float(lower), float(upper)
