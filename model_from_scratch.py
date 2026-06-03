"""Regression models from scratch plus sklearn checker.

Each model has its own class with a small sklearn-like API:
fit(X, y) and predict(X).

The checker class trains one sklearn model on the same data and compares
metrics, which is useful for validating the scratch implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


ArrayLike = np.ndarray | pd.DataFrame | pd.Series


class LinearRegressionScratch:
    """Linear Regression using the normal equation.

    Formula:
        beta = pinv(X^T X) X^T y
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.weights_: np.ndarray | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> "LinearRegressionScratch":
        x_np = to_numpy_2d(x)
        y_np = to_numpy_1d(y)
        x_design = add_intercept(x_np) if self.fit_intercept else x_np
        weights = np.linalg.pinv(x_design.T @ x_design) @ x_design.T @ y_np
        self.weights_ = weights
        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before predict().")
        x_np = to_numpy_2d(x)
        return x_np @ self.coef_ + self.intercept_


class RidgeRegressionScratch:
    """Ridge Regression using closed-form L2 regularization."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.weights_: np.ndarray | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> "RidgeRegressionScratch":
        x_np = to_numpy_2d(x)
        y_np = to_numpy_1d(y)
        x_design = add_intercept(x_np) if self.fit_intercept else x_np
        penalty = self.alpha * np.eye(x_design.shape[1])
        if self.fit_intercept:
            penalty[0, 0] = 0.0
        weights = np.linalg.pinv(x_design.T @ x_design + penalty) @ x_design.T @ y_np
        self.weights_ = weights
        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before predict().")
        x_np = to_numpy_2d(x)
        return x_np @ self.coef_ + self.intercept_


class LinearSVRRegressorScratch:
    """Linear SVR trained with epsilon-insensitive sub-gradient descent.

    This is intentionally educational and lightweight. It is suitable for
    numeric feature matrices after scaling, not for huge sparse one-hot matrices.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        learning_rate: float = 0.01,
        epochs: int = 300,
        l2: float = 0.001,
        fit_intercept: bool = True,
        random_state: int = 42,
    ) -> None:
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, x: ArrayLike, y: ArrayLike) -> "LinearSVRRegressorScratch":
        x_np = to_numpy_2d(x)
        y_np = to_numpy_1d(y)
        rng = np.random.default_rng(self.random_state)
        weights = rng.normal(0, 0.01, size=x_np.shape[1])
        intercept = 0.0
        n_samples = x_np.shape[0]

        for _ in range(self.epochs):
            pred = x_np @ weights + intercept
            err = pred - y_np
            outside = np.abs(err) > self.epsilon
            signed_grad = np.sign(err[outside])

            grad_w = self.l2 * weights
            grad_b = 0.0
            if outside.any():
                grad_w += (x_np[outside].T @ signed_grad) / n_samples
                grad_b = float(signed_grad.mean()) if self.fit_intercept else 0.0

            weights -= self.learning_rate * grad_w
            intercept -= self.learning_rate * grad_b

            epsilon_loss = np.maximum(0.0, np.abs(err) - self.epsilon).mean()
            reg_loss = 0.5 * self.l2 * float(weights @ weights)
            self.loss_history_.append(float(epsilon_loss + reg_loss))

        self.coef_ = weights
        self.intercept_ = float(intercept if self.fit_intercept else 0.0)
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before predict().")
        x_np = to_numpy_2d(x)
        return x_np @ self.coef_ + self.intercept_


@dataclass
class ModelCheckResult:
    scratch_metrics: dict[str, float]
    sklearn_metrics: dict[str, float]
    metric_diff: dict[str, float]


class SklearnModelChecker:
    """Train one sklearn model and compare it with one scratch model."""

    def __init__(self, sklearn_model: Any | None = None, scale: bool = True) -> None:
        self.sklearn_model = sklearn_model if sklearn_model is not None else LinearRegression()
        self.scale = scale
        self.pipeline_: Pipeline | Any | None = None

    def fit_predict(self, x_train: ArrayLike, y_train: ArrayLike, x_test: ArrayLike) -> np.ndarray:
        if self.scale:
            self.pipeline_ = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", self.sklearn_model),
                ]
            )
        else:
            self.pipeline_ = self.sklearn_model
        self.pipeline_.fit(to_numpy_2d(x_train), to_numpy_1d(y_train))
        return self.pipeline_.predict(to_numpy_2d(x_test))

    def compare(
        self,
        scratch_model: Any,
        x_train: ArrayLike,
        y_train: ArrayLike,
        x_test: ArrayLike,
        y_test: ArrayLike,
    ) -> ModelCheckResult:
        scratch_model.fit(x_train, y_train)
        scratch_pred = scratch_model.predict(x_test)
        sklearn_pred = self.fit_predict(x_train, y_train, x_test)
        scratch_metrics = regression_metrics(y_test, scratch_pred)
        sklearn_metrics = regression_metrics(y_test, sklearn_pred)
        metric_diff = {
            key: float(scratch_metrics[key] - sklearn_metrics[key])
            for key in scratch_metrics
        }
        return ModelCheckResult(
            scratch_metrics=scratch_metrics,
            sklearn_metrics=sklearn_metrics,
            metric_diff=metric_diff,
        )


def default_sklearn_ridge_checker(alpha: float = 1.0, scale: bool = True) -> SklearnModelChecker:
    return SklearnModelChecker(sklearn_model=Ridge(alpha=alpha), scale=scale)


def default_sklearn_linear_checker(scale: bool = True) -> SklearnModelChecker:
    return SklearnModelChecker(sklearn_model=LinearRegression(), scale=scale)


def default_sklearn_linear_svr_checker(
    epsilon: float = 0.1,
    c: float = 1.0,
    max_iter: int = 5000,
    scale: bool = True,
) -> SklearnModelChecker:
    return SklearnModelChecker(
        sklearn_model=LinearSVR(
            epsilon=epsilon,
            C=c,
            max_iter=max_iter,
            random_state=42,
        ),
        scale=scale,
    )


def to_numpy_2d(x: ArrayLike) -> np.ndarray:
    arr = x.to_numpy() if isinstance(x, (pd.DataFrame, pd.Series)) else np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(float)


def to_numpy_1d(y: ArrayLike) -> np.ndarray:
    arr = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
    return arr.reshape(-1).astype(float)


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(x.shape[0]), x]


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    y_true_np = to_numpy_1d(y_true)
    y_pred_np = to_numpy_1d(y_pred)
    mse = mean_squared_error(y_true_np, y_pred_np)
    return {
        "MAE": float(mean_absolute_error(y_true_np, y_pred_np)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true_np, y_pred_np)),
    }


class MeanBaselineRegressorScratch:
    """Baseline regressor that always predicts the train target mean."""

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> "MeanBaselineRegressorScratch":
        del x
        self.mean_ = float(np.mean(to_numpy_1d(y)))
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Model must be fitted before predict().")
        x_np = to_numpy_2d(x)
        return np.full(x_np.shape[0], self.mean_, dtype=float)
