"""Unified regression models from scratch and sklearn checkers for the lab."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ArrayLike = np.ndarray | pd.DataFrame | pd.Series


# =====================================================================
# UTILITIES AND HELPERS
# =====================================================================

def to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "toarray"):
        return data.toarray()
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    return np.asarray(data)


def to_numpy_2d(x: ArrayLike) -> np.ndarray:
    arr = to_numpy(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(float)


def to_numpy_1d(y: ArrayLike) -> np.ndarray:
    arr = to_numpy(y)
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


# =====================================================================
# LINEAR REGRESSION SCRATCH
# =====================================================================

class LinearRegressionScratch(BaseEstimator, RegressorMixin):
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


# =====================================================================
# RIDGE REGRESSION SCRATCH
# =====================================================================

class RidgeRegressionScratch(BaseEstimator, RegressorMixin):
    """Ridge Regression using closed-form L2 regularization."""

    def __init__(self, alpha: float = 10.0, fit_intercept: bool = True) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.weights_: np.ndarray | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> "RidgeRegressionScratch":
        x_np = to_numpy_2d(x).astype(np.float32)
        y_np = to_numpy_1d(y).astype(np.float32).reshape(-1, 1)
        n_samples, n_features = x_np.shape

        if self.fit_intercept:
            x_design = np.c_[np.ones(n_samples, dtype=np.float32), x_np]
            penalty = self.alpha * np.eye(n_features + 1, dtype=np.float32)
            penalty[0, 0] = 0.0  # Do not penalize the intercept
        else:
            x_design = x_np
            penalty = self.alpha * np.eye(n_features, dtype=np.float32)

        A = x_design.T @ x_design + penalty
        weights = np.linalg.pinv(A) @ x_design.T @ y_np
        self.weights_ = weights

        if self.fit_intercept:
            self.intercept_ = float(weights[0, 0])
            self.coef_ = weights[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.flatten()

        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before predict().")
        x_np = to_numpy_2d(x).astype(np.float32)
        return x_np @ self.coef_ + self.intercept_


# =====================================================================
# LINEAR SVR SCRATCH
# =====================================================================

class LinearSVRRegressorScratch(BaseEstimator, RegressorMixin):
    """Linear SVR trained with epsilon-insensitive sub-gradient descent."""

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


# =====================================================================
# MLP REGRESSOR SCRATCH
# =====================================================================

class MLPRegressorScratch(BaseEstimator, RegressorMixin):
    """Neural network with 1 hidden layer trained with SGD.
    Structure: Input -> Hidden (ReLU) -> Output (Linear)
    """

    def __init__(self, hidden_dim: int = 16, lr: float = 0.01, epochs: int = 20, batch_size: int = 1024, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.loss_history_: list[float] = []
        self.is_fitted_ = False

    def _relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, Z)

    def _relu_deriv(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0.0).astype(np.float32)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPRegressorScratch":
        X_np = to_numpy(X).astype(np.float32)
        y_np = to_numpy(y).astype(np.float32).reshape(-1, 1)

        n_samples, n_features = X_np.shape
        rng = np.random.default_rng(self.random_state)

        # He Initialization
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_features), size=(n_features, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden_dim), size=(self.hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)
        self.loss_history_ = []

        for _ in range(self.epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X_np[indices]
            y_shuffled = y_np[indices]

            epoch_loss = 0.0
            num_batches = int(np.ceil(n_samples / self.batch_size))

            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n_samples)
                batch_len = end - start

                if batch_len == 0:
                    continue

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward Pass
                Z1 = X_batch @ self.W1 + self.b1
                A1 = self._relu(Z1)
                y_pred = A1 @ self.W2 + self.b2

                loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss * batch_len

                # Backward Pass
                dZ2 = y_pred - y_batch
                dW2 = (A1.T @ dZ2) / batch_len
                db2 = np.mean(dZ2, axis=0, keepdims=True)

                dA1 = dZ2 @ self.W2.T
                dZ1 = dA1 * self._relu_deriv(Z1)
                dW1 = (X_batch.T @ dZ1) / batch_len
                db1 = np.mean(dZ1, axis=0, keepdims=True)

                # SGD update
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

            self.loss_history_.append(epoch_loss / n_samples)

        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict().")
        X_np = to_numpy(X).astype(np.float32)
        Z1 = X_np @ self.W1 + self.b1
        A1 = self._relu(Z1)
        pred = A1 @ self.W2 + self.b2
        return pred.flatten()


# =====================================================================
# HISTOGRAM GRADIENT BOOSTING SCRATCH
# =====================================================================

class HistGradientBoostingRegressorScratch(BaseEstimator, RegressorMixin):
    """Histogram-based Gradient Boosting Regressor built from scratch."""

    def __init__(self, max_iter: int = 160, learning_rate: float = 0.04, max_leaf_nodes: int = 31, l2_regularization: float = 0.1, random_state: int = 42):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.estimators_: list[DecisionTreeRegressor] = []
        self.init_value_ = 0.0
        self.bin_thresholds_: list[np.ndarray] = []
        self.is_fitted_ = False

    def fit(self, X: ArrayLike, y: ArrayLike) -> "HistGradientBoostingRegressorScratch":
        X_np = to_numpy(X).astype(np.float32)
        y_np = to_numpy(y).astype(np.float32)
        n_samples, n_features = X_np.shape

        # 1. Feature Binning (Percentile-based)
        self.bin_thresholds_ = []
        X_binned = np.zeros_like(X_np, dtype=np.float32)

        for j in range(n_features):
            col = X_np[:, j]
            thresholds = np.unique(np.percentile(col, np.linspace(0, 100, 256)))
            self.bin_thresholds_.append(thresholds)
            X_binned[:, j] = np.digitize(col, thresholds[1:-1]).astype(np.float32)

        # 2. Init F0 = mean(y)
        self.init_value_ = float(np.mean(y_np))
        F = np.full(n_samples, self.init_value_, dtype=np.float32)

        self.estimators_ = []

        # 3. Boosting loop
        for m in range(self.max_iter):
            residuals = y_np - F
            tree = DecisionTreeRegressor(
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_state + m,
                ccp_alpha=self.l2_regularization * 0.01
            )
            tree.fit(X_binned, residuals)
            F += self.learning_rate * tree.predict(X_binned)
            self.estimators_.append(tree)

        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict().")
        X_np = to_numpy(X).astype(np.float32)
        n_samples, n_features = X_np.shape

        # Binning test data based on thresholds learned on train
        X_binned = np.zeros_like(X_np, dtype=np.float32)
        for j in range(n_features):
            thresholds = self.bin_thresholds_[j]
            X_binned[:, j] = np.digitize(X_np[:, j], thresholds[1:-1]).astype(np.float32)

        F = np.full(n_samples, self.init_value_, dtype=np.float32)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X_binned)

        return F


# =====================================================================
# MEAN BASELINE REGRESSOR
# =====================================================================

class MeanBaselineRegressorScratch(BaseEstimator, RegressorMixin):
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


# =====================================================================
# SKLEARN MODEL CHECKERS & WRAPPERS
# =====================================================================

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


class SklearnMLPRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for MLPRegressor to check pipeline compatibility."""

    def __init__(self, hidden_dim: int = 16, lr: float = 0.01, epochs: int = 20, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim,),
            activation='relu',
            solver='sgd',
            learning_rate='constant',
            learning_rate_init=self.lr,
            max_iter=self.epochs,
            batch_size=1024,
            random_state=self.random_state
        )
        self.is_fitted_ = False

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SklearnMLPRegressorWrapper":
        X_np = to_numpy(X)
        y_np = to_numpy(y)
        self.model.fit(X_np, y_np)
        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X_np = to_numpy(X)
        return self.model.predict(X_np)


class ModelDoubleChecker:
    """Class to double check the predictions of scratch models against Scikit-Learn libraries."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results_ = {}

    def check_models(self, X_train: ArrayLike, y_train: ArrayLike, X_val: ArrayLike, y_val: ArrayLike) -> pd.DataFrame:
        X_tr = to_numpy(X_train).astype(np.float32)
        y_tr = to_numpy(y_train).astype(np.float32)
        X_v = to_numpy(X_val).astype(np.float32)
        y_v = to_numpy(y_val).astype(np.float32)

        # 1. Compare Ridge
        ridge_scratch = RidgeRegressionScratch(alpha=10.0)
        ridge_sklearn = Ridge(alpha=10.0)

        ridge_scratch.fit(X_tr, y_tr)
        ridge_sklearn.fit(X_tr, y_tr)

        p_scratch = ridge_scratch.predict(X_v)
        p_sklearn = ridge_sklearn.predict(X_v)

        self.results_["Ridge"] = {
            "Scratch_MAE": float(mean_absolute_error(y_v, p_scratch)),
            "Sklearn_MAE": float(mean_absolute_error(y_v, p_sklearn)),
            "Absolute_Diff": float(abs(mean_absolute_error(y_v, p_scratch) - mean_absolute_error(y_v, p_sklearn)))
        }

        # 2. Compare MLP
        mlp_scratch = MLPRegressorScratch(hidden_dim=16, lr=0.01, epochs=10, batch_size=1024, random_state=self.random_state)
        mlp_sklearn = MLPRegressor(hidden_layer_sizes=(16,), activation='relu', solver='sgd',
                                   learning_rate='constant', learning_rate_init=0.01, max_iter=10,
                                   batch_size=1024, random_state=self.random_state)

        mlp_scratch.fit(X_tr, y_tr)
        mlp_sklearn.fit(X_tr, y_tr)

        p_scratch = mlp_scratch.predict(X_v)
        p_sklearn = mlp_sklearn.predict(X_v)

        self.results_["MLP"] = {
            "Scratch_MAE": float(mean_absolute_error(y_v, p_scratch)),
            "Sklearn_MAE": float(mean_absolute_error(y_v, p_sklearn)),
            "Absolute_Diff": float(abs(mean_absolute_error(y_v, p_scratch) - mean_absolute_error(y_v, p_sklearn)))
        }

        # 3. Compare HistGradientBoosting
        hgb_scratch = HistGradientBoostingRegressorScratch(max_iter=30, learning_rate=0.1, random_state=self.random_state)
        hgb_sklearn = HistGradientBoostingRegressor(max_iter=30, learning_rate=0.1, random_state=self.random_state)

        hgb_scratch.fit(X_tr, y_tr)
        hgb_sklearn.fit(X_tr, y_tr)

        p_scratch = hgb_scratch.predict(X_v)
        p_sklearn = hgb_sklearn.predict(X_v)

        self.results_["HistGradientBoosting"] = {
            "Scratch_MAE": float(mean_absolute_error(y_v, p_scratch)),
            "Sklearn_MAE": float(mean_absolute_error(y_v, p_sklearn)),
            "Absolute_Diff": float(abs(mean_absolute_error(y_v, p_scratch) - mean_absolute_error(y_v, p_sklearn)))
        }

        return pd.DataFrame(self.results_).T


def get_models(one_hot_preprocessor: Any, ordinal_preprocessor: Any, random_state: int = 42) -> dict[str, Pipeline]:
    """Returns a dict of pipelines for notebook evaluations."""
    return {
        "Ridge_Scratch": Pipeline(
            steps=[("preprocess", one_hot_preprocessor), ("model", RidgeRegressionScratch(alpha=10.0))]
        ),
        "MLP_Scratch": Pipeline(
            steps=[
                ("preprocess", one_hot_preprocessor),
                ("model", MLPRegressorScratch(hidden_dim=16, lr=0.01, epochs=20, batch_size=1024, random_state=random_state))
            ]
        ),
        "HistGradientBoosting_Scratch_160": Pipeline(
            steps=[
                ("preprocess", ordinal_preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressorScratch(
                        max_iter=160,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
