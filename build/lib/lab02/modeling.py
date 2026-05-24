"""Model construction, training, tuning, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeRegressor

from lab02.config import MODELS_DIR, RANDOM_STATE
from lab02.preprocessing import build_preprocessing_pipeline


def build_linear_regression_pipeline():
    """Build the baseline linear regression pipeline."""
    return make_pipeline(build_preprocessing_pipeline(), LinearRegression())


def build_tree_regression_pipeline(random_state: int = RANDOM_STATE):
    """Build the decision tree regression pipeline."""
    return make_pipeline(
        build_preprocessing_pipeline(),
        DecisionTreeRegressor(random_state=random_state),
    )


def build_random_forest_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Build the full random forest pipeline."""
    return Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline(random_state=random_state)),
            ("random_forest", RandomForestRegressor(random_state=random_state)),
        ]
    )


def cross_validate_rmse(model, X, y, cv: int = 10):
    """Return positive RMSE scores from cross-validation."""
    return -cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
    )


def random_search_random_forest(
    X,
    y,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """Tune the random forest pipeline with randomized search."""
    param_distribs = {
        "preprocessing__geo__n_clusters": randint(low=3, high=50),
        "random_forest__max_features": randint(low=2, high=20),
    }
    search = RandomizedSearchCV(
        build_random_forest_pipeline(random_state=random_state),
        param_distributions=param_distribs,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
    )
    search.fit(X, y)
    return search


def save_model(model, filename: str = "california_housing_model.pkl") -> Path:
    """Persist a fitted model under the project models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(path: Union[str, Path]):
    """Load a persisted model."""
    return joblib.load(path)
