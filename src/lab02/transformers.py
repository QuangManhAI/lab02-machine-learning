"""Custom transformers used by the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_array, check_is_fitted, validate_data


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Return RBF similarities to learned geographic cluster centers."""

    def __init__(self, n_clusters: int = 10, gamma: float = 1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"cluster_{i}_similarity" for i in range(self.n_clusters)]


class FeatureFromRegressor(MetaEstimatorMixin, TransformerMixin, BaseEstimator):
    """Train a regressor and expose its predictions as transformed features."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        check_array(X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        return self

    def transform(self, X):
        check_is_fitted(self)
        predictions = self.estimator_.predict(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def get_feature_names_out(self, names=None):
        check_is_fitted(self)
        n_outputs = getattr(self.estimator_, "n_outputs_", 1)
        estimator_class_name = self.estimator_.__class__.__name__
        estimator_short_name = estimator_class_name.lower().replace("_", "")
        return [
            f"{estimator_short_name}_prediction_{i}"
            for i in range(n_outputs)
        ]


class StandardScalerClone(TransformerMixin, BaseEstimator):
    """Educational reimplementation of scikit-learn's StandardScaler."""

    def __init__(self, with_mean: bool = True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return getattr(
                self,
                "feature_names_in_",
                np.array([f"x{i}" for i in range(self.n_features_in_)]),
            )

        if len(input_features) != self.n_features_in_:
            raise ValueError("Invalid number of features")
        if hasattr(self, "feature_names_in_") and not np.all(
            self.feature_names_in_ == input_features
        ):
            raise ValueError("input_features != feature_names_in_")
        return input_features
