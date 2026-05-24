"""Preprocessing pipeline definitions."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from lab02.config import RANDOM_STATE
from lab02.transformers import ClusterSimilarity


def column_ratio(X):
    """Compute the ratio between the first and second columns."""
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    """Feature name callback for ratio transformers."""
    return ["ratio"]


def ratio_pipeline():
    """Build the impute-ratio-scale pipeline used for engineered ratios."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


def build_preprocessing_pipeline(
    n_geo_clusters: int = 10,
    geo_gamma: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> ColumnTransformer:
    """Build the full preprocessing pipeline from the chapter notebook."""
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    cluster_similarity = ClusterSimilarity(
        n_clusters=n_geo_clusters,
        gamma=geo_gamma,
        random_state=random_state,
    )

    return ColumnTransformer(
        [
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            (
                "log",
                log_pipeline,
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            ("geo", cluster_similarity, ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline,
    )
