"""Data download, loading, and train/test splitting helpers."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Union
from zlib import crc32

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lab02.config import DATASETS_DIR, HOUSING_URL, RANDOM_STATE, TEST_SIZE


def load_housing_data(
    data_dir: Path = DATASETS_DIR,
    source_url: str = HOUSING_URL,
) -> pd.DataFrame:
    """Download the housing archive if needed and return the housing CSV."""
    tarball_path = data_dir / "housing.tgz"
    housing_csv = data_dir / "housing" / "housing.csv"

    if not tarball_path.is_file():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(source_url, tarball_path)

    if not housing_csv.is_file():
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=data_dir)

    return pd.read_csv(housing_csv)


def add_income_category(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the data with the stratification income category."""
    data = data.copy()
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return data


def stratified_train_test_split(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split housing data using the income category for stratification."""
    data_with_income_cat = add_income_category(data)
    train_set, test_set = train_test_split(
        data_with_income_cat,
        test_size=test_size,
        stratify=data_with_income_cat["income_cat"],
        random_state=random_state,
    )

    return (
        train_set.drop(columns="income_cat"),
        test_set.drop(columns="income_cat"),
    )


def split_features_labels(
    data: pd.DataFrame,
    target_column: str = "median_house_value",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target."""
    return data.drop(columns=target_column), data[target_column].copy()


def shuffle_and_split_data(
    data: pd.DataFrame,
    test_ratio: float,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a random train/test split, kept for parity with the source notebook."""
    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def is_id_in_test_set(identifier: Union[int, float], test_ratio: float) -> bool:
    """Stable hash-based test-set membership."""
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(
    data: pd.DataFrame,
    test_ratio: float,
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data using a stable hash of a unique identifier column."""
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
