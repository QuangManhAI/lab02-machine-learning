"""Data processing utilities for product sales prediction.

This module contains reusable code extracted from the notebook:
loading, validation, cleaning, feature engineering, time split, and
single-row feature preparation for prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "product_sales_dataset_final.csv"

TARGET = "Revenue"
DATE_COLUMN = "Order_Date"
DATE_FORMAT = "%m-%d-%y"
TEST_START_DATE = pd.Timestamp("2024-10-01")
RANDOM_STATE = 42

DROP_FROM_FEATURES = ["Order_ID", "Customer_Name", "Country", "Profit", TARGET]
BASE_CATEGORICAL_FEATURES = [
    "City",
    "State",
    "Region",
    "Category",
    "Sub_Category",
    "Product_Name",
]
TIME_FEATURES = ["year", "month", "quarter", "day_of_week", "is_weekend"]
AGG_FEATURES = [
    "product_order_count",
    "product_avg_revenue",
    "category_avg_revenue",
    "state_avg_revenue",
    "region_avg_revenue",
]


@dataclass(frozen=True)
class DatasetChecks:
    rows: int
    columns: int
    stripped_columns: list[str]
    missing_total: int
    duplicate_rows: int
    date_parse_failures: int
    date_min: str
    date_max: str
    revenue_formula_mismatch_rows: int
    revenue_formula_max_abs_diff: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_raw_data(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the product sales CSV."""
    return pd.read_csv(data_path)


def clean_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Strip leading/trailing whitespace in column names."""
    out = df.copy()
    original_columns = list(out.columns)
    out.columns = out.columns.str.strip()
    stripped_columns = [col for col in original_columns if col != col.strip()]
    return out, stripped_columns


def parse_order_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Order_Date into pandas datetime."""
    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
    return out


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "Order_ID",
        DATE_COLUMN,
        "Customer_Name",
        "City",
        "State",
        "Region",
        "Country",
        "Category",
        "Sub_Category",
        "Product_Name",
        "Quantity",
        "Unit_Price",
        TARGET,
        "Profit",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_dataset(df: pd.DataFrame, stripped_columns: list[str]) -> DatasetChecks:
    """Run deterministic data checks used in the notebook."""
    validate_required_columns(df)
    formula_diff = (df[TARGET].round(2) - (df["Quantity"] * df["Unit_Price"]).round(2)).abs()

    checks = DatasetChecks(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        stripped_columns=stripped_columns,
        missing_total=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        date_parse_failures=int(df[DATE_COLUMN].isna().sum()),
        date_min=str(df[DATE_COLUMN].min().date()),
        date_max=str(df[DATE_COLUMN].max().date()),
        revenue_formula_mismatch_rows=int((formula_diff > 0.01).sum()),
        revenue_formula_max_abs_diff=float(formula_diff.max()),
    )

    assert not df.columns.str.contains(r"^\s|\s$").any(), "Column names still contain whitespace"
    assert checks.missing_total == 0, "Dataset contains missing values"
    assert checks.duplicate_rows == 0, "Dataset contains duplicate rows"
    assert checks.date_parse_failures == 0, "Order_Date parse failed"
    assert checks.revenue_formula_mismatch_rows == 0, "Revenue formula check failed"
    return checks


def load_clean_validate(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, DatasetChecks]:
    """Load, clean, parse dates, and validate the dataset."""
    raw = load_raw_data(data_path)
    cleaned, stripped_columns = clean_column_names(raw)
    cleaned = parse_order_date(cleaned)
    checks = validate_dataset(cleaned, stripped_columns)
    return cleaned, checks


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[DATE_COLUMN].dt.year
    out["month"] = out[DATE_COLUMN].dt.month
    out["quarter"] = out[DATE_COLUMN].dt.quarter
    out["day_of_week"] = out[DATE_COLUMN].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def time_split(
    df: pd.DataFrame, test_start_date: pd.Timestamp = TEST_START_DATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[DATE_COLUMN] < test_start_date].copy()
    test_df = df[df[DATE_COLUMN] >= test_start_date].copy()
    assert train_df[DATE_COLUMN].max() < test_df[DATE_COLUMN].min()
    return train_df, test_df


def add_aggregate_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[Any, float]], dict[str, float]]:
    """Create history/proxy aggregate features from train only."""
    train = train_df.copy()
    test = test_df.copy()
    global_mean = float(train[TARGET].mean())

    specs = {
        "product_order_count": ("Product_Name", train.groupby("Product_Name").size()),
        "product_avg_revenue": ("Product_Name", train.groupby("Product_Name")[TARGET].mean()),
        "category_avg_revenue": ("Category", train.groupby("Category")[TARGET].mean()),
        "state_avg_revenue": ("State", train.groupby("State")[TARGET].mean()),
        "region_avg_revenue": ("Region", train.groupby("Region")[TARGET].mean()),
    }

    mappings: dict[str, dict[Any, float]] = {}
    defaults: dict[str, float] = {}
    for feature_name, (source_col, series) in specs.items():
        mapping = series.to_dict()
        default = float(series.median() if feature_name.endswith("count") else global_mean)
        mappings[feature_name] = mapping
        defaults[feature_name] = default
        train[feature_name] = train[source_col].map(mapping).fillna(default)
        test[feature_name] = test[source_col].map(mapping).fillna(default)

    return train, test, mappings, defaults


def prepare_model_data(
    df: pd.DataFrame, test_start_date: pd.Timestamp = TEST_START_DATE
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[Any, float]], dict[str, float]]:
    """Add features and create a time-based train/test split."""
    with_time = add_time_features(df)
    train_df, test_df = time_split(with_time, test_start_date)
    return add_aggregate_features(train_df, test_df)


def get_feature_sets(include_quantity: bool) -> tuple[list[str], list[str], list[str]]:
    numeric_features = ["Unit_Price"] + TIME_FEATURES + AGG_FEATURES
    if include_quantity:
        numeric_features = ["Quantity"] + numeric_features
    categorical_features = BASE_CATEGORICAL_FEATURES.copy()
    feature_columns = numeric_features + categorical_features
    return numeric_features, categorical_features, feature_columns


def assert_no_leakage(feature_columns: list[str], include_quantity: bool) -> None:
    forbidden = {TARGET, "Profit", "Order_ID", "Customer_Name", "Country"}
    assert TARGET not in feature_columns
    assert forbidden.isdisjoint(feature_columns)
    if not include_quantity:
        assert "Quantity" not in feature_columns


def prepare_single_input(
    raw: dict[str, Any],
    feature_columns: list[str],
    aggregate_mappings: dict[str, dict[Any, float]],
    aggregate_defaults: dict[str, float],
) -> pd.DataFrame:
    """Create model-ready features for one raw order-like input."""
    row = pd.DataFrame([raw])
    row[DATE_COLUMN] = pd.to_datetime(row[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
    if row[DATE_COLUMN].isna().any():
        raise ValueError(f"{DATE_COLUMN} must match format {DATE_FORMAT}, e.g. 12-15-24")

    row = add_time_features(row)
    source_by_aggregate = {
        "product_order_count": "Product_Name",
        "product_avg_revenue": "Product_Name",
        "category_avg_revenue": "Category",
        "state_avg_revenue": "State",
        "region_avg_revenue": "Region",
    }
    for feature_name, source_col in source_by_aggregate.items():
        row[feature_name] = (
            row[source_col]
            .map(aggregate_mappings[feature_name])
            .fillna(aggregate_defaults[feature_name])
        )

    missing = sorted(set(feature_columns).difference(row.columns))
    if missing:
        raise ValueError(f"Missing required input-derived features: {missing}")
    return row[feature_columns]
