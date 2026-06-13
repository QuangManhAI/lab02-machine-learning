"""End-to-end product sales prediction pipeline.

This script implements the plan in agents/requirements.md for the current
product sales CSV: data validation, EDA artifacts, feature engineering,
model comparison, and deployment-style model export.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "product_sales_dataset_final.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_PLOT_DIR = DEFAULT_REPORT_DIR / "plots"

TARGET = "Revenue"
DATE_COLUMN = "Order_Date"
DATE_FORMAT = "%m-%d-%y"
RANDOM_STATE = 42
TEST_START_DATE = pd.Timestamp("2024-10-01")

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
    "product_avg_quantity",
    "category_avg_quantity",
    "sub_category_avg_quantity",
    "state_avg_quantity",
    "region_avg_quantity",
    "expected_revenue_product",
    "expected_revenue_category",
    "expected_revenue_sub_category",
    "expected_revenue_state",
    "expected_revenue_region",
]
QUANTITY_AGG_FEATURES = [
    "product_avg_quantity",
    "category_avg_quantity",
    "sub_category_avg_quantity",
    "state_avg_quantity",
    "region_avg_quantity",
]
EXPECTED_REVENUE_FEATURE_SOURCES = {
    "expected_revenue_product": "product_avg_quantity",
    "expected_revenue_category": "category_avg_quantity",
    "expected_revenue_sub_category": "sub_category_avg_quantity",
    "expected_revenue_state": "state_avg_quantity",
    "expected_revenue_region": "region_avg_quantity",
}


@dataclass(frozen=True)
class DatasetChecks:
    rows: int
    columns: int
    missing_total: int
    duplicate_rows: int
    date_min: str
    date_max: str
    date_parse_failures: int
    revenue_formula_mismatch_rows: int
    revenue_formula_max_abs_diff: float
    stripped_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA and regression modeling for product sales prediction."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_and_clean_data(data_path: Path) -> tuple[pd.DataFrame, DatasetChecks]:
    df = pd.read_csv(data_path)
    original_columns = list(df.columns)
    df.columns = df.columns.str.strip()
    stripped_columns = [
        original for original in original_columns if original != original.strip()
    ]

    parsed_dates = pd.to_datetime(df[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
    date_parse_failures = int(parsed_dates.isna().sum())
    df[DATE_COLUMN] = parsed_dates

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
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    formula_diff = (df[TARGET].round(2) - (df["Quantity"] * df["Unit_Price"]).round(2)).abs()
    checks = DatasetChecks(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        missing_total=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        date_min=str(df[DATE_COLUMN].min().date()),
        date_max=str(df[DATE_COLUMN].max().date()),
        date_parse_failures=date_parse_failures,
        revenue_formula_mismatch_rows=int((formula_diff > 0.01).sum()),
        revenue_formula_max_abs_diff=float(formula_diff.max()),
        stripped_columns=stripped_columns,
    )

    assert not df.columns.str.contains(r"^\s|\s$").any(), "Column names still have whitespace"
    assert checks.date_parse_failures == 0, "Order_Date contains unparseable values"
    assert checks.missing_total == 0, "Dataset contains missing values"
    assert checks.duplicate_rows == 0, "Dataset contains duplicate rows"
    assert checks.revenue_formula_mismatch_rows == 0, "Revenue formula validation failed"

    return df, checks


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[DATE_COLUMN].dt.year
    out["month"] = out[DATE_COLUMN].dt.month
    out["quarter"] = out[DATE_COLUMN].dt.quarter
    out["day_of_week"] = out[DATE_COLUMN].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[DATE_COLUMN] < TEST_START_DATE].copy()
    test_df = df[df[DATE_COLUMN] >= TEST_START_DATE].copy()
    assert train_df[DATE_COLUMN].max() < test_df[DATE_COLUMN].min()
    return train_df, test_df


def add_aggregate_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[Any, float]], dict[str, float]]:
    train = train_df.copy()
    test = test_df.copy()
    global_mean = float(train[TARGET].mean())
    global_quantity_mean = float(train["Quantity"].mean())

    mapping_specs = {
        "product_order_count": ("Product_Name", train.groupby("Product_Name").size()),
        "product_avg_revenue": ("Product_Name", train.groupby("Product_Name")[TARGET].mean()),
        "category_avg_revenue": ("Category", train.groupby("Category")[TARGET].mean()),
        "state_avg_revenue": ("State", train.groupby("State")[TARGET].mean()),
        "region_avg_revenue": ("Region", train.groupby("Region")[TARGET].mean()),
        "product_avg_quantity": ("Product_Name", train.groupby("Product_Name")["Quantity"].mean()),
        "category_avg_quantity": ("Category", train.groupby("Category")["Quantity"].mean()),
        "sub_category_avg_quantity": (
            "Sub_Category",
            train.groupby("Sub_Category")["Quantity"].mean(),
        ),
        "state_avg_quantity": ("State", train.groupby("State")["Quantity"].mean()),
        "region_avg_quantity": ("Region", train.groupby("Region")["Quantity"].mean()),
    }

    mappings: dict[str, dict[Any, float]] = {}
    defaults: dict[str, float] = {}
    for feature_name, (source_col, series) in mapping_specs.items():
        mapping = series.to_dict()
        mappings[feature_name] = mapping
        if feature_name.endswith("count"):
            default = float(series.median())
        elif feature_name in QUANTITY_AGG_FEATURES:
            default = global_quantity_mean
        else:
            default = global_mean
        defaults[feature_name] = default
        train[feature_name] = train[source_col].map(mapping).fillna(defaults[feature_name])
        test[feature_name] = test[source_col].map(mapping).fillna(defaults[feature_name])

    for feature_name, quantity_feature in EXPECTED_REVENUE_FEATURE_SOURCES.items():
        train[feature_name] = train["Unit_Price"] * train[quantity_feature]
        test[feature_name] = test["Unit_Price"] * test[quantity_feature]

    return train, test, mappings, defaults


def prepare_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, dict[Any, float]]]:
    with_time = add_time_features(df)
    train_df, test_df = time_split(with_time)
    train_df, test_df, mappings, defaults = add_aggregate_features(train_df, test_df)

    metadata = {
        "target": TARGET,
        "test_start_date": str(TEST_START_DATE.date()),
        "drop_from_features": DROP_FROM_FEATURES,
        "aggregate_feature_defaults": defaults,
        "aggregate_feature_source": "computed from training data only to avoid time leakage",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df[DATE_COLUMN].min().date()),
        "train_date_max": str(train_df[DATE_COLUMN].max().date()),
        "test_date_min": str(test_df[DATE_COLUMN].min().date()),
        "test_date_max": str(test_df[DATE_COLUMN].max().date()),
        "aggregate_mapping_sizes": {key: len(value) for key, value in mappings.items()},
    }
    return train_df, test_df, metadata, mappings


def safe_filename(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "and")
        .replace("__", "_")
    )


def save_bar_chart(series: pd.Series, title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    series.plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_eda_plots(df: pd.DataFrame, plot_dir: Path) -> list[str]:
    ensure_dirs(plot_dir)
    saved: list[str] = []

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[TARGET], bins=60, color="#4c78a8", edgecolor="white")
    ax.set_title("Revenue Distribution")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Order Count")
    fig.tight_layout()
    path = plot_dir / "revenue_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(df[TARGET], vert=False)
    ax.set_title("Revenue Boxplot")
    ax.set_xlabel("Revenue")
    fig.tight_layout()
    path = plot_dir / "revenue_boxplot.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    monthly = df.set_index(DATE_COLUMN).resample("ME")[TARGET].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly.plot(ax=ax, marker="o", color="#f58518")
    ax.set_title("Monthly Revenue")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    fig.tight_layout()
    path = plot_dir / "monthly_revenue.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    charts = {
        "revenue_by_category.png": (
            df.groupby("Category")[TARGET].sum().sort_values(ascending=False),
            "Revenue by Category",
            "Revenue",
        ),
        "revenue_by_region.png": (
            df.groupby("Region")[TARGET].sum().sort_values(ascending=False),
            "Revenue by Region",
            "Revenue",
        ),
        "top_products_by_revenue.png": (
            df.groupby("Product_Name")[TARGET].sum().sort_values(ascending=False).head(15),
            "Top Products by Revenue",
            "Revenue",
        ),
        "top_states_by_revenue.png": (
            df.groupby("State")[TARGET].sum().sort_values(ascending=False).head(15),
            "Top States by Revenue",
            "Revenue",
        ),
    }
    for filename, (series, title, ylabel) in charts.items():
        path = plot_dir / filename
        save_bar_chart(series, title, ylabel, path)
        saved.append(str(path))

    for feature in ["Unit_Price", "Quantity"]:
        sample = df.sample(n=min(15000, len(df)), random_state=RANDOM_STATE)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sample[feature], sample[TARGET], alpha=0.18, s=8, color="#54a24b")
        ax.set_title(f"{feature} vs Revenue")
        ax.set_xlabel(feature)
        ax.set_ylabel("Revenue")
        fig.tight_layout()
        path = plot_dir / f"{safe_filename(feature)}_vs_revenue.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))

    return saved


def top_table(df: pd.DataFrame, group_col: str, rows: int = 10) -> pd.DataFrame:
    return (
        df.groupby(group_col)
        .agg(
            orders=("Order_ID", "count"),
            quantity=("Quantity", "sum"),
            revenue=(TARGET, "sum"),
            avg_revenue=(TARGET, "mean"),
            median_revenue=(TARGET, "median"),
        )
        .sort_values("revenue", ascending=False)
        .head(rows)
        .round(2)
    )


def make_eda_summary(
    df: pd.DataFrame, checks: DatasetChecks, plot_paths: list[str], report_dir: Path
) -> dict[str, Any]:
    target_percentiles = (
        df[TARGET]
        .quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .round(2)
        .to_dict()
    )
    monthly = df.set_index(DATE_COLUMN).resample("ME")[TARGET].sum()
    weekday = df.groupby(df[DATE_COLUMN].dt.day_name())[TARGET].sum().sort_values(ascending=False)
    summary = {
        "checks": checks.__dict__,
        "target": {
            "mean": float(df[TARGET].mean()),
            "median": float(df[TARGET].median()),
            "std": float(df[TARGET].std()),
            "min": float(df[TARGET].min()),
            "max": float(df[TARGET].max()),
            "skew": float(df[TARGET].skew()),
            "percentiles": {str(k): float(v) for k, v in target_percentiles.items()},
        },
        "cardinality": df.nunique().to_dict(),
        "numeric_describe": df[["Quantity", "Unit_Price", TARGET, "Profit"]]
        .describe()
        .round(2)
        .to_dict(),
        "top_category_revenue": top_table(df, "Category", 10).reset_index().to_dict("records"),
        "top_sub_category_revenue": top_table(df, "Sub_Category", 10).reset_index().to_dict("records"),
        "top_product_revenue": top_table(df, "Product_Name", 10).reset_index().to_dict("records"),
        "top_region_revenue": top_table(df, "Region", 10).reset_index().to_dict("records"),
        "top_state_revenue": top_table(df, "State", 10).reset_index().to_dict("records"),
        "top_city_revenue": top_table(df, "City", 10).reset_index().to_dict("records"),
        "yearly_revenue": df.groupby(df[DATE_COLUMN].dt.year)[TARGET].sum().round(2).to_dict(),
        "quarterly_revenue": df.groupby(df[DATE_COLUMN].dt.to_period("Q"))[TARGET]
        .sum()
        .round(2)
        .astype(float)
        .rename(index=str)
        .to_dict(),
        "best_month": {
            "month": str(monthly.idxmax().date()),
            "revenue": float(monthly.max()),
        },
        "best_weekday": {
            "weekday": str(weekday.index[0]),
            "revenue": float(weekday.iloc[0]),
        },
        "correlation": df[["Quantity", "Unit_Price", TARGET, "Profit"]].corr().round(4).to_dict(),
        "plots": plot_paths,
    }

    json_path = report_dir / "eda_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = build_eda_markdown(summary)
    (report_dir / "eda_summary.md").write_text(md, encoding="utf-8")
    return summary


def markdown_table(records: list[dict[str, Any]], first_col: str) -> str:
    if not records:
        return ""
    columns = list(records[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for record in records:
        values = []
        for col in columns:
            value = record[col]
            if isinstance(value, float):
                values.append(f"{value:,.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_eda_markdown(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    target = summary["target"]
    best_month = summary["best_month"]
    best_weekday = summary["best_weekday"]
    return f"""# EDA Summary: Product Sales Dataset

## Data Quality

- Rows: {checks["rows"]:,}
- Columns: {checks["columns"]}
- Missing values: {checks["missing_total"]}
- Duplicate rows: {checks["duplicate_rows"]}
- Date range: {checks["date_min"]} to {checks["date_max"]}
- Date parse failures: {checks["date_parse_failures"]}
- Revenue formula mismatches: {checks["revenue_formula_mismatch_rows"]}
- Columns stripped for whitespace: {checks["stripped_columns"]}

## Target: Revenue

- Mean revenue: {target["mean"]:,.2f}
- Median revenue: {target["median"]:,.2f}
- Std revenue: {target["std"]:,.2f}
- Min revenue: {target["min"]:,.2f}
- Max revenue: {target["max"]:,.2f}
- Skewness: {target["skew"]:,.4f}

Revenue is right-skewed, so MAE and RMSE should both be reported. RMSE will penalize high-revenue errors more strongly.

## Time Insights

- Best revenue month: {best_month["month"]} with {best_month["revenue"]:,.2f}
- Best weekday by revenue: {best_weekday["weekday"]} with {best_weekday["revenue"]:,.2f}
- The dataset has daily records across 2023 and 2024, so a time-based train/test split is more realistic than a random split.

## Product Insights

### Top Categories by Revenue

{markdown_table(summary["top_category_revenue"], "Category")}

### Top Sub-Categories by Revenue

{markdown_table(summary["top_sub_category_revenue"], "Sub_Category")}

### Top Products by Revenue

{markdown_table(summary["top_product_revenue"], "Product_Name")}

## Geography Insights

### Top Regions by Revenue

{markdown_table(summary["top_region_revenue"], "Region")}

### Top States by Revenue

{markdown_table(summary["top_state_revenue"], "State")}

### Top Cities by Revenue

{markdown_table(summary["top_city_revenue"], "City")}

## Modeling Notes

- `Revenue` is the target.
- `Profit` is excluded because it is a post-sale outcome.
- `Order_ID` is excluded because it is only a row identifier.
- `Customer_Name` is excluded from the main model because it has very high cardinality and privacy/noise risk.
- `Country` is excluded because it has one value only.
- Model A includes `Quantity` and is useful to demonstrate leakage because `Revenue = Quantity * Unit_Price`.
- Model B excludes `Quantity` and is the business-realistic model for predicting revenue before the sold quantity is known.
- Regularized linear models such as Ridge and Lasso are included to reduce variance and keep Model B usable without the leaked quantity signal.
- Model B uses train-only historical average quantity features and `Unit_Price * historical_avg_quantity` proxies, never the current row's `Quantity`.

## Plot Files

{chr(10).join(f"- `{path}`" for path in summary["plots"])}
"""


def make_one_hot_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_ordinal_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_feature_sets(include_quantity: bool) -> tuple[list[str], list[str]]:
    numeric_features = ["Unit_Price"] + TIME_FEATURES + AGG_FEATURES
    if include_quantity:
        numeric_features = ["Quantity"] + numeric_features
    categorical_features = BASE_CATEGORICAL_FEATURES
    return numeric_features, categorical_features


def make_models(include_quantity: bool) -> dict[str, Pipeline]:
    numeric_features, categorical_features = get_feature_sets(include_quantity)
    one_hot = make_one_hot_preprocessor(numeric_features, categorical_features)
    ordinal = make_ordinal_preprocessor(numeric_features, categorical_features)

    return {
        "Baseline_Mean": Pipeline(
            steps=[
                ("preprocess", one_hot),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("preprocess", one_hot),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Lasso": Pipeline(
            steps=[
                ("preprocess", one_hot),
                ("model", Lasso(alpha=0.2, max_iter=20000, random_state=RANDOM_STATE)),
            ]
        ),
        "DecisionTree": Pipeline(
            steps=[
                ("preprocess", one_hot),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=16,
                        min_samples_leaf=40,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", one_hot),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=20,
                        max_depth=12,
                        min_samples_leaf=80,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("preprocess", ordinal),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=100,
                        learning_rate=0.08,
                        l2_regularization=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def metrics_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant_name: str,
    include_quantity: bool,
) -> tuple[list[dict[str, Any]], Pipeline, np.ndarray, list[str]]:
    numeric_features, categorical_features = get_feature_sets(include_quantity)
    feature_columns = numeric_features + categorical_features

    forbidden_features = {TARGET, "Profit", "Order_ID", "Customer_Name", "Country"}
    assert TARGET not in feature_columns
    assert forbidden_features.isdisjoint(feature_columns)
    if not include_quantity:
        assert "Quantity" not in feature_columns

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}

    for model_name, pipeline in make_models(include_quantity).items():
        print(f"Training {variant_name} / {model_name}...")
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        model_metrics = metrics_dict(y_test, y_pred)
        results.append(
            {
                "variant": variant_name,
                "model": model_name,
                "include_quantity": include_quantity,
                "features": feature_columns,
                **model_metrics,
            }
        )
        fitted_models[model_name] = pipeline
        predictions[model_name] = y_pred

    best_record = min(results, key=lambda item: item["RMSE"])
    best_model_name = best_record["model"]
    return results, fitted_models[best_model_name], predictions[best_model_name], feature_columns


def save_prediction_plots(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    variant_name: str,
    plot_dir: Path,
) -> list[str]:
    ensure_dirs(plot_dir)
    y_true = test_df[TARGET]
    saved: list[str] = []

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.2, s=8, color="#4c78a8")
    max_value = max(float(y_true.max()), float(np.max(y_pred)))
    ax.plot([0, max_value], [0, max_value], color="#e45756", linewidth=2)
    ax.set_title(f"Predicted vs Actual Revenue - {variant_name}")
    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Predicted Revenue")
    fig.tight_layout()
    path = plot_dir / f"predicted_vs_actual_{variant_name.lower()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred, residuals, alpha=0.2, s=8, color="#f58518")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title(f"Residual Plot - {variant_name}")
    ax.set_xlabel("Predicted Revenue")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    path = plot_dir / f"residuals_{variant_name.lower()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    residual_table = test_df[[DATE_COLUMN, "Category"]].copy()
    residual_table["actual"] = y_true.to_numpy()
    residual_table["predicted"] = y_pred
    residual_table["residual"] = residuals.to_numpy()
    residual_by_category = residual_table.groupby("Category")["residual"].mean().sort_values()
    path = plot_dir / f"residual_by_category_{variant_name.lower()}.png"
    save_bar_chart(
        residual_by_category,
        f"Mean Residual by Category - {variant_name}",
        "Mean Residual",
        path,
    )
    saved.append(str(path))

    residual_by_month = residual_table.groupby(residual_table[DATE_COLUMN].dt.to_period("M"))[
        "residual"
    ].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    residual_by_month.rename(index=str).plot(kind="bar", ax=ax, color="#72b7b2")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title(f"Mean Residual by Month - {variant_name}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Residual")
    fig.tight_layout()
    path = plot_dir / f"residual_by_month_{variant_name.lower()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    return saved


def build_model_report(
    metrics: list[dict[str, Any]],
    best_business_record: dict[str, Any],
    metadata: dict[str, Any],
    plot_paths: list[str],
) -> str:
    metric_rows = []
    for row in sorted(metrics, key=lambda item: (item["variant"], item["RMSE"])):
        metric_rows.append(
            {
                "variant": row["variant"],
                "model": row["model"],
                "MAE": row["MAE"],
                "MSE": row["MSE"],
                "RMSE": row["RMSE"],
                "R2": row["R2"],
            }
        )

    return f"""# Model Report: Product Sales Prediction

## Split Strategy

- Split type: time-based split.
- Train rows: {metadata["train_rows"]:,}
- Test rows: {metadata["test_rows"]:,}
- Train dates: {metadata["train_date_min"]} to {metadata["train_date_max"]}
- Test dates: {metadata["test_date_min"]} to {metadata["test_date_max"]}

## Model Variants

- Model A includes `Quantity`. This demonstrates leakage because `Revenue = Quantity * Unit_Price`.
- Model B excludes `Quantity`. This is the business-realistic model for predicting revenue before sold quantity is known.
- Excluded from all models: `Order_ID`, `Customer_Name`, `Country`, `Profit`, `Revenue`.
- Ridge and Lasso are compared as regularized linear models for the non-leaking Model B setup.
- Model B includes train-only historical average quantity proxies and expected-revenue interaction features, but still excludes the current order's `Quantity`.

## Metrics

{markdown_table(metric_rows, "variant")}

## Selected Model

The selected deployment model is **{best_business_record["model"]}** from **{best_business_record["variant"]}**.

- MAE: {best_business_record["MAE"]:,.2f}
- RMSE: {best_business_record["RMSE"]:,.2f}
- R2: {best_business_record["R2"]:,.4f}

Model B is selected even if Model A has stronger metrics, because Model A uses `Quantity` and therefore knows a major component of `Revenue` in advance.

## Limitations

- The dataset does not contain true campaign/ad-spend fields; time features are used as seasonal proxies.
- The dataset does not contain age/gender demographics; geography fields are used as customer-context proxies.
- Aggregate features are computed from training data only to reduce time leakage.

## Plot Files

{chr(10).join(f"- `{path}`" for path in plot_paths)}
"""


def save_metrics_and_model(
    artifact_dir: Path,
    metrics: list[dict[str, Any]],
    best_model: Pipeline,
    best_record: dict[str, Any],
    feature_columns: list[str],
    metadata: dict[str, Any],
    aggregate_mappings: dict[str, dict[Any, float]],
) -> None:
    ensure_dirs(artifact_dir)
    compact_metrics = []
    for row in metrics:
        compact = {key: value for key, value in row.items() if key != "features"}
        compact["feature_count"] = len(row["features"])
        compact_metrics.append(compact)

    payload = {
        "selected_model": best_record,
        "metrics": compact_metrics,
        "feature_columns": feature_columns,
        "metadata": metadata,
        "leakage_note": "Deployment model excludes Quantity; Model A with Quantity is included only for comparison.",
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (artifact_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )
    joblib.dump(best_model, artifact_dir / "best_model.joblib")
    joblib.dump(
        {
            "model": best_model,
            "selected_model": best_record,
            "feature_columns": feature_columns,
            "aggregate_mappings": aggregate_mappings,
            "aggregate_feature_defaults": metadata["aggregate_feature_defaults"],
            "date_column": DATE_COLUMN,
            "date_format": DATE_FORMAT,
            "target": TARGET,
            "excluded_columns": DROP_FROM_FEATURES,
        },
        artifact_dir / "model_bundle.joblib",
    )


def main() -> None:
    args = parse_args()
    ensure_dirs(args.report_dir, args.artifact_dir, args.plot_dir)

    df, checks = load_and_clean_data(args.data_path)
    plot_paths = save_eda_plots(df, args.plot_dir)
    make_eda_summary(df, checks, plot_paths, args.report_dir)

    train_df, test_df, metadata, aggregate_mappings = prepare_model_data(df)

    all_metrics: list[dict[str, Any]] = []
    model_plot_paths: list[str] = []

    model_a_metrics, _, model_a_pred, _ = train_and_evaluate_variant(
        train_df, test_df, "Model_A_With_Quantity", include_quantity=True
    )
    all_metrics.extend(model_a_metrics)
    model_plot_paths.extend(
        save_prediction_plots(test_df, model_a_pred, "Model_A_With_Quantity", args.plot_dir)
    )

    model_b_metrics, model_b_best, model_b_pred, model_b_features = train_and_evaluate_variant(
        train_df, test_df, "Model_B_No_Quantity", include_quantity=False
    )
    all_metrics.extend(model_b_metrics)
    model_plot_paths.extend(
        save_prediction_plots(test_df, model_b_pred, "Model_B_No_Quantity", args.plot_dir)
    )

    best_business_record = min(model_b_metrics, key=lambda item: item["RMSE"])
    save_metrics_and_model(
        args.artifact_dir,
        all_metrics,
        model_b_best,
        best_business_record,
        model_b_features,
        metadata,
        aggregate_mappings,
    )
    model_report = build_model_report(
        all_metrics, best_business_record, metadata, model_plot_paths
    )
    (args.report_dir / "model_report.md").write_text(model_report, encoding="utf-8")

    print("Pipeline completed.")
    print(f"EDA summary: {args.report_dir / 'eda_summary.md'}")
    print(f"Model report: {args.report_dir / 'model_report.md'}")
    print(f"Metrics: {args.artifact_dir / 'metrics.json'}")
    print(f"Best model: {args.artifact_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
