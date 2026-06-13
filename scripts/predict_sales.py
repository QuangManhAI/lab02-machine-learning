"""Predict product revenue from a raw JSON order-like record.

Example:
    python scripts/predict_sales.py \
      --input-json '{"Order_Date":"12-15-24","City":"Boston","State":"Massachusetts","Region":"East","Category":"Electronics","Sub_Category":"Laptops","Product_Name":"MacBook Air","Unit_Price":999.99}'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "model_bundle.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict product sales revenue.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Raw order JSON with date, product, geography, and Unit_Price fields.",
    )
    return parser.parse_args()


def add_features(raw: dict[str, Any], bundle: dict[str, Any]) -> pd.DataFrame:
    row = dict(raw)
    date_column = bundle["date_column"]
    date_format = bundle["date_format"]
    feature_columns = bundle["feature_columns"]
    aggregate_mappings = bundle["aggregate_mappings"]
    aggregate_defaults = bundle["aggregate_feature_defaults"]

    df = pd.DataFrame([row])
    df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors="coerce")
    if df[date_column].isna().any():
        raise ValueError(f"{date_column} must match date format {date_format}, e.g. 12-15-24")

    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["quarter"] = df[date_column].dt.quarter
    df["day_of_week"] = df[date_column].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    source_by_aggregate = {
        "product_order_count": "Product_Name",
        "product_avg_revenue": "Product_Name",
        "category_avg_revenue": "Category",
        "state_avg_revenue": "State",
        "region_avg_revenue": "Region",
        "product_avg_quantity": "Product_Name",
        "category_avg_quantity": "Category",
        "sub_category_avg_quantity": "Sub_Category",
        "state_avg_quantity": "State",
        "region_avg_quantity": "Region",
    }
    for feature_name, source_col in source_by_aggregate.items():
        mapping = aggregate_mappings[feature_name]
        default = aggregate_defaults[feature_name]
        df[feature_name] = df[source_col].map(mapping).fillna(default)

    expected_revenue_feature_sources = {
        "expected_revenue_product": "product_avg_quantity",
        "expected_revenue_category": "category_avg_quantity",
        "expected_revenue_sub_category": "sub_category_avg_quantity",
        "expected_revenue_state": "state_avg_quantity",
        "expected_revenue_region": "region_avg_quantity",
    }
    for feature_name, quantity_feature in expected_revenue_feature_sources.items():
        df[feature_name] = df["Unit_Price"] * df[quantity_feature]

    missing = sorted(set(feature_columns).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required input-derived features: {missing}")

    return df[feature_columns]


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.bundle_path)
    raw = json.loads(args.input_json)
    features = add_features(raw, bundle)
    prediction = float(bundle["model"].predict(features)[0])
    print(json.dumps({"predicted_revenue": round(prediction, 2)}, indent=2))


if __name__ == "__main__":
    main()
