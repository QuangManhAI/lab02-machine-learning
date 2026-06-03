"""EDA and visualization utilities for product sales prediction.

The file name follows the user's requested name: era.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from process import DATE_COLUMN, RANDOM_STATE, TARGET


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def group_sales(df: pd.DataFrame, group_col: str, n: int = 10) -> pd.DataFrame:
    """Aggregate order count, quantity, revenue, and average revenue by group."""
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
        .head(n)
        .round(2)
    )


def plot_hist_with_kde(ax, values: Iterable[float], bins: int, color: str, title: str, xlabel: str) -> None:
    values = pd.Series(values).dropna().astype(float)
    ax.hist(values, bins=bins, density=True, alpha=0.55, color=color, edgecolor="white")
    sample = values.sample(min(len(values), 20000), random_state=RANDOM_STATE)
    kde = stats.gaussian_kde(sample)
    x_grid = np.linspace(values.quantile(0.005), values.quantile(0.995), 300)
    ax.plot(x_grid, kde(x_grid), color="#222222", linewidth=2, label="KDE")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")


def plot_target_distribution(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[TARGET], bins=60, color="#4c78a8", edgecolor="white")
    axes[0].set_title("Revenue Distribution")
    axes[0].set_xlabel("Revenue")
    axes[0].set_ylabel("Order count")
    axes[1].boxplot(df[TARGET], vert=False)
    axes[1].set_title("Revenue Boxplot")
    axes[1].set_xlabel("Revenue")
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_revenue_distribution_boxplot.png", dpi=160)
    plt.show()


def plot_advanced_target_distribution(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].hist(df[TARGET], bins=60, edgecolor="white", alpha=0.85, color="#4c78a8")
    axes[0, 0].set_title("Distribution of Revenue")
    axes[0, 1].hist(np.log1p(df[TARGET]), bins=60, edgecolor="white", alpha=0.85, color="#f58518")
    axes[0, 1].set_title("Distribution of log1p(Revenue)")
    stats.probplot(df[TARGET].sample(15000, random_state=RANDOM_STATE), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("QQ Plot - Revenue Sample")
    stats.probplot(
        np.log1p(df[TARGET].sample(15000, random_state=RANDOM_STATE)),
        dist="norm",
        plot=axes[1, 1],
    )
    axes[1, 1].set_title("QQ Plot - log1p(Revenue) Sample")
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_advanced_target_distribution.png", dpi=160)
    plt.show()


def plot_product_geography_eda(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    top_category = group_sales(df, "Category")
    top_sub_category = group_sales(df, "Sub_Category")
    top_region = group_sales(df, "Region")
    top_product = group_sales(df, "Product_Name")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    top_category["revenue"].plot(kind="bar", ax=axes[0, 0], color="#4c78a8", title="Revenue by Category")
    top_sub_category["revenue"].head(10).plot(
        kind="bar", ax=axes[0, 1], color="#f58518", title="Top Sub-Categories by Revenue"
    )
    top_region["revenue"].plot(kind="bar", ax=axes[1, 0], color="#54a24b", title="Revenue by Region")
    top_product["revenue"].head(10).plot(
        kind="bar", ax=axes[1, 1], color="#b279a2", title="Top Products by Revenue"
    )
    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("Revenue")
        ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_product_geography_eda.png", dpi=160)
    plt.show()


def plot_time_eda(df: pd.DataFrame, output_dir: str | Path | None = None) -> tuple[pd.Series, pd.Series, pd.Series]:
    monthly_revenue = df.set_index(DATE_COLUMN).resample("ME")[TARGET].sum()
    quarterly_revenue = df.groupby(df[DATE_COLUMN].dt.to_period("Q"))[TARGET].sum()
    weekday_revenue = df.groupby(df[DATE_COLUMN].dt.day_name())[TARGET].sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    monthly_revenue.plot(ax=axes[0], marker="o", color="#e45756", title="Monthly Revenue")
    weekday_revenue.plot(kind="bar", ax=axes[1], color="#72b7b2", title="Revenue by Weekday")
    axes[0].set_ylabel("Revenue")
    axes[1].set_ylabel("Revenue")
    axes[1].tick_params(axis="x", rotation=35)
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_time_eda.png", dpi=160)
    plt.show()
    return monthly_revenue, quarterly_revenue, weekday_revenue


def plot_relationship_eda(df: pd.DataFrame, output_dir: str | Path | None = None) -> pd.DataFrame:
    sample = df.sample(n=min(15000, len(df)), random_state=RANDOM_STATE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(sample["Unit_Price"], sample[TARGET], alpha=0.18, s=8, color="#4c78a8")
    axes[0].set_title("Unit_Price vs Revenue")
    axes[0].set_xlabel("Unit_Price")
    axes[0].set_ylabel("Revenue")
    axes[1].scatter(sample["Quantity"], sample[TARGET], alpha=0.18, s=8, color="#f58518")
    axes[1].set_title("Quantity vs Revenue")
    axes[1].set_xlabel("Quantity")
    axes[1].set_ylabel("Revenue")
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_relationship_eda.png", dpi=160)
    plt.show()
    return df[["Quantity", "Unit_Price", TARGET, "Profit"]].corr().round(4)


def plot_numeric_correlation_heatmap(df: pd.DataFrame, output_dir: str | Path | None = None) -> pd.DataFrame:
    numeric_cols = ["Quantity", "Unit_Price", TARGET, "Profit"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=35, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Correlation Matrix - Numeric Features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_numeric_correlation_heatmap.png", dpi=160)
    plt.show()
    return corr.round(4)


def plot_segment_boxplots(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    df.boxplot(column=TARGET, by="Category", ax=axes[0], grid=False)
    axes[0].set_title("Revenue by Category")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Revenue")
    axes[0].tick_params(axis="x", rotation=25)
    df.boxplot(column=TARGET, by="Region", ax=axes[1], grid=False)
    axes[1].set_title("Revenue by Region")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Revenue")
    fig.suptitle("")
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_revenue_boxplot_by_segments.png", dpi=160)
    plt.show()


def plot_daily_revenue_rolling_mean(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    daily_revenue = df.set_index(DATE_COLUMN).resample("D")[TARGET].sum()
    daily_revenue_7d = daily_revenue.rolling(window=7, min_periods=1).mean()
    daily_revenue_30d = daily_revenue.rolling(window=30, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(daily_revenue.index, daily_revenue.values, alpha=0.25, label="Daily revenue", color="#4c78a8")
    ax.plot(daily_revenue_7d.index, daily_revenue_7d.values, label="7-day rolling mean", color="#f58518", linewidth=2)
    ax.plot(daily_revenue_30d.index, daily_revenue_30d.values, label="30-day rolling mean", color="#54a24b", linewidth=2)
    ax.set_title("Daily Revenue with Rolling Means")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_daily_revenue_rolling_mean.png", dpi=160)
    plt.show()


def plot_full_eda_dashboard(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    category_counts = df["Category"].value_counts().sort_values(ascending=False)
    category_revenue = df.groupby("Category")[TARGET].sum().sort_values(ascending=False)
    monthly_sales = df.set_index(DATE_COLUMN).resample("ME")[TARGET].sum()

    fig = plt.figure(figsize=(20, 15))
    ax1 = plt.subplot(2, 3, 1)
    plot_hist_with_kde(ax1, df["Unit_Price"], 50, "skyblue", "Unit Price Distribution", "Unit_Price")
    ax2 = plt.subplot(2, 3, 2)
    plot_hist_with_kde(ax2, df[TARGET], 60, "salmon", "Revenue Distribution", "Revenue")
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(category_counts.index, category_counts.values)
    ax3.set_title("Order Count by Category")
    ax3.tick_params(axis="x", rotation=25)
    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(category_revenue.index, category_revenue.values)
    ax4.set_title("Total Revenue by Category")
    ax4.tick_params(axis="x", rotation=25)
    ax5 = plt.subplot(2, 3, 5)
    box_data = [df.loc[df["Category"] == cat, TARGET] for cat in category_revenue.index]
    ax5.boxplot(box_data, tick_labels=category_revenue.index, showfliers=False)
    ax5.set_title("Revenue Boxplot by Category")
    ax5.tick_params(axis="x", rotation=25)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(monthly_sales.index.astype(str), monthly_sales.values, marker="o", color="purple", linewidth=2)
    ax6.set_title("Monthly Revenue Trend")
    ax6.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_full_eda_dashboard_like_sample.png", dpi=160)
    plt.show()


def plot_density_original_vs_scaled(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    scaled_values = pd.DataFrame(
        StandardScaler().fit_transform(df[["Unit_Price", TARGET]]),
        columns=["Unit_Price_scaled", "Revenue_scaled"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_hist_with_kde(axes[0], df["Unit_Price"], 50, "#e45756", "Original Unit_Price Density", "Unit_Price")
    plot_hist_with_kde(
        axes[1],
        scaled_values["Unit_Price_scaled"],
        50,
        "#4c78a8",
        "Scaled Unit_Price Density",
        "Standardized Unit_Price",
    )
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_density_original_vs_scaled.png", dpi=160)
    plt.show()


def plot_scatter_grid_features_vs_revenue(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    scatter_base = df.copy()
    scatter_base["month"] = scatter_base[DATE_COLUMN].dt.month
    scatter_base["day_of_week"] = scatter_base[DATE_COLUMN].dt.dayofweek
    scatter_base["quarter"] = scatter_base[DATE_COLUMN].dt.quarter
    scatter_df = scatter_base.sample(n=min(20000, len(scatter_base)), random_state=RANDOM_STATE)
    features = ["Quantity", "Unit_Price", "Profit", "month", "day_of_week", "quarter"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, col in zip(axes.ravel(), features):
        ax.scatter(scatter_df[col], scatter_df[TARGET], alpha=0.22, s=9)
        ax.set_title(f"Revenue vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Revenue")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_scatter_grid_features_vs_revenue.png", dpi=160)
    plt.show()


def plot_predictions(y_true, y_pred, title_prefix: str, output_dir: str | Path | None = None) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.2, s=8, color="#4c78a8")
    max_value = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[0].plot([0, max_value], [0, max_value], color="#e45756")
    axes[0].set_title(f"{title_prefix}: Predicted vs Actual")
    axes[0].set_xlabel("Actual Revenue")
    axes[0].set_ylabel("Predicted Revenue")
    axes[1].scatter(y_pred, residuals, alpha=0.2, s=8, color="#f58518")
    axes[1].axhline(0, color="#333333")
    axes[1].set_title(f"{title_prefix}: Residuals")
    axes[1].set_xlabel("Predicted Revenue")
    axes[1].set_ylabel("Residual")
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / f"notebook_{title_prefix.lower()}_prediction_plots.png", dpi=160)
    plt.show()


def plot_actual_vs_predicted_line(
    test_df: pd.DataFrame, y_pred, output_dir: str | Path | None = None, plot_n: int = 500
) -> None:
    ordered = test_df.copy()
    ordered["prediction"] = y_pred
    ordered = ordered.sort_values(DATE_COLUMN).reset_index(drop=True)
    plot_n = min(plot_n, len(ordered))
    plt.figure(figsize=(15, 6))
    plt.plot(ordered.loc[: plot_n - 1, TARGET].to_numpy(), label="Actual", linewidth=2)
    plt.plot(ordered.loc[: plot_n - 1, "prediction"].to_numpy(), label="Predicted", linewidth=2)
    plt.title("Model B - Actual vs Predicted Revenue Over Test Order")
    plt.xlabel("Test sample index sorted by date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_model_b_actual_vs_predicted_line.png", dpi=160)
    plt.show()


def plot_epsilon_tube_diagnostic(y_true, y_pred, output_dir: str | Path | None = None) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    epsilon = mean_absolute_error(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    padding = (max_val - min_val) * 0.05
    ideal_line = np.linspace(min_val - padding, max_val + padding, 200)
    plt.fill_between(
        ideal_line,
        ideal_line - epsilon,
        ideal_line + epsilon,
        color="#87CEEB",
        alpha=0.35,
        label=f"Epsilon-style tube (+/- MAE = {epsilon:.2f})",
    )
    plt.plot(ideal_line, ideal_line + epsilon, linestyle="--", color="#104E8B", linewidth=2)
    plt.plot(ideal_line, ideal_line - epsilon, linestyle="--", color="#104E8B", linewidth=2)
    plt.plot(ideal_line, ideal_line, label="Ideal line (y = x)", linewidth=2.5, color="#1f77b4")
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(y_true), size=min(5000, len(y_true)), replace=False)
    plt.scatter(y_true[sample_idx], y_pred[sample_idx], color="#ff7f0e", edgecolors="white", linewidth=0.3, alpha=0.55, s=25)
    plt.title("Model B - Epsilon-Tube Style Prediction Diagnostic")
    plt.xlabel("Actual Revenue")
    plt.ylabel("Predicted Revenue")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_model_b_epsilon_tube_style.png", dpi=160)
    plt.show()


def plot_linear_feature_weights(model_pipeline, output_dir: str | Path | None = None) -> pd.DataFrame | None:
    estimator = model_pipeline.named_steps["model"]
    preprocessor = model_pipeline.named_steps["preprocess"]
    if not hasattr(estimator, "coef_") or not hasattr(preprocessor, "get_feature_names_out"):
        print("Final model does not expose linear coefficients; skipping feature-weight plot.")
        return None

    feature_names = preprocessor.get_feature_names_out()
    coef_df = pd.DataFrame({"feature": feature_names, "weight": estimator.coef_})
    coef_df["abs_weight"] = coef_df["weight"].abs()
    top_coef = coef_df.sort_values("abs_weight", ascending=False).head(15).sort_values("abs_weight")
    plt.figure(figsize=(11, 7))
    colors = np.where(top_coef["weight"] >= 0, "#54a24b", "#e45756")
    plt.barh(top_coef["feature"], top_coef["abs_weight"], color=colors)
    plt.title("Top 15 Absolute Feature Weights - Final Ridge Model")
    plt.xlabel("Absolute Weight")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if output_dir:
        plt.savefig(ensure_dir(output_dir) / "notebook_final_ridge_feature_weights.png", dpi=160)
    plt.show()
    return coef_df.sort_values("abs_weight", ascending=False).head(15).round(4)
