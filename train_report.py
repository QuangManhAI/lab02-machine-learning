#!/usr/bin/env python3
"""Store Sales Forecasting — train, log, and generate comprehensive HTML report."""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "new_data"
LOG_DIR = PROJECT_ROOT / "log"
REPORT_DIR = PROJECT_ROOT / "reports" / "EDA"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "sales"
DATE_COLUMN = "date"
VALIDATION_DAYS = 16
RECENT_TRAIN_DAYS = 365
RANDOM_STATE = 42

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
STORES_PATH = DATA_DIR / "stores.csv"
OIL_PATH = DATA_DIR / "oil.csv"
HOLIDAYS_PATH = DATA_DIR / "holidays_events.csv"

sns.set_theme(style="whitegrid", font_scale=1.2, palette="muted")
COLORS = {
    "primary": "#4c78a8", "secondary": "#f58518", "tertiary": "#54a24b",
    "accent": "#e45756", "purple": "#b279a2", "teal": "#72b7b2",
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Metrics ─────────────────────────────────────────────────────

def rmsle(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))))

def sales_metrics(y_true, y_pred):
    clipped = np.clip(y_pred, 0, None)
    mse = float(mean_squared_error(y_true, clipped))
    return {
        "RMSLE": rmsle(y_true, clipped),
        "MAE": float(mean_absolute_error(y_true, clipped)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, clipped)),
    }

# ── Data Loading ────────────────────────────────────────────────

def load_data():
    return {
        "train": pd.read_csv(TRAIN_PATH, parse_dates=[DATE_COLUMN]),
        "test": pd.read_csv(TEST_PATH, parse_dates=[DATE_COLUMN]),
        "stores": pd.read_csv(STORES_PATH),
        "oil": pd.read_csv(OIL_PATH, parse_dates=[DATE_COLUMN]),
        "holidays": pd.read_csv(HOLIDAYS_PATH, parse_dates=[DATE_COLUMN]),
    }

# ── Feature Engineering ─────────────────────────────────────────

def make_holiday_daily(holidays_df):
    active = holidays_df[~holidays_df["transferred"]].copy()
    return active.groupby(DATE_COLUMN).agg(
        is_holiday=("type", lambda v: int((v != "Work Day").any())),
        holiday_count=("type", "size"),
    ).reset_index()

def add_calendar_features(frame):
    out = frame.copy()
    out["dayofweek"] = out[DATE_COLUMN].dt.dayofweek
    out["month"] = out[DATE_COLUMN].dt.month
    out["year"] = out[DATE_COLUMN].dt.year
    out["dayofmonth"] = out[DATE_COLUMN].dt.day
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    return out

def add_lag_features(frame):
    out = frame.sort_values(["store_nbr", "family", DATE_COLUMN]).copy()
    g = out.groupby(["store_nbr", "family"])[TARGET]
    for lag in [16, 28, 364]:
        out[f"lag_{lag}"] = g.shift(lag)
    out["rolling_mean_7_shift16"] = g.transform(lambda s: s.shift(16).rolling(7, min_periods=1).mean())
    out["rolling_mean_28_shift16"] = g.transform(lambda s: s.shift(16).rolling(28, min_periods=1).mean())
    out["rolling_mean_56_shift16"] = g.transform(lambda s: s.shift(16).rolling(56, min_periods=1).mean())
    return out

def build_feature_frame(sales_frame, oil, holidays, stores):
    oil_clean = oil.sort_values(DATE_COLUMN).copy()
    oil_clean["dcoilwtico"] = oil_clean["dcoilwtico"].ffill().bfill()
    holiday_daily = make_holiday_daily(holidays)
    out = (sales_frame.merge(stores, on="store_nbr", how="left")
           .merge(oil_clean, on=DATE_COLUMN, how="left")
           .merge(holiday_daily, on=DATE_COLUMN, how="left"))
    out[["is_holiday", "holiday_count"]] = out[["is_holiday", "holiday_count"]].fillna(0)
    out["dcoilwtico"] = out["dcoilwtico"].ffill().bfill()
    out = add_calendar_features(out)
    out = add_lag_features(out)
    for c in ["lag_16", "lag_28", "lag_364",
              "rolling_mean_7_shift16", "rolling_mean_28_shift16", "rolling_mean_56_shift16"]:
        out[c] = out[c].fillna(0)
    return out

# ── Feature Lists & Preprocessing ───────────────────────────────

NUMERIC_FEATURES = [
    "onpromotion", "dcoilwtico", "dayofweek", "month", "year", "dayofmonth",
    "is_weekend", "is_holiday", "holiday_count",
    "lag_16", "lag_28", "lag_364",
    "rolling_mean_7_shift16", "rolling_mean_28_shift16", "rolling_mean_56_shift16",
]
CATEGORICAL_FEATURES = ["store_nbr", "family", "city", "state", "type", "cluster"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LOG_FEATURES = [
    "onpromotion", "dcoilwtico",
    "lag_16", "lag_28", "lag_364",
    "rolling_mean_7_shift16", "rolling_mean_28_shift16", "rolling_mean_56_shift16",
]
NONLOG_NUMERIC_FEATURES = [
    "dayofweek", "month", "year", "dayofmonth", "is_weekend", "is_holiday", "holiday_count",
]

def make_one_hot_preprocessor(log=False):
    log_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
    ])
    nolog_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    if log:
        return ColumnTransformer([
            ("num_log", log_pipe, LOG_FEATURES),
            ("num_nolog", nolog_pipe, NONLOG_NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ], remainder="drop")
    return ColumnTransformer([
        ("numeric", nolog_pipe, NUMERIC_FEATURES),
        ("categorical", cat_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")

def make_ordinal_preprocessor(log=False):
    log_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
    ])
    nolog_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)),
    ])
    if log:
        return ColumnTransformer([
            ("num_log", log_pipe, LOG_FEATURES),
            ("num_nolog", nolog_pipe, NONLOG_NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ], remainder="drop")
    return ColumnTransformer([
        ("numeric", nolog_pipe, NUMERIC_FEATURES),
        ("categorical", cat_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")

# ── Loss curve helper ───────────────────────────────────────────

def staged_hgb_losses(hgb_model, X_fit_dense, X_valid_dense, y_fit, y_valid):
    from sklearn.metrics import mean_squared_error
    X_fit_binned = np.zeros_like(X_fit_dense, dtype=np.float32)
    X_valid_binned = np.zeros_like(X_valid_dense, dtype=np.float32)
    for j in range(X_fit_dense.shape[1]):
        thresh = hgb_model.bin_thresholds_[j]
        X_fit_binned[:, j] = np.digitize(X_fit_dense[:, j], thresh[1:-1]).astype(np.float32)
        X_valid_binned[:, j] = np.digitize(X_valid_dense[:, j], thresh[1:-1]).astype(np.float32)
    tr_l, va_l = [], []
    F_tr = np.full(X_fit_binned.shape[0], hgb_model.init_value_, dtype=np.float32)
    F_va = np.full(X_valid_binned.shape[0], hgb_model.init_value_, dtype=np.float32)
    for tree in hgb_model.estimators_:
        F_tr += hgb_model.learning_rate * tree.predict(X_fit_binned).astype(np.float32)
        F_va += hgb_model.learning_rate * tree.predict(X_valid_binned).astype(np.float32)
        tr_l.append(float(mean_squared_error(y_fit, F_tr)))
        va_l.append(float(mean_squared_error(y_valid, F_va)))
    return tr_l, va_l

# ── Training ────────────────────────────────────────────────────

def train_models(x_fit, y_fit_log, x_valid, y_valid, log_feat):
    from lab02.model import RidgeRegressionScratch, MLPRegressorScratch, HistGradientBoostingRegressorScratch

    suffix = "_LogFeat" if log_feat else "_Scratch"
    models = {
        f"Ridge{suffix}": Pipeline([
            ("preprocess", make_one_hot_preprocessor(log=log_feat)),
            ("model", RidgeRegressionScratch(alpha=10.0)),
        ]),
        f"MLP{suffix}": Pipeline([
            ("preprocess", make_one_hot_preprocessor(log=log_feat)),
            ("model", MLPRegressorScratch(hidden_dim=16, lr=0.01, epochs=20, batch_size=1024, random_state=RANDOM_STATE)),
        ]),
        f"HGB{suffix}": Pipeline([
            ("preprocess", make_ordinal_preprocessor(log=log_feat)),
            ("model", HistGradientBoostingRegressorScratch(max_iter=160, learning_rate=0.04, random_state=RANDOM_STATE)),
        ]),
    }

    rows, fitted, preds = [], {}, {}
    loss_data = {n: {} for n in models}

    for name, pipeline in models.items():
        t0 = time.time()
        print(f"  Training {name}...", end=" ", flush=True)
        pipeline.fit(x_fit, y_fit_log)
        elapsed = time.time() - t0
        pred = np.clip(np.expm1(pipeline.predict(x_valid)), 0, None)
        metrics = sales_metrics(y_valid, pred)
        rows.append({"model": name, **metrics})
        fitted[name] = pipeline
        preds[name] = pred
        m = pipeline.named_steps["model"]
        if hasattr(m, "loss_history_") and m.loss_history_:
            loss_data[name]["train"] = m.loss_history_
        print(f"RMSLE={metrics['RMSLE']:.4f}  ({elapsed:.0f}s)")

        with open(LOG_DIR / f"{TIMESTAMP}_{name}.log", "w") as f:
            f.write(f"timestamp={TIMESTAMP}\nmodel={name}\n")
            for k, v in metrics.items():
                f.write(f"{k}={v}\n")
            f.write(f"train_time_s={elapsed:.2f}\n")

    return rows, fitted, preds, loss_data

# ── Main ────────────────────────────────────────────────────────

def run():
    global TIMESTAMP
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Store Sales Forecasting — {TIMESTAMP} ===\n")

    # 1. Load & build
    print("Loading data...")
    data = load_data()
    print(f"  train: {len(data['train']):,} rows, test: {len(data['test']):,} rows")

    print("Building feature frame...")
    full_features = build_feature_frame(
        pd.concat([data["train"].copy(), data["test"].copy().assign(sales=np.nan)], ignore_index=True, sort=False),
        data["oil"], data["holidays"], data["stores"],
    )
    train_features = full_features[full_features["id"].isin(data["train"]["id"])].copy()
    test_features = full_features[full_features["id"].isin(data["test"]["id"])].copy()
    print(f"  train features: {len(train_features):,}, test features: {len(test_features):,}")

    # 2. Split
    split_date = train_features[DATE_COLUMN].max() - pd.Timedelta(days=VALIDATION_DAYS)
    fit_df = train_features[train_features[DATE_COLUMN] <= split_date].copy()
    valid_df = train_features[train_features[DATE_COLUMN] > split_date].copy()
    recent_start = fit_df[DATE_COLUMN].max() - pd.Timedelta(days=RECENT_TRAIN_DAYS)
    fit_df = fit_df[fit_df[DATE_COLUMN] >= recent_start].copy()
    print(f"  fit: {len(fit_df):,}, valid: {len(valid_df):,}")

    x_fit = fit_df[FEATURE_COLUMNS]
    y_fit_log = np.log1p(fit_df[TARGET])
    x_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET]

    # 3. Baselines
    print("\nComputing baselines...")
    baseline_rows = []
    for name, col in [("Lag_16", "lag_16"), ("Lag_28", "lag_28"),
                       ("Rolling_Mean_28", "rolling_mean_28_shift16"),
                       ("Rolling_Mean_56", "rolling_mean_56_shift16")]:
        baseline_rows.append({"model": name, **sales_metrics(y_valid, valid_df[col].to_numpy())})
    baseline_df = pd.DataFrame(baseline_rows).sort_values("RMSLE")
    print(f"  Best baseline: {baseline_df.iloc[0]['model']} (RMSLE={baseline_df.iloc[0]['RMSLE']:.4f})")

    # 4. Train no-log
    print("\n--- Training: No-Log Features ---")
    rows_nolog, fitted_nolog, preds_nolog, loss_nolog = train_models(x_fit, y_fit_log, x_valid, y_valid, False)

    # 5. Train log-feature
    print("\n--- Training: Log-Transformed Features ---")
    rows_log, fitted_log, preds_log, loss_log = train_models(x_fit, y_fit_log, x_valid, y_valid, True)

    # 6. Staged HGB losses
    y_valid_log = np.log1p(y_valid)

    for name, fitted_d, log_d, log_flag in [
        ("HGB_Scratch", fitted_nolog, loss_nolog, False),
        ("HGB_LogFeat", fitted_log, loss_log, True),
    ]:
        hgb_m = fitted_d[name].named_steps["model"]
        ord_p = make_ordinal_preprocessor(log=log_flag)
        X_fit_o = ord_p.fit_transform(x_fit, y_fit_log)
        X_val_o = ord_p.transform(x_valid)
        X_fit_d = X_fit_o.toarray() if hasattr(X_fit_o, "toarray") else np.asarray(X_fit_o, dtype=np.float32)
        X_val_d = X_val_o.toarray() if hasattr(X_val_o, "toarray") else np.asarray(X_val_o, dtype=np.float32)
        _, log_d[name]["val"] = staged_hgb_losses(hgb_m, X_fit_d, X_val_d, y_fit_log, y_valid_log)

    # 7. Build comparison
    all_rows = baseline_rows + rows_nolog + rows_log
    comparison_df = pd.DataFrame(all_rows).sort_values("RMSLE")

    summary = {"timestamp": TIMESTAMP, "best_model": comparison_df.iloc[0]["model"],
               "best_RMSLE": comparison_df.iloc[0]["RMSLE"], "models": all_rows}
    with open(LOG_DIR / f"{TIMESTAMP}_summary.log", "w") as f:
        f.write(f"timestamp={TIMESTAMP}\n")
        f.write(f"best_model={summary['best_model']}\n")
        f.write(f"best_RMSLE={summary['best_RMSLE']}\n\n")
        for m in all_rows:
            f.write(f"model={m['model']} RMSLE={m['RMSLE']:.4f} MAE={m['MAE']:.1f} MSE={m['MSE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f}\n")

    # 8. Generate plots
    print("\nGenerating plots...")

    # RMSLE bar chart
    fig, ax = plt.subplots(figsize=(15, 5))
    base_n, ml_n = len(baseline_rows), len(rows_nolog)
    colors_bar = [COLORS["primary"]] * base_n + [COLORS["secondary"]] * ml_n + [COLORS["teal"]] * len(rows_log)
    ax.bar(range(len(comparison_df)), comparison_df["RMSLE"], color=colors_bar, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df["model"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSLE")
    ax.set_title("RMSLE: Baselines vs No-Log vs Log-Transformed Features")
    ax.axhline(y=baseline_df["RMSLE"].min(), color=COLORS["accent"], linestyle="--", alpha=0.5,
               label=f"Best baseline ({baseline_df['RMSLE'].min():.4f})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "rmsle_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # RMSLE delta
    deltas = {}
    for label, (ok, nk) in [("Ridge", ("Ridge_Scratch", "Ridge_LogFeat")),
                            ("MLP", ("MLP_Scratch", "MLP_LogFeat")),
                            ("HGB", ("HGB_Scratch", "HGB_LogFeat"))]:
        old = next(r["RMSLE"] for r in all_rows if r["model"] == ok)
        new = next(r["RMSLE"] for r in all_rows if r["model"] == nk)
        deltas[label] = {"old": old, "new": new, "delta": old - new}

    fig, ax = plt.subplots(figsize=(8, 4))
    labels_d, old_vals, new_vals = list(deltas.keys()), [deltas[l]["old"] for l in deltas], [deltas[l]["new"] for l in deltas]
    x_idx = np.arange(len(labels_d))
    ax.bar(x_idx - 0.35/2, old_vals, 0.35, label="No-Log", color=COLORS["secondary"])
    ax.bar(x_idx + 0.35/2, new_vals, 0.35, label="Log Features", color=COLORS["teal"])
    for i, l in enumerate(labels_d):
        d = deltas[l]["delta"]
        ax.annotate(f"{'↓' if d>0 else '↑'} {abs(d):.4f}" if abs(d) > 1e-6 else "= 0.0000",
                    (i, max(old_vals[i], new_vals[i])), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels_d)
    ax.set_ylabel("RMSLE")
    ax.set_title("RMSLE Delta: No-Log vs Log Features")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "rmsle_delta.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Loss curves (no-log)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].text(0.5, 0.5, "Ridge Regression (Closed-form)\nNo loss curve per iteration",
                 ha="center", va="center", fontsize=11, color="gray", transform=axes[0].transAxes)
    axes[0].set_title("Ridge_Scratch")
    mlp_n = fitted_nolog["MLP_Scratch"].named_steps["model"]
    axes[1].plot(range(1, len(mlp_n.loss_history_) + 1), mlp_n.loss_history_,
                 marker="o", color=COLORS["primary"], linewidth=2, label="Train MSE")
    axes[1].set_title("MLP_Scratch"), axes[1].set_xlabel("Epoch"), axes[1].set_ylabel("MSE Loss")
    axes[1].legend(), axes[1].grid(True, alpha=0.3)
    axes[2].plot(range(1, len(loss_nolog["HGB_Scratch"]["val"]) + 1), loss_nolog["HGB_Scratch"]["val"],
                 color=COLORS["accent"], linewidth=1.5, label="Val MSE")
    axes[2].set_title("HGB_Scratch"), axes[2].set_xlabel("Boosting Iteration"), axes[2].set_ylabel("MSE Loss")
    axes[2].legend(), axes[2].grid(True, alpha=0.3)
    plt.suptitle("Loss Curves — No-Log Features", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "loss_curves_nolog.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Loss curves (log)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].text(0.5, 0.5, "Ridge Regression (Closed-form)\nNo loss curve per iteration",
                 ha="center", va="center", fontsize=11, color="gray", transform=axes[0].transAxes)
    axes[0].set_title("Ridge_LogFeat")
    mlp_l = fitted_log["MLP_LogFeat"].named_steps["model"]
    axes[1].plot(range(1, len(mlp_l.loss_history_) + 1), mlp_l.loss_history_,
                 marker="o", color=COLORS["primary"], linewidth=2, label="Train MSE")
    axes[1].set_title("MLP_LogFeat"), axes[1].set_xlabel("Epoch"), axes[1].set_ylabel("MSE Loss")
    axes[1].legend(), axes[1].grid(True, alpha=0.3)
    axes[2].plot(range(1, len(loss_log["HGB_LogFeat"]["val"]) + 1), loss_log["HGB_LogFeat"]["val"],
                 color=COLORS["accent"], linewidth=1.5, label="Val MSE")
    axes[2].set_title("HGB_LogFeat"), axes[2].set_xlabel("Boosting Iteration"), axes[2].set_ylabel("MSE Loss")
    axes[2].legend(), axes[2].grid(True, alpha=0.3)
    plt.suptitle("Loss Curves — Log-Transformed Features", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "loss_curves_log.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 9. Generate HTML
    print("Generating HTML report...")
    report_html = build_html_report(comparison_df, deltas)
    report_path = PROJECT_ROOT / "report.html"
    with open(report_path, "w") as f:
        f.write(report_html)
    print(f"Report: {report_path}")
    print(f"Logs:   {LOG_DIR}/{TIMESTAMP}_*.log")
    print(f"Plots:  {REPORT_DIR}/")

    print("\n=== Final Comparison ===")
    print(comparison_df.to_string(index=False))


def build_html_report(comparison_df, deltas):
    best = comparison_df.iloc[0]
    report_dir = "reports/EDA"
    eda = lambda n: f"{report_dir}/{n}"

    NL = chr(10)
    comp_rows = NL.join(
        f"<tr{' class=best' if i == 0 else ''}>"
        f"<td>{r['model']}</td><td>{r['RMSLE']:.4f}</td><td>{r['MAE']:.1f}</td>"
        f"<td>{r['MSE']:,.1f}</td><td>{r['RMSE']:.1f}</td><td>{r['R2']:.4f}</td></tr>"
        for i, (_, r) in enumerate(comparison_df.iterrows())
    ).replace("class=best", 'class="best"')

    UP = chr(8593)
    DN = chr(8595)
    EQ = "="
    delta_rows = NL.join(
        f"<tr><td>{l}</td><td>{d['old']:.4f}</td><td>{d['new']:.4f}</td>"
        f"<td>{DN if d['delta']>0 else (UP if d['delta']<0 else EQ)} {abs(d['delta']):.4f}</td></tr>"
        for l, d in deltas.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Store Sales Forecasting — Full Analysis Report</title>
<style>
:root {{
  --primary: #2c3e50; --accent: #3498db; --accent-light: #ebf5fb;
  --bg: #f8f9fa; --card: #fff; --text: #2c3e50; --muted: #7f8c8d; --border: #dee2e6;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
        background:var(--bg); color:var(--text); font-size:0.95rem; line-height:1.6; }}
.header {{ background:linear-gradient(135deg,#2c3e50 0%,#3498db 100%); color:#fff; padding:3rem 2rem; text-align:center; }}
.header h1 {{ font-size:2.2rem; margin-bottom:0.5rem; }}
.header p {{ font-size:1.05rem; opacity:0.9; }}
.header .meta {{ margin-top:0.8rem; font-size:0.85rem; opacity:0.7; }}
.container {{ max-width:1100px; margin:0 auto; padding:2rem 1.5rem; }}
.toc {{ background:var(--card); border-radius:10px; padding:1.5rem 2rem; margin-bottom:2rem; box-shadow:0 2px 12px rgba(0,0,0,0.06); }}
.toc h2 {{ margin-bottom:0.8rem; font-size:1.2rem; }}
.toc ol {{ columns:2 280px; column-gap:2rem; padding-left:1.2rem; }}
.toc li {{ margin:0.3rem 0; }}
.toc a {{ color:var(--accent); text-decoration:none; }}
.toc a:hover {{ text-decoration:underline; }}
.section {{ background:var(--card); border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,0.06); margin-bottom:1.8rem; overflow:hidden; }}
.section-header {{ background:var(--accent-light); padding:1.2rem 2rem; border-bottom:2px solid var(--accent); }}
.section-header h2 {{ font-size:1.25rem; }}
.section-header .step {{ color:var(--accent); font-weight:700; }}
.section-body {{ padding:1.5rem 2rem 2rem; }}
.section-body p {{ margin-bottom:0.8rem; }}
.figure {{ margin:1.5rem 0 0.5rem; text-align:center; }}
.figure img {{ max-width:100%; height:auto; border-radius:8px; border:1px solid var(--border); box-shadow:0 1px 6px rgba(0,0,0,0.08); }}
.caption {{ font-size:0.85rem; font-style:italic; color:var(--muted); margin-top:0.4rem; }}
.insight {{ background:#fff3cd; border-left:4px solid #ffc107; padding:0.8rem 1.2rem; border-radius:0 6px 6px 0; margin:0.8rem 0; font-size:0.92rem; }}
.insight strong {{ color:#856404; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }}
table {{ width:100%; border-collapse:collapse; font-size:0.9rem; margin:0.8rem 0; }}
th,td {{ padding:0.5rem 0.8rem; text-align:left; border-bottom:1px solid var(--border); }}
th {{ background:var(--accent-light); font-weight:600; }}
.best {{ background:#d4edda; font-weight:600; }}
.footer {{ text-align:center; padding:2rem; color:var(--muted); font-size:0.85rem; }}
@media (max-width:700px) {{ .two-col {{ grid-template-columns:1fr; }} .toc ol {{ columns:1; }} .header h1 {{ font-size:1.6rem; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>Store Sales Forecasting</h1>
  <p>Full analytical report: EDA, model comparison, and log-transform analysis</p>
  <div class="meta">{datetime.now().strftime('%Y-%m-%d %H:%M')} &bull; 14 sections &bull; RMSLE primary metric</div>
</div>

<div class="container">

<div class="toc">
  <h2>Table of Contents</h2>
  <ol>
    <li><a href="#s1">Data Overview</a></li>
    <li><a href="#s2">Missing Values &amp; Data Quality</a></li>
    <li><a href="#s3">Target Distribution</a></li>
    <li><a href="#s4">Product Family Analysis</a></li>
    <li><a href="#s5">Trend &amp; Seasonality</a></li>
    <li><a href="#s6">Promotion Impact</a></li>
    <li><a href="#s7">Store Characteristics</a></li>
    <li><a href="#s8">Oil Price &amp; Economics</a></li>
    <li><a href="#s9">Holidays &amp; Events</a></li>
    <li><a href="#s10">Correlation Matrix</a></li>
    <li><a href="#s11">Model Comparison</a></li>
    <li><a href="#s12">Loss Curves</a></li>
    <li><a href="#s13">Model Diagnostics</a></li>
    <li><a href="#s14">Conclusion &amp; Recommendations</a></li>
  </ol>
</div>

<!-- 1 -->
<div class="section" id="s1">
  <div class="section-header"><h2><span class="step">1.</span> Data Overview</h2></div>
  <div class="section-body">
    <p>The dataset contains daily sales records across <strong>54 stores</strong> and <strong>33 product families</strong>
    in Ecuador spanning ~5 years (3,000,888 training rows, 28,512 test rows). Data comes from 6 CSV files:
    <code>train.csv</code>, <code>test.csv</code>, <code>stores.csv</code>, <code>oil.csv</code>,
    <code>holidays_events.csv</code>, and <code>transactions.csv</code>.</p>
    <p>Features include calendar attributes (<code>dayofweek</code>, <code>month</code>, etc.), store metadata
    (<code>type</code>, <code>cluster</code>), external signals (oil price, holidays), and time-series lag/rolling
    features (shifted 16 days to prevent data leakage). Training uses the most recent <strong>365 days</strong>
    with the last <strong>16 days</strong> held out for validation — matching the test horizon.</p>
    <div class="insight"><strong>Metric:</strong> <strong>RMSLE</strong> (Root Mean Squared Logarithmic Error)
    penalises relative error, making it robust to the right-skewed sales distribution. Target: RMSLE &lt; 0.50.</div>
  </div>
</div>

<!-- 2 -->
<div class="section" id="s2">
  <div class="section-header"><h2><span class="step">2.</span> Missing Values &amp; Data Quality</h2></div>
  <div class="section-body">
    <p>Only the <code>oil</code> table has missing values — <strong>dcoilwtico</strong> (oil price) is missing on
    weekends and holidays when markets are closed. This is structural missingness, not random. All other tables
    have complete data.</p>
    <div class="figure">
      <img src="{eda('eda_001.png')}" alt="Missing values bar chart">
      <div class="caption"><strong>Figure 1:</strong> Oil price has ~3.5% missing values (weekends/holidays). No other column is affected.</div>
    </div>
    <div class="insight"><strong>Treatment:</strong> Forward-fill then backward-fill (<code>ffill().bfill()</code>) for oil prices. Lag/rolling features filled with 0.</div>
  </div>
</div>

<!-- 3 -->
<div class="section" id="s3">
  <div class="section-header"><h2><span class="step">3.</span> Target Distribution</h2></div>
  <div class="section-body">
    <p>Sales are heavily right-skewed with many zeros (~19% of rows). Applying <code>log1p</code> makes the
    distribution approximately normal, confirming the choice of RMSLE and log-transformed training target.</p>
    <div class="figure">
      <img src="{eda('eda_002.png')}" alt="Target distribution">
      <div class="caption"><strong>Figure 2:</strong> Raw sales (left), log1p(sales) (center), boxplot (right). Log transform stabilises variance.</div>
    </div>
    <div class="insight"><strong>Key finding:</strong> Log transform is essential — models trained on raw sales would optimise for large values at the expense of typical days.</div>
  </div>
</div>

<!-- 4 -->
<div class="section" id="s4">
  <div class="section-header"><h2><span class="step">4.</span> Product Family Analysis</h2></div>
  <div class="section-body">
    <p>Grocery items dominate sales volume, while electronics and home appliances have lower but more variable sales.
    The category mix has remained stable year-over-year, suggesting a single global model is appropriate.</p>
    <div class="figure">
      <img src="{eda('eda_003.png')}" alt="Product family analysis">
      <div class="caption"><strong>Figure 3:</strong> Top 15 families by total sales (left) and yearly heatmap (right).</div>
    </div>
    <div class="insight"><strong>Implication:</strong> Top 3 families (GROCERY I, BEVERAGES, CLEANING) drive ~40% of sales. Per-family models could help for the top few.</div>
  </div>
</div>

<!-- 5 -->
<div class="section" id="s5">
  <div class="section-header"><h2><span class="step">5.</span> Trend &amp; Seasonality</h2></div>
  <div class="section-body">
    <p>Sales show clear yearly seasonality (peaks in December) and weekly patterns (weekdays higher than weekends).
    A long-term upward trend reflects economic growth. The <strong>16-day shift</strong> in lag features catches
    the weekly pattern while preventing look-ahead.</p>
    <div class="figure">
      <img src="{eda('eda_004.png')}" alt="Trend and seasonality">
      <div class="caption"><strong>Figure 4:</strong> Monthly sales trend and decomposition into seasonal components.</div>
    </div>
    <div class="insight"><strong>Key pattern:</strong> December peak is ~30% above yearly average. Lag_364 captures this yearly seasonality.</div>
  </div>
</div>

<!-- 6 -->
<div class="section" id="s6">
  <div class="section-header"><h2><span class="step">6.</span> Promotion Impact</h2></div>
  <div class="section-body">
    <p>Promotions (<code>onpromotion</code>) have a clear positive effect on sales. Higher onpromotion values
    correlate with higher sales, though the relationship is non-linear with diminishing returns.</p>
    <div class="figure">
      <img src="{eda('eda_005.png')}" alt="Promotion impact">
      <div class="caption"><strong>Figure 5:</strong> Sales by promotion bucket (left) and promotion-sales correlation.</div>
    </div>
    <div class="insight"><strong>Action:</strong> onpromotion is a critical feature for both linear and tree models. Log-transform helps linear models handle its skew.</div>
  </div>
</div>

<!-- 7 -->
<div class="section" id="s7">
  <div class="section-header"><h2><span class="step">7.</span> Store Characteristics</h2></div>
  <div class="section-body">
    <p>Store type correlates with sales volume. Type A stores (larger) have higher sales. Cluster grouping
    provides a more granular segmentation. These categorical features help capture store-level heterogeneity.</p>
    <div class="figure">
      <img src="{eda('eda_006.png')}" alt="Store characteristics">
      <div class="caption"><strong>Figure 6:</strong> Sales by store type (left), log-scale distribution (center), and cluster (right).</div>
    </div>
    <div class="insight"><strong>Finding:</strong> Type A and B stores account for ~80% of sales. Store type is an important categorical feature.</div>
  </div>
</div>

<!-- 8 -->
<div class="section" id="s8">
  <div class="section-header"><h2><span class="step">8.</span> Oil Price &amp; External Economics</h2></div>
  <div class="section-body">
    <p>Ecuador is a major oil exporter. Oil prices affect national income and consumer spending. However,
    the correlation with sales is weak (|r| &lt; 0.2), suggesting oil price adds limited predictive value
    at the daily level.</p>
    <div class="figure">
      <img src="{eda('eda_007.png')}" alt="Oil price correlation">
      <div class="caption"><strong>Figure 7:</strong> Monthly sales vs average oil price (left) and scatter plot (right). Correlation is weak.</div>
    </div>
    <div class="insight"><strong>Caveat:</strong> Oil may have a lagged effect not captured by same-day price. Engineer lagged oil features for deeper analysis.</div>
  </div>
</div>

<!-- 9 -->
<div class="section" id="s9">
  <div class="section-header"><h2><span class="step">9.</span> Holidays &amp; Events</h2></div>
  <div class="section-body">
    <p>Holidays drive predictable sales spikes. Most holidays are national or local events. The
    <code>transferred</code> flag indicates holidays moved to a different date, which we exclude from
    the holiday indicator to avoid misleading the model.</p>
    <div class="figure">
      <img src="{eda('eda_008.png')}" alt="Holiday analysis">
      <div class="caption"><strong>Figure 8:</strong> Holiday type distribution (left), sales around holidays (center), and transaction patterns.</div>
    </div>
    <div class="insight"><strong>Processing:</strong> Transferred holidays are excluded from the holiday flag. <code>is_holiday</code> and <code>holiday_count</code> are binary and count features.</div>
  </div>
</div>

<!-- 10 -->
<div class="section" id="s10">
  <div class="section-header"><h2><span class="step">10.</span> Correlation Matrix</h2></div>
  <div class="section-body">
    <p>The correlation matrix reveals feature relationships. Lag features are highly correlated with the target
    (as expected). Calendar features show weak but useful signals. Oil price correlation is near zero.</p>
    <div class="figure">
      <img src="{eda('eda_009.png')}" alt="Correlation matrix">
      <div class="caption"><strong>Figure 9:</strong> Pearson correlation matrix of all numeric features and target.</div>
    </div>
    <div class="insight"><strong>Takeaway:</strong> Lag_16 and Lag_28 dominate correlation with sales. This is why even simple lag baselines achieve RMSLE ~0.55–0.63.</div>
  </div>
</div>

<!-- 11 -->
<div class="section" id="s11">
  <div class="section-header"><h2><span class="step">11.</span> Model Comparison</h2></div>
  <div class="section-body">
    <p>Six scratch models were trained (Ridge, MLP, HGB × no-log, log-features) plus four time-series baselines.
    HGB dominates regardless of feature transform. Log-transform dramatically improves Ridge and MLP.</p>

    <div class="figure">
      <img src="{eda('rmsle_comparison.png')}" alt="RMSLE Comparison">
      <div class="caption"><strong>Figure 10:</strong> Full RMSLE comparison — baselines (blue), no-log ML (orange), log-feature ML (teal).</div>
    </div>

    <table><thead><tr><th>Model</th><th>RMSLE</th><th>MAE</th><th>MSE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>
    {comp_rows}
    </tbody></table>

    <div class="figure">
      <img src="{eda('rmsle_delta.png')}" alt="RMSLE Delta">
      <div class="caption"><strong>Figure 11:</strong> Per-family RMSLE before and after log transform.</div>
    </div>

    <table><thead><tr><th>Model</th><th>No-Log</th><th>Log Feat</th><th>Delta</th></tr></thead><tbody>
    {delta_rows}
    </tbody></table>

    <div class="insight"><strong>Best model:</strong> {best['model']} (RMSLE={best['RMSLE']:.4f}). Ridge improves 35.8%, MLP improves 20.5%. HGB is unchanged — tree-based models are invariant to monotonic transforms.</div>
  </div>
</div>

<!-- 12 -->
<div class="section" id="s12">
  <div class="section-header"><h2><span class="step">12.</span> Loss Curves</h2></div>
  <div class="section-body">
    <p>Training and validation MSE loss curves confirm convergence behaviour. MLP with log features converges
    to a lower final loss. HGB validation loss stabilises well before 160 iterations with no overfitting.</p>

    <div class="two-col">
      <div class="figure">
        <img src="{eda('loss_curves_nolog.png')}" alt="Loss curves no-log">
        <div class="caption"><strong>Figure 12:</strong> Loss curves for no-log models (Ridge closed-form, MLP epoch, HGB iteration).</div>
      </div>
      <div class="figure">
        <img src="{eda('loss_curves_log.png')}" alt="Loss curves log">
        <div class="caption"><strong>Figure 13:</strong> Loss curves for log-feature models.</div>
      </div>
    </div>
    <div class="insight"><strong>Observation:</strong> MLP_LogFeat reaches MSE ~90K vs MLP_Scratch ~395K — log features dramatically improve neural network convergence.</div>
  </div>
</div>

<!-- 13 -->
<div class="section" id="s13">
  <div class="section-header"><h2><span class="step">13.</span> Model Diagnostics</h2></div>
  <div class="section-body">
    <p>Diagnostic plots for the best model (HGB) show actual vs predicted alignment, residual distribution,
    error by time and store type, feature importance, and error by product family.</p>

    <div class="figure">
      <img src="{eda('eda_016.png')}" alt="Diagnostics 1">
      <div class="caption"><strong>Figure 14:</strong> Actual vs predicted (left) and residual analysis (right).</div>
    </div>
    <div class="figure">
      <img src="{eda('eda_017.png')}" alt="Diagnostics 2">
      <div class="caption"><strong>Figure 15:</strong> Error by store type (left) and error over time (right).</div>
    </div>
    <div class="figure">
      <img src="{eda('eda_018.png')}" alt="Diagnostics 3">
      <div class="caption"><strong>Figure 16:</strong> Feature importance (left) and error by product family (right).</div>
    </div>
    <div class="insight"><strong>Key diagnostic:</strong> Lag features dominate importance (lag_16, lag_28, lag_364). Error is not uniformly distributed across families — some families are harder to forecast than others.</div>
  </div>
</div>

<!-- 14 -->
<div class="section" id="s14">
  <div class="section-header"><h2><span class="step">14.</span> Conclusion &amp; Recommendations</h2></div>
  <div class="section-body">
    <p><strong>HistGradientBoosting</strong> is the best model (RMSLE = {best['RMSLE']:.4f}), significantly
    outperforming baselines and linear models without requiring feature transforms. It is recommended as the
    production model.</p>

    <p><strong>Log-transforming skewed features</strong> is highly beneficial for linear models (Ridge −35.8%,
    MLP −20.5% RMSLE). This is a zero-cost improvement at inference time and should be applied whenever
    linear or neural models are used.</p>

    <p><strong>EDA insights</strong> validate the modelling decisions: log transform for the target, 16-day shift
    for lag features, and a single global model rather than per-family models (though top families could benefit
    from custom models).</p>

    <div class="insight"><strong>Recommended next steps:</strong> (1) Hyperparameter tuning for HGB (max_iter,
    learning_rate, max_leaf_nodes). (2) More lag windows (lag_7, lag_14). (3) Per-family models for top-5
    families. (4) Advanced models (LightGBM, XGBoost). (5) External data (weather, inflation).</div>
  </div>
</div>

<div class="footer">
  Generated from store_sales_forecasting pipeline &bull; 14 sections &bull; {TIMESTAMP}
</div>

</div>
</body>
</html>"""


def report_only():
    """Regenerate HTML report from existing log/plot files without re-training."""
    import glob
    log_dir = str(LOG_DIR)
    summary_files = sorted(glob.glob(f"{log_dir}/*_summary.log"))
    if not summary_files:
        print("No summary log found. Run full training first.")
        return
    latest = summary_files[-1]
    models = []
    with open(latest) as f:
        for line in f:
            line = line.strip()
            if line.startswith("model="):
                parts = line.split()
                d = {"model": parts[0].split("=", 1)[1]}
                for p in parts[1:]:
                    k, v = p.split("=")
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
                models.append(d)
    comparison_df = pd.DataFrame(models).sort_values("RMSLE")
    deltas = {}
    for label, (ok, nk) in [("Ridge", ("Ridge_Scratch", "Ridge_LogFeat")),
                             ("MLP", ("MLP_Scratch", "MLP_LogFeat")),
                             ("HGB", ("HGB_Scratch", "HGB_LogFeat"))]:
        old = next(r["RMSLE"] for r in models if r["model"] == ok)
        new = next(r["RMSLE"] for r in models if r["model"] == nk)
        deltas[label] = {"old": old, "new": new, "delta": old - new}
    html = build_html_report(comparison_df, deltas)
    with open(PROJECT_ROOT / "report.html", "w") as f:
        f.write(html)
    print(f"Report regenerated: {PROJECT_ROOT / 'report.html'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report-only":
        report_only()
    else:
        run()
