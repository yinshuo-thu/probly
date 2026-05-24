"""
Task 4 - Part 2: Train ML models on the built dataset.

Uses TimeSeriesSplit to avoid leakage.
Trains: LogisticRegression, RandomForest, XGBoost, LightGBM
Saves: metrics_history.json, test_predictions.parquet, feature_importance_xgb.csv
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = "outputs/training_dataset.parquet"
METRICS_PATH = "outputs/metrics_history.json"
PREDICTIONS_PATH = "outputs/test_predictions.parquet"
MODELS_DIR = "outputs/models"
FEAT_IMP_PATH = "outputs/models/feature_importance_xgb.csv"

os.makedirs(MODELS_DIR, exist_ok=True)


FEATURE_COLS = [
    # Rolling event features
    "event_density_30s", "event_density_60s", "event_density_120s",
    "weight_sum_30s", "weight_sum_60s", "weight_sum_120s",
    "conf_mean_30s", "conf_mean_60s", "conf_mean_120s",
    "conf_min_30s", "conf_min_60s", "conf_min_120s",
    "shot_on_target_30s", "shot_on_target_60s", "shot_on_target_120s",
    "dangerous_attack_30s", "dangerous_attack_60s", "dangerous_attack_120s",
    # Match state
    "score_home", "score_away", "score_diff", "total_goals",
    "seconds_elapsed", "period_id", "minutes_remaining",
    "confidence_grade",
    # Current price context
    "current_price",
]


def load_data():
    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded dataset: {len(df)} rows, {df['fixture_id'].nunique()} fixtures")
    print(f"Label distribution: {df['high_volatility_label'].value_counts().to_dict()}")

    # Sort by timestamp for time-series split
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Keep only available feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Missing feature columns (will skip): {missing}")

    X = df[available].fillna(0).values
    y = df["high_volatility_label"].values.astype(int)
    feature_names = available

    return df, X, y, feature_names


def evaluate(y_true, y_pred, y_proba, name):
    metrics = {
        "model": name,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        "support_positive": int(y_true.sum()),
        "support_total": int(len(y_true)),
        "timestamp": datetime.utcnow().isoformat(),
    }
    print(f"  {name}: F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}, "
          f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
    return metrics


def train_models(X, y, feature_names, df):
    n_splits = min(5, len(np.unique(df["fixture_id"])) // 2)
    n_splits = max(2, min(n_splits, 5))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\nUsing {n_splits}-fold TimeSeriesSplit")
    all_metrics = []

    # ── Logistic Regression ──────────────────────────────────────────────────
    print("\nTraining LogisticRegression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1)),
    ])
    lr_preds = np.zeros(len(y))
    lr_probas = np.zeros(len(y))
    for train_idx, test_idx in tscv.split(X):
        lr_pipe.fit(X[train_idx], y[train_idx])
        lr_preds[test_idx] = lr_pipe.predict(X[test_idx])
        lr_probas[test_idx] = lr_pipe.predict_proba(X[test_idx])[:, 1]
    all_metrics.append(evaluate(y, lr_preds, lr_probas, "LogisticRegression"))

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\nTraining RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf_preds = np.zeros(len(y))
    rf_probas = np.zeros(len(y))
    for train_idx, test_idx in tscv.split(X):
        rf.fit(X[train_idx], y[train_idx])
        rf_preds[test_idx] = rf.predict(X[test_idx])
        rf_probas[test_idx] = rf.predict_proba(X[test_idx])[:, 1]
    all_metrics.append(evaluate(y, rf_preds, rf_probas, "RandomForest"))

    # ── XGBoost ──────────────────────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        print("\nTraining XGBoost...")
        scale_pos_weight = float((y == 0).sum()) / max((y == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist", eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb_preds = np.zeros(len(y))
        xgb_probas = np.zeros(len(y))
        for train_idx, test_idx in tscv.split(X):
            xgb.fit(X[train_idx], y[train_idx], verbose=False)
            xgb_preds[test_idx] = xgb.predict(X[test_idx])
            xgb_probas[test_idx] = xgb.predict_proba(X[test_idx])[:, 1]
        all_metrics.append(evaluate(y, xgb_preds, xgb_probas, "XGBoost"))

        # Feature importance
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp.to_csv(FEAT_IMP_PATH, index=False)
        print(f"\nTop 10 XGBoost features:")
        print(imp.head(10).to_string())

        # Save XGBoost predictions for final test set (last fold)
        last_train, last_test = list(tscv.split(X))[-1]
        xgb_final = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist", eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb_final.fit(X[last_train], y[last_train], verbose=False)
        test_preds = xgb_final.predict(X[last_test])
        test_probas = xgb_final.predict_proba(X[last_test])[:, 1]

        pred_df = df.iloc[last_test].copy()
        pred_df["predicted_label"] = test_preds
        pred_df["predicted_proba"] = test_probas
        pred_df.to_parquet(PREDICTIONS_PATH, index=False)
        print(f"\nTest predictions saved: {len(pred_df)} rows → {PREDICTIONS_PATH}")

    except ImportError:
        print("  XGBoost not installed, skipping")
        xgb_probas = lr_probas  # fallback for prediction save
        last_test = list(tscv.split(X))[-1][1]
        pred_df = df.iloc[last_test].copy()
        pred_df["predicted_label"] = lr_preds[last_test]
        pred_df["predicted_proba"] = lr_probas[last_test]
        pred_df.to_parquet(PREDICTIONS_PATH, index=False)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    try:
        from lightgbm import LGBMClassifier
        print("\nTraining LightGBM...")
        lgb = LGBMClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1,
        )
        lgb_preds = np.zeros(len(y))
        lgb_probas = np.zeros(len(y))
        for train_idx, test_idx in tscv.split(X):
            lgb.fit(X[train_idx], y[train_idx])
            lgb_preds[test_idx] = lgb.predict(X[test_idx])
            lgb_probas[test_idx] = lgb.predict_proba(X[test_idx])[:, 1]
        all_metrics.append(evaluate(y, lgb_preds, lgb_probas, "LightGBM"))
    except ImportError:
        print("  LightGBM not installed, skipping")

    return all_metrics


def main():
    df, X, y, feature_names = load_data()

    if len(np.unique(y)) < 2:
        print("ERROR: Only one class in labels - cannot train classifier")
        print(f"y distribution: {np.unique(y, return_counts=True)}")
        return

    pos_rate = y.mean()
    print(f"\nPositive rate: {pos_rate:.3f}")
    if pos_rate < 0.01:
        print("WARNING: Very low positive rate - model may struggle")

    metrics = train_models(X, y, feature_names, df)

    # Save metrics
    history = []
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            try:
                history = json.load(f)
            except Exception:
                history = []

    history.extend(metrics)
    with open(METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # Best model summary
    if metrics:
        best = max(metrics, key=lambda m: m["f1"])
        print(f"\nBest model: {best['model']} with F1={best['f1']:.4f}")
        print(f"Target: F1 > 0.80 → {'ACHIEVED!' if best['f1'] >= 0.80 else 'not yet, may need more data'}")


if __name__ == "__main__":
    main()
