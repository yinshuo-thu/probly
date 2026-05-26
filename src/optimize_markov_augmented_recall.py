"""
Recall-first optimization with Markov xT explainability features.

This script tests whether the interpretable Markov layer improves the practical
warning model. It writes separate artifacts first, so the current production
`test_predictions.parquet` is only changed manually after comparison.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from markov_xt_model import (
    add_markov_scores,
    add_state_columns,
    build_impact_tables,
    build_transition_model,
    value_iteration,
)
from optimize_recall_models import (
    FEATURE_COLS,
    VIZ_COLS,
    choose_threshold,
    fit_predict,
    metrics_for,
)
from train_models import DATA_FILE, MODEL_DIR, OUT_DIR, TARGET, engineer_features


MARKOV_FEATURES = [
    "markov_state_risk",
    "markov_event_risk",
    "markov_event_move",
    "markov_state_value_raw",
    "markov_path_score",
    "markov_path_confidence",
]


def candidate_specs_augmented():
    """Fast recall candidates for the Markov feature ablation."""
    return [
        {"name": "RF recall shallow", "kind": "rf", "n_estimators": 180, "max_depth": 8, "min_samples_leaf": 3, "class_weight": {0: 1, 1: 3}},
        {"name": "RF recall medium", "kind": "rf", "n_estimators": 220, "max_depth": 11, "min_samples_leaf": 4, "class_weight": {0: 1, 1: 4}},
        {"name": "RF high weight", "kind": "rf", "n_estimators": 260, "max_depth": 12, "min_samples_leaf": 3, "class_weight": {0: 1, 1: 6}},
        {"name": "RF shallow high weight", "kind": "rf", "n_estimators": 220, "max_depth": 7, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 8}},
        {"name": "RF medium recall low leaf", "kind": "rf", "n_estimators": 220, "max_depth": 10, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 7}},
        {"name": "RF medium recall high leaf", "kind": "rf", "n_estimators": 220, "max_depth": 10, "min_samples_leaf": 8, "class_weight": {0: 1, 1: 7}},
        {"name": "ExtraTrees recall", "kind": "et", "n_estimators": 240, "max_depth": 14, "min_samples_leaf": 3, "class_weight": {0: 1, 1: 4}},
        {"name": "ExtraTrees balanced", "kind": "et", "n_estimators": 260, "max_depth": 16, "min_samples_leaf": 2, "class_weight": "balanced"},
        {"name": "ExtraTrees high weight", "kind": "et", "n_estimators": 300, "max_depth": 18, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 6}},
        {"name": "ExtraTrees shallow high weight", "kind": "et", "n_estimators": 260, "max_depth": 10, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 8}},
        {"name": "ExtraTrees medium recall low leaf", "kind": "et", "n_estimators": 280, "max_depth": 14, "min_samples_leaf": 1, "class_weight": {0: 1, 1: 7}},
        {"name": "ExtraTrees medium recall high leaf", "kind": "et", "n_estimators": 280, "max_depth": 14, "min_samples_leaf": 8, "class_weight": {0: 1, 1: 7}},
        {"name": "ExtraTrees deep recall low leaf", "kind": "et", "n_estimators": 300, "max_depth": 24, "min_samples_leaf": 1, "class_weight": {0: 1, 1: 7}},
    ]


def load_augmented_split():
    df = pd.read_parquet(DATA_FILE)
    df = engineer_features(df)
    df["fixture_id"] = df["fixture_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    base_features = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=base_features + [TARGET, "timestamp"])

    fixture_order = (
        df.groupby("fixture_id")["timestamp"]
        .min()
        .sort_values()
        .index
        .to_numpy()
    )
    n = len(fixture_order)
    n_train = max(1, int(n * 0.60))
    n_val = max(1, int(n * 0.20))
    train_ids = set(fixture_order[:n_train])
    val_ids = set(fixture_order[n_train : n_train + n_val])
    test_ids = set(fixture_order[n_train + n_val :])

    train = df[df["fixture_id"].isin(train_ids)].copy()
    val = df[df["fixture_id"].isin(val_ids)].copy()
    test = df[df["fixture_id"].isin(test_ids)].copy()

    print("Building Markov xT feature layer for augmented recall model...")
    train, edges = add_state_columns(train)
    val, _ = add_state_columns(val, edges)
    test, _ = add_state_columns(test, edges)
    transitions, _ = build_transition_model(train)
    state_risk, event_time_risk, event_time_move, global_rate = build_impact_tables(train)
    values = value_iteration(transitions, state_risk, global_rate)
    train = add_markov_scores(train, state_risk, event_time_risk, event_time_move, values, global_rate)
    val = add_markov_scores(val, state_risk, event_time_risk, event_time_move, values, global_rate)
    test = add_markov_scores(test, state_risk, event_time_risk, event_time_move, values, global_rate)

    features = base_features + MARKOV_FEATURES
    train = train.dropna(subset=features + [TARGET])
    val = val.dropna(subset=features + [TARGET])
    test = test.dropna(subset=features + [TARGET])
    return train, val, test, features


def main():
    train, val, test, features = load_augmented_split()
    X_train, y_train = train[features].values, train[TARGET].values
    X_val, y_val = val[features].values, val[TARGET].values
    X_test, y_test = test[features].values, test[TARGET].values
    precision_floor = max(0.30, float(y_val.mean()) + 0.06)

    print(f"Rows train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"Features: {len(features)} ({len(MARKOV_FEATURES)} Markov xT)")
    print(f"Vol rates train/val/test: {y_train.mean():.3f}/{y_val.mean():.3f}/{y_test.mean():.3f}")
    print(f"Precision floor on validation: {precision_floor:.3f}")

    results = []
    best = None
    no_improve = 0
    min_delta = 0.003

    for i, spec in enumerate(candidate_specs_augmented(), start=1):
        print(f"\n[{i}] {spec['name']} + Markov")
        model, scaler, val_prob, test_prob = fit_predict(spec, X_train, y_train, X_val, X_test)
        thresh_info = choose_threshold(y_val, val_prob, min_precision=precision_floor)
        val_metrics = metrics_for(spec["name"] + " + Markov", y_val, val_prob, thresh_info["threshold"])
        test_metrics = metrics_for(spec["name"] + " + Markov", y_test, test_prob, thresh_info["threshold"])
        row = {
            **test_metrics,
            "iteration": i,
            "kind": spec["kind"],
            "validation": val_metrics,
            "selection": thresh_info,
            "optimize_for": "recall_with_precision_floor",
            "precision_floor": round(precision_floor, 4),
        }
        results.append(row)
        print(
            f"val recall={val_metrics['recall']:.4f} precision={val_metrics['precision']:.4f} "
            f"f3={val_metrics['f3']:.4f} threshold={thresh_info['threshold']:.2f}"
        )
        print(
            f"test recall={test_metrics['recall']:.4f} precision={test_metrics['precision']:.4f} "
            f"f3={test_metrics['f3']:.4f} alert_rate={test_metrics['alert_rate']:.4f}"
        )

        val_score = (val_metrics["recall"], val_metrics["f3"], val_metrics["precision"])
        if best is None or val_score > best["val_score"]:
            improved = best is None or val_metrics["recall"] >= best["val_score"][0] + min_delta
            best = {
                "val_score": val_score,
                "row": row,
                "model": model,
                "scaler": scaler,
                "test_prob": test_prob,
                "threshold": thresh_info["threshold"],
            }
            no_improve = 0 if improved else no_improve + 1
        else:
            no_improve += 1

        if no_improve >= 10:
            print("Early stopping: 10 consecutive candidates without effective validation recall improvement.")
            break

    best_row = best["row"]
    threshold = float(best["threshold"])

    with open(MODEL_DIR / "markov_augmented_recall_best_model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": best["model"],
                "scaler": best["scaler"],
                "features": features,
                "threshold": threshold,
                "metrics": best_row,
            },
            f,
        )

    prob = best["test_prob"]
    extra_viz = MARKOV_FEATURES + [
        "event_group",
        "score_bucket",
        "time_bucket",
        "period_bucket",
        "momentum_bucket",
        "confidence_bucket",
        "markov_event_move",
        "markov_jump_level",
    ]
    out_cols = list(dict.fromkeys([c for c in VIZ_COLS + extra_viz if c in test.columns]))
    test_out = test[out_cols].copy()
    test_out["volatility_prob"] = prob
    test_out["predicted_volatile"] = (prob >= threshold).astype(int)
    test_out.to_parquet(OUT_DIR / "markov_augmented_test_predictions.parquet", index=False)

    out = {
        "models": results,
        "best_model": best_row["model"],
        "best_f1": best_row["f1"],
        "best_f3": best_row["f3"],
        "best_recall": best_row["recall"],
        "best_precision": best_row["precision"],
        "best_threshold": round(threshold, 4),
        "selection_metric": "validation recall, precision floor, then F3",
        "optimize_for": "coverage_recall_with_markov_xt_features",
        "precision_floor": round(precision_floor, 4),
        "dataset_rows": int(len(train) + len(val) + len(test)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_vol_rate": round(float(y_train.mean()), 4),
        "val_vol_rate": round(float(y_val.mean()), 4),
        "test_vol_rate": round(float(y_test.mean()), 4),
        "n_features": len(features),
        "markov_features": MARKOV_FEATURES,
        "features": features,
        "outputs": {
            "predictions": "outputs/markov_augmented_test_predictions.parquet",
            "model": "outputs/models/markov_augmented_recall_best_model.pkl",
        },
    }
    (OUT_DIR / "markov_augmented_recall_history.json").write_text(json.dumps(out, indent=2))

    print("\nBest Markov-augmented recall model:")
    print(json.dumps(best_row, indent=2))
    return out


if __name__ == "__main__":
    main()
