"""
Recall-first model optimization for adverse-selection avoidance.

This script keeps the same chronological fixture split as train_models.py, but
optimizes for coverage of future high-volatility windows. Thresholds are chosen
on validation only. The objective is:

1. precision must be meaningfully above the validation base volatility rate
2. maximize recall
3. use F3 as a tie-breaker

Training stops after 10 consecutive candidate iterations fail to improve the
validation objective.
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_models import DATA_FILE, MODEL_DIR, OUT_DIR, TARGET, engineer_features


FEATURE_COLS = [
    "confidence_grade", "conf_trend", "xt_weight",
    "event_density_30s", "event_density_60s", "event_density_120s",
    "risk_event_count_30s", "risk_event_count_60s",
    "score_diff", "minutes_remaining", "period_id", "mid_price",
    "xt_x_conf", "is_high_impact", "is_timer_period",
    "risk_density_ratio", "price_extremeness",
    "total_goals", "is_leading", "is_drawing", "game_intensity",
    "event_density_300s", "high_impact_60s", "high_impact_300s",
    "xt_sum_60s", "xt_sum_300s",
    "price_velocity", "price_realized_vol", "price_change_1m",
    "pressure_score",
]

VIZ_COLS = [
    "fixture_id", "timestamp", "incident_name",
    "mid_price", "minutes_remaining", "minutes_elapsed",
    "score_diff", "total_goals", "period_id",
    "xt_weight", "confidence_grade", "conf_trend",
    "event_density_30s", "event_density_60s", "event_density_120s",
    "event_density_300s", "risk_event_count_30s", "risk_event_count_60s",
    "high_impact_60s", "high_impact_300s", "xt_sum_60s", "xt_sum_300s",
    "price_velocity", "price_realized_vol", "price_change_1m",
    "high_volatility", "max_abs_move_120s", "next_60s_high_impact",
]


def load_split():
    df = pd.read_parquet(DATA_FILE)
    df = engineer_features(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    features = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=features + [TARGET, "timestamp"])

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
    val_ids = set(fixture_order[n_train:n_train + n_val])
    test_ids = set(fixture_order[n_train + n_val:])

    train = df[df["fixture_id"].isin(train_ids)].copy()
    val = df[df["fixture_id"].isin(val_ids)].copy()
    test = df[df["fixture_id"].isin(test_ids)].copy()
    return train, val, test, features


def choose_threshold(y_true, y_prob, beta=3.0, min_precision=None):
    base = float(np.mean(y_true))
    if min_precision is None:
        min_precision = max(0.30, base + 0.06)

    best = None
    thresholds = np.r_[np.arange(0.02, 0.50, 0.01), np.arange(0.50, 0.91, 0.02)]
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f3 = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        alert_rate = float(pred.mean())
        ok = precision >= min_precision
        score = (1 if ok else 0, recall if ok else precision, f3, -alert_rate)
        row = {
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f3": float(f3),
            "alert_rate": alert_rate,
            "meets_precision_floor": bool(ok),
            "precision_floor": float(min_precision),
        }
        if best is None or score > best[0]:
            best = (score, row)
    return best[1]


def metrics_for(name, y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    return {
        "model": name,
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "f3": round(fbeta_score(y_true, pred, beta=3, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "alert_rate": round(float(pred.mean()), 4),
        "miss_rate": round(1 - recall_score(y_true, pred, zero_division=0), 4),
    }


def candidate_specs():
    specs = []
    try:
        import lightgbm  # noqa: F401
        specs.extend([
            {"name": "LightGBM recall leaves63", "kind": "lgbm", "n_estimators": 550, "num_leaves": 63, "learning_rate": 0.04, "pos_weight": 3.0},
            {"name": "LightGBM recall leaves127", "kind": "lgbm", "n_estimators": 650, "num_leaves": 127, "learning_rate": 0.035, "pos_weight": 4.0},
            {"name": "LightGBM conservative", "kind": "lgbm", "n_estimators": 450, "num_leaves": 31, "learning_rate": 0.05, "pos_weight": 3.5},
            {"name": "LightGBM high recall", "kind": "lgbm", "n_estimators": 800, "num_leaves": 95, "learning_rate": 0.025, "pos_weight": 5.0},
            {"name": "LightGBM very high recall", "kind": "lgbm", "n_estimators": 900, "num_leaves": 63, "learning_rate": 0.025, "pos_weight": 7.0},
            {"name": "LightGBM wide low lr", "kind": "lgbm", "n_estimators": 1000, "num_leaves": 127, "learning_rate": 0.02, "pos_weight": 6.0},
            {"name": "LightGBM compact recall", "kind": "lgbm", "n_estimators": 700, "num_leaves": 31, "learning_rate": 0.03, "pos_weight": 8.0},
        ])
    except ImportError:
        pass
    specs.extend([
        {"name": "RF recall shallow", "kind": "rf", "n_estimators": 260, "max_depth": 8, "min_samples_leaf": 3, "class_weight": {0: 1, 1: 3}},
        {"name": "RF recall medium", "kind": "rf", "n_estimators": 320, "max_depth": 11, "min_samples_leaf": 4, "class_weight": {0: 1, 1: 4}},
        {"name": "RF balanced deep", "kind": "rf", "n_estimators": 360, "max_depth": 14, "min_samples_leaf": 3, "class_weight": "balanced"},
        {"name": "ExtraTrees recall", "kind": "et", "n_estimators": 360, "max_depth": 14, "min_samples_leaf": 3, "class_weight": {0: 1, 1: 4}},
        {"name": "ExtraTrees balanced", "kind": "et", "n_estimators": 420, "max_depth": 16, "min_samples_leaf": 2, "class_weight": "balanced"},
        {"name": "RF high weight", "kind": "rf", "n_estimators": 420, "max_depth": 12, "min_samples_leaf": 3, "class_weight": {0: 1, 1: 6}},
        {"name": "RF shallow high weight", "kind": "rf", "n_estimators": 360, "max_depth": 7, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 8}},
        {"name": "RF deep high weight", "kind": "rf", "n_estimators": 420, "max_depth": 18, "min_samples_leaf": 4, "class_weight": {0: 1, 1: 5}},
        {"name": "ExtraTrees high weight", "kind": "et", "n_estimators": 460, "max_depth": 18, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 6}},
        {"name": "ExtraTrees shallow high weight", "kind": "et", "n_estimators": 420, "max_depth": 10, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 8}},
        {"name": "ExtraTrees wide", "kind": "et", "n_estimators": 620, "max_depth": 16, "min_samples_leaf": 4, "class_weight": {0: 1, 1: 5}},
        {"name": "ExtraTrees deep balanced", "kind": "et", "n_estimators": 520, "max_depth": 22, "min_samples_leaf": 3, "class_weight": "balanced"},
        {"name": "RF wide balanced", "kind": "rf", "n_estimators": 520, "max_depth": 16, "min_samples_leaf": 5, "class_weight": "balanced"},
        {"name": "RF tiny recall", "kind": "rf", "n_estimators": 220, "max_depth": 6, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 10}},
        {"name": "RF medium recall low leaf", "kind": "rf", "n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 7}},
        {"name": "RF medium recall high leaf", "kind": "rf", "n_estimators": 300, "max_depth": 10, "min_samples_leaf": 8, "class_weight": {0: 1, 1: 7}},
        {"name": "RF deep recall low leaf", "kind": "rf", "n_estimators": 320, "max_depth": 20, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 7}},
        {"name": "ExtraTrees tiny recall", "kind": "et", "n_estimators": 260, "max_depth": 8, "min_samples_leaf": 2, "class_weight": {0: 1, 1: 10}},
        {"name": "ExtraTrees medium recall low leaf", "kind": "et", "n_estimators": 360, "max_depth": 14, "min_samples_leaf": 1, "class_weight": {0: 1, 1: 7}},
        {"name": "ExtraTrees medium recall high leaf", "kind": "et", "n_estimators": 360, "max_depth": 14, "min_samples_leaf": 8, "class_weight": {0: 1, 1: 7}},
        {"name": "ExtraTrees deep recall low leaf", "kind": "et", "n_estimators": 420, "max_depth": 24, "min_samples_leaf": 1, "class_weight": {0: 1, 1: 7}},
        {"name": "ExtraTrees deep recall high leaf", "kind": "et", "n_estimators": 420, "max_depth": 24, "min_samples_leaf": 8, "class_weight": {0: 1, 1: 7}},
        {"name": "RF final balanced recall", "kind": "rf", "n_estimators": 420, "max_depth": 9, "min_samples_leaf": 6, "class_weight": {0: 1, 1: 9}},
    ])
    return specs


def fit_predict(spec, X_train, y_train, X_val, X_test):
    scaler = None
    kind = spec["kind"]
    if kind == "rf":
        model = RandomForestClassifier(
            n_estimators=spec["n_estimators"],
            max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"],
            class_weight=spec["class_weight"],
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
    elif kind == "et":
        model = ExtraTreesClassifier(
            n_estimators=spec["n_estimators"],
            max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"],
            class_weight=spec["class_weight"],
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
    elif kind == "lgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=spec["n_estimators"],
            num_leaves=spec["num_leaves"],
            learning_rate=spec["learning_rate"],
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=spec["pos_weight"],
            min_child_samples=30,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"Unknown candidate kind: {kind}")
    return model, scaler, val_prob, test_prob


def main():
    train, val, test, features = load_split()
    X_train, y_train = train[features].values, train[TARGET].values
    X_val, y_val = val[features].values, val[TARGET].values
    X_test, y_test = test[features].values, test[TARGET].values
    precision_floor = max(0.30, float(y_val.mean()) + 0.06)

    print(f"Rows train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"Vol rates train/val/test: {y_train.mean():.3f}/{y_val.mean():.3f}/{y_test.mean():.3f}")
    print(f"Precision floor on validation: {precision_floor:.3f}")

    results = []
    best = None
    no_improve = 0
    min_delta = 0.003

    for i, spec in enumerate(candidate_specs(), start=1):
        print(f"\n[{i}] {spec['name']}")
        model, scaler, val_prob, test_prob = fit_predict(spec, X_train, y_train, X_val, X_test)
        thresh_info = choose_threshold(y_val, val_prob, min_precision=precision_floor)
        val_metrics = metrics_for(spec["name"], y_val, val_prob, thresh_info["threshold"])
        test_metrics = metrics_for(spec["name"], y_test, test_prob, thresh_info["threshold"])
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
    best_name = best_row["model"]
    threshold = float(best["threshold"])

    with open(MODEL_DIR / "recall_best_model.pkl", "wb") as f:
        pickle.dump({
            "model": best["model"],
            "scaler": best["scaler"],
            "features": features,
            "threshold": threshold,
            "metrics": best_row,
        }, f)

    prob = best["test_prob"]
    test_out = test[[c for c in VIZ_COLS if c in test.columns]].copy()
    test_out["volatility_prob"] = prob
    test_out["predicted_volatile"] = (prob >= threshold).astype(int)
    test_out.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)

    out = {
        "models": results,
        "best_model": best_name,
        "best_f1": best_row["f1"],
        "best_f3": best_row["f3"],
        "best_recall": best_row["recall"],
        "best_precision": best_row["precision"],
        "best_threshold": round(threshold, 4),
        "selection_metric": "validation recall, precision floor, then F3",
        "optimize_for": "coverage_recall",
        "precision_floor": round(precision_floor, 4),
        "dataset_rows": int(len(train) + len(val) + len(test)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_vol_rate": round(float(y_train.mean()), 4),
        "val_vol_rate": round(float(y_val.mean()), 4),
        "test_vol_rate": round(float(y_test.mean()), 4),
        "n_features": len(features),
        "features": features,
    }
    with open(OUT_DIR / "metrics_history.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(OUT_DIR / "recall_optimization_history.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nBest recall-first model:")
    print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
