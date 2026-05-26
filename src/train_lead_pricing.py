"""
Lead-time pricing model for LSports -> Polymarket price discovery.

This script is intentionally stricter than the same-window continuous model:
features at timestamp t predict Polymarket movement that starts after a lead
gap, e.g. the maximum absolute move from t+30s to t+150s. That gives a direct
test of whether the sports-state signal has usable advance pricing power.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from optimize_recall_models import FEATURE_COLS
from train_continuous_pricing import (
    MARKOV_V2_FEATURES,
    add_markov_v2_features,
    add_markov_v2_state,
    chronological_split,
    learn_tables,
    learn_transitions,
    value_iteration,
)
from train_models import DATA_FILE, MODEL_DIR, OUT_DIR, engineer_features


LEAD_SECONDS = 30
HORIZON_SECONDS = 120
ABS_TARGET = f"lead{LEAD_SECONDS}_max_abs_move_{HORIZON_SECONDS}s"
SIGNED_TARGET = f"lead{LEAD_SECONDS}_signed_move_{HORIZON_SECONDS}s"
HIGH_TARGET = f"lead{LEAD_SECONDS}_high_volatility"


def add_lead_targets(df: pd.DataFrame, lead_s: int = LEAD_SECONDS, horizon_s: int = HORIZON_SECONDS) -> pd.DataFrame:
    out = df.copy()
    out[ABS_TARGET] = np.nan
    out[SIGNED_TARGET] = np.nan
    out[HIGH_TARGET] = 0
    lead_ns = np.timedelta64(int(lead_s), "s")
    end_ns = np.timedelta64(int(lead_s + horizon_s), "s")

    for _, grp in out.sort_values(["fixture_id", "timestamp"]).groupby("fixture_id"):
        idx = grp.index.to_numpy()
        ts = grp["timestamp"].to_numpy(dtype="datetime64[ns]")
        price = pd.to_numeric(grp["mid_price"], errors="coerce").to_numpy(dtype=float)
        n = len(grp)
        if n < 3:
            continue
        abs_move = np.zeros(n, dtype=float)
        signed = np.zeros(n, dtype=float)
        for i, t in enumerate(ts):
            lo = np.searchsorted(ts, t + lead_ns, side="left")
            hi = np.searchsorted(ts, t + end_ns, side="right")
            if lo >= hi or lo >= n or not math.isfinite(price[i]):
                continue
            window = price[lo:hi]
            window = window[np.isfinite(window)]
            if len(window) == 0:
                continue
            abs_move[i] = float(np.max(np.abs(window - price[i])))
            signed[i] = float(window[-1] - price[i])
        out.loc[idx, ABS_TARGET] = abs_move
        out.loc[idx, SIGNED_TARGET] = signed
        out.loc[idx, HIGH_TARGET] = (abs_move > 0.03).astype(int)
    return out.dropna(subset=[ABS_TARGET, SIGNED_TARGET])


def weighted_mean(values: pd.Series, weights: pd.Series, default: float) -> float:
    v = pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.2).clip(0.05, 1.0).to_numpy(dtype=float)
    den = w.sum()
    return float(default if den <= 0 else np.dot(v, w) / den)


def prepare_data():
    df = pd.read_parquet(DATA_FILE)
    df = engineer_features(df)
    df["fixture_id"] = df["fixture_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    base_features = [c for c in FEATURE_COLS if c in df.columns]
    needed = base_features + ["timestamp", "fixture_id", "incident_name", "mid_price"]
    df = df.dropna(subset=needed).sort_values(["fixture_id", "timestamp"]).reset_index(drop=True)
    df = add_lead_targets(df)
    train, val, test = chronological_split(df)

    train, edges = add_markov_v2_state(train)
    val, _ = add_markov_v2_state(val, edges)
    test, _ = add_markov_v2_state(test, edges)

    transitions, transition_weight = learn_transitions(train)
    abs_tables = learn_tables(train, ABS_TARGET)
    signed_tables = learn_tables(train, SIGNED_TARGET)
    abs_value = value_iteration(transitions, abs_tables[1], abs_tables[0], gamma=0.68)
    signed_value = value_iteration(transitions, signed_tables[1], signed_tables[0], gamma=0.52)
    tables = {
        "abs": abs_tables,
        "signed": signed_tables,
        "transition_weight": transition_weight,
        "abs_value": abs_value,
        "signed_value": signed_value,
        "edges": edges,
    }
    train = add_markov_v2_features(train, tables)
    val = add_markov_v2_features(val, tables)
    test = add_markov_v2_features(test, tables)
    features = list(dict.fromkeys(base_features + MARKOV_V2_FEATURES))
    return train, val, test, features, tables


def fit_model(train, y_col, features, loss="squared_error"):
    model = HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=0.035,
        max_iter=320,
        max_leaf_nodes=47,
        min_samples_leaf=45,
        l2_regularization=0.03,
        random_state=43,
    )
    X = train[features].to_numpy(dtype=float)
    y = train[y_col].to_numpy(dtype=float)
    w = pd.to_numeric(train["confidence_grade"], errors="coerce").fillna(0.2).clip(0.1, 1.0).to_numpy(dtype=float)
    model.fit(X, y, sample_weight=w)
    return model


def fit_classifier(train, features):
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=260,
        max_leaf_nodes=47,
        min_samples_leaf=45,
        l2_regularization=0.04,
        random_state=44,
    )
    X = train[features].to_numpy(dtype=float)
    y = train[HIGH_TARGET].to_numpy(dtype=int)
    w = pd.to_numeric(train["confidence_grade"], errors="coerce").fillna(0.2).clip(0.1, 1.0).to_numpy(dtype=float)
    model.fit(X, y, sample_weight=w)
    return model


def regression_metrics(y_true, y_pred, label):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    top_n = max(1, int(len(y_pred) * 0.10))
    top_idx = np.argsort(y_pred)[-top_n:]
    mean = float(y_true.mean()) or 1.0
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        f"{label}_mae": round(float(mean_absolute_error(y_true, y_pred)), 5),
        f"{label}_rmse": round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 5),
        f"{label}_r2": round(float(r2_score(y_true, y_pred)), 4),
        f"{label}_spearman": round(float(spearman if pd.notna(spearman) else 0.0), 4),
        f"{label}_top_decile_lift": round(float(y_true[top_idx].mean() / mean), 4),
    }


def select_threshold(y_true, pred):
    thresholds = np.quantile(pred, np.linspace(0.55, 0.96, 42))
    best = None
    y = np.asarray(y_true, dtype=int)
    for th in sorted(set(float(x) for x in thresholds)):
        fired = pred >= th
        tp = int((fired & (y == 1)).sum())
        fp = int((fired & (y == 0)).sum())
        fn = int(((~fired) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f3 = (10 * precision * recall) / max(9 * precision + recall, 1e-12)
        score = (recall >= 0.70, f3, recall, precision)
        row = {"threshold": th, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f3": f3}
        if best is None or score > best["score"]:
            best = {"score": score, "row": row}
    return best["row"]


def select_precision_floor(y_true, pred, floor=0.60):
    thresholds = np.quantile(pred, np.linspace(0.50, 0.985, 80))
    best = None
    y = np.asarray(y_true, dtype=int)
    for th in sorted(set(float(x) for x in thresholds)):
        fired = pred >= th
        tp = int((fired & (y == 1)).sum())
        fp = int((fired & (y == 0)).sum())
        fn = int(((~fired) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        row = {"threshold": th, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}
        metric = (precision >= floor, recall, precision)
        if best is None or metric > best["metric"]:
            best = {"metric": metric, "row": row}
    return best["row"]


def choose_intercept_score(y_true, abs_pred, cls_prob):
    best = None
    abs_scaled = abs_pred / max(np.quantile(abs_pred, 0.90), 1e-6)
    abs_scaled = np.clip(abs_scaled, 0, 2)
    for w in np.linspace(0.0, 1.0, 11):
        score = w * cls_prob + (1 - w) * abs_scaled
        row = select_threshold(y_true, score)
        row["blend_classifier_weight"] = round(float(w), 2)
        metric = (row["recall"] >= 0.72, row["f3"], row["recall"], row["precision"])
        if best is None or metric > best["metric"]:
            best = {"metric": metric, "row": row, "score": score}
    return best


def intercept_metrics(y_true, pred, threshold):
    y = np.asarray(y_true, dtype=int)
    fired = np.asarray(pred) >= threshold
    tp = int((fired & (y == 1)).sum())
    fp = int((fired & (y == 0)).sum())
    fn = int(((~fired) & (y == 1)).sum())
    return {
        "lead_seconds": LEAD_SECONDS,
        "horizon_seconds": HORIZON_SECONDS,
        "total_future_high_volatility_intervals": int(y.sum()),
        "predicted_intercept_intervals": int(fired.sum()),
        "accurate_intercepts": tp,
        "missed_intervals": fn,
        "false_alarm_intervals": fp,
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "threshold": round(float(threshold), 6),
    }


def promote_to_main_predictions(pred: pd.DataFrame) -> bool:
    path = OUT_DIR / "test_predictions.parquet"
    if not path.exists():
        return False
    main = pd.read_parquet(path)
    main["timestamp"] = pd.to_datetime(main["timestamp"], utc=True, errors="coerce")
    pred = pred.copy()
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
    main_sorted = main.sort_values(["fixture_id", "timestamp", "incident_name"]).reset_index()
    pred_sorted = pred.sort_values(["fixture_id", "timestamp", "incident_name"]).reset_index(drop=True)
    if len(main_sorted) != len(pred_sorted):
        return False
    keys = ["fixture_id", "timestamp", "incident_name"]
    if not main_sorted[keys].astype(str).equals(pred_sorted[keys].astype(str)):
        return False
    cols = [
        ABS_TARGET,
        SIGNED_TARGET,
        HIGH_TARGET,
        "lead30_pred_abs_move_120s",
        "lead30_pred_signed_move_120s",
        "lead30_pred_price_after_150s",
        "lead30_intercept_score",
    ]
    for col in cols:
        main_sorted[col] = pred_sorted[col].to_numpy()
    main_sorted.sort_values("index").drop(columns=["index"]).to_parquet(path, index=False)
    return True


def main():
    train, val, test, features, tables = prepare_data()
    print(f"Rows train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"Lead pricing features: {len(features)}")

    abs_model = fit_model(train, ABS_TARGET, features, loss="poisson")
    signed_model = fit_model(train, SIGNED_TARGET, features, loss="squared_error")
    cls_model = fit_classifier(train, features)
    X_val = val[features].to_numpy(dtype=float)
    X_test = test[features].to_numpy(dtype=float)
    val_abs_pred = np.clip(abs_model.predict(X_val), 0, 1)
    test_abs_pred = np.clip(abs_model.predict(X_test), 0, 1)
    val_signed_pred = np.clip(signed_model.predict(X_val), -1, 1)
    test_signed_pred = np.clip(signed_model.predict(X_test), -1, 1)
    val_cls_prob = cls_model.predict_proba(X_val)[:, 1]
    test_cls_prob = cls_model.predict_proba(X_test)[:, 1]
    intercept = choose_intercept_score(val[HIGH_TARGET].to_numpy(dtype=int), val_abs_pred, val_cls_prob)
    th = intercept["row"]["threshold"]
    cls_w = intercept["row"]["blend_classifier_weight"]
    val_abs_scale = max(np.quantile(val_abs_pred, 0.90), 1e-6)
    val_score = cls_w * val_cls_prob + (1 - cls_w) * np.clip(val_abs_pred / val_abs_scale, 0, 2)
    test_score = cls_w * test_cls_prob + (1 - cls_w) * np.clip(test_abs_pred / val_abs_scale, 0, 2)
    high_conf = select_precision_floor(val[HIGH_TARGET].to_numpy(dtype=int), val_score, floor=0.60)
    high_conf["blend_classifier_weight"] = cls_w

    out = test.copy()
    out["lead30_pred_abs_move_120s"] = test_abs_pred
    out["lead30_pred_signed_move_120s"] = test_signed_pred
    out["lead30_pred_price_after_150s"] = (pd.to_numeric(out["mid_price"], errors="coerce").fillna(0.5) + test_signed_pred).clip(0, 1)
    out["lead30_intercept_score"] = test_score

    keep = [c for c in [
        "fixture_id", "timestamp", "incident_name", "mid_price", "volatility_prob",
        "high_volatility", "max_abs_move_120s", ABS_TARGET, SIGNED_TARGET, HIGH_TARGET,
        "lead30_pred_abs_move_120s", "lead30_pred_signed_move_120s",
        "lead30_pred_price_after_150s", "lead30_intercept_score",
    ] if c in out.columns]
    out[keep].to_parquet(OUT_DIR / "lead_pricing_predictions.parquet", index=False)
    promoted = promote_to_main_predictions(out)

    metrics = {
        "model": "Markov v2 lead-time pricing",
        "objective": "predict Polymarket movement beginning after a 30s lead gap",
        "lead_seconds": LEAD_SECONDS,
        "horizon_seconds": HORIZON_SECONDS,
        "rows": int(len(train) + len(val) + len(test)),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "features": features,
        "markov_v2_states": int(len(tables["transition_weight"])),
        "validation": {
            **regression_metrics(val[ABS_TARGET], val_abs_pred, "lead_abs_move"),
            **regression_metrics(val[SIGNED_TARGET], val_signed_pred, "lead_signed_move"),
            "threshold_selection": intercept["row"],
            "high_confidence_threshold_selection": high_conf,
        },
        "test": {
            **regression_metrics(test[ABS_TARGET], test_abs_pred, "lead_abs_move"),
            **regression_metrics(test[SIGNED_TARGET], test_signed_pred, "lead_signed_move"),
            **intercept_metrics(test[HIGH_TARGET].to_numpy(dtype=int), test_score, th),
            "high_confidence": intercept_metrics(test[HIGH_TARGET].to_numpy(dtype=int), test_score, high_conf["threshold"]),
        },
        "promoted_to_test_predictions": promoted,
        "outputs": {
            "predictions": "outputs/lead_pricing_predictions.parquet",
            "metrics": "outputs/lead_pricing_metrics.json",
            "model": "outputs/models/lead_pricing_models.pkl",
        },
    }
    (OUT_DIR / "lead_pricing_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(MODEL_DIR / "lead_pricing_models.pkl", "wb") as f:
        pickle.dump({"abs_model": abs_model, "signed_model": signed_model, "classifier": cls_model, "features": features, "threshold": th, "classifier_weight": cls_w}, f)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
