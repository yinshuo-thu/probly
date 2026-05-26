"""
Continuous pricing model for LSports -> Polymarket price-path research.

Outputs continuous forecasts:
- expected absolute mid-price move over the next 120 seconds
- expected signed 1-minute price change

The model keeps the recall classifier available, but adds a richer Markov v2
state layer and gradient-boosted continuous regressors for practical pricing
detail in the frontend.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from markov_xt_model import event_group, confidence_bucket, period_bucket, score_bucket, time_bucket
from optimize_recall_models import FEATURE_COLS, VIZ_COLS
from train_models import DATA_FILE, MODEL_DIR, OUT_DIR, TARGET, engineer_features


ABS_TARGET = "max_abs_move_120s"
SIGNED_TARGET = "price_change_1m"

MARKOV_V2_FEATURES = [
    "markov_v2_state_abs_move",
    "markov_v2_event_abs_move",
    "markov_v2_value_abs_move",
    "markov_v2_state_signed_move",
    "markov_v2_event_signed_move",
    "markov_v2_value_signed_move",
    "markov_v2_transition_weight",
    "markov_v2_path_abs_score",
]


def chronological_split(df: pd.DataFrame):
    fixture_order = (
        df.groupby("fixture_id")["timestamp"].min().sort_values().index.to_numpy()
    )
    n = len(fixture_order)
    n_train = max(1, int(n * 0.60))
    n_val = max(1, int(n * 0.20))
    train_ids = set(fixture_order[:n_train])
    val_ids = set(fixture_order[n_train : n_train + n_val])
    test_ids = set(fixture_order[n_train + n_val :])
    return (
        df[df["fixture_id"].isin(train_ids)].copy(),
        df[df["fixture_id"].isin(val_ids)].copy(),
        df[df["fixture_id"].isin(test_ids)].copy(),
    )


def quantile_edges(s: pd.Series, qs=(0.25, 0.5, 0.75), fallback=(0.1, 0.3, 0.6)):
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 100:
        return tuple(float(x) for x in fallback)
    edges = tuple(float(x) for x in np.quantile(vals, qs))
    if any(not math.isfinite(x) for x in edges) or len(set(edges)) < len(edges):
        return tuple(float(x) for x in fallback)
    return edges


def bucket_by_edges(v, edges, labels):
    if pd.isna(v):
        return labels[0]
    x = float(v)
    for edge, label in zip(edges, labels):
        if x <= edge:
            return label
    return labels[-1]


def add_markov_v2_state(df: pd.DataFrame, edges: dict | None = None):
    df = df.copy()
    if edges is None:
        edges = {
            "momentum": quantile_edges(df["event_density_60s"], (0.33, 0.66), (180, 380)),
            "pressure": quantile_edges(df.get("pressure_score", pd.Series(0, index=df.index)), (0.33, 0.66), (0.1, 0.4)),
            "realized_vol": quantile_edges(df.get("price_realized_vol", pd.Series(0, index=df.index)), (0.33, 0.66), (0.01, 0.04)),
        }
    df["event_group"] = df["incident_name"].map(event_group)
    df["score_bucket"] = df["score_diff"].map(score_bucket)
    df["time_bucket"] = df["minutes_remaining"].map(time_bucket)
    df["period_bucket"] = df["period_id"].map(period_bucket)
    df["confidence_bucket"] = df["confidence_grade"].map(confidence_bucket)
    df["momentum_bucket"] = [
        bucket_by_edges(x, edges["momentum"], ["low", "medium", "high"])
        for x in pd.to_numeric(df["event_density_60s"], errors="coerce")
    ]
    df["pressure_bucket"] = [
        bucket_by_edges(x, edges["pressure"], ["calm", "active", "surge"])
        for x in pd.to_numeric(df.get("pressure_score", 0), errors="coerce")
    ]
    df["realized_vol_bucket"] = [
        bucket_by_edges(x, edges["realized_vol"], ["quiet", "moving", "unstable"])
        for x in pd.to_numeric(df.get("price_realized_vol", 0), errors="coerce")
    ]
    price = pd.to_numeric(df["mid_price"], errors="coerce").fillna(0.5)
    df["price_zone_bucket"] = pd.cut(
        price,
        bins=[-0.01, 0.12, 0.30, 0.70, 0.88, 1.01],
        labels=["near_zero", "low", "middle", "high", "near_one"],
    ).astype(str)
    df["score_context_bucket"] = np.select(
        [
            pd.to_numeric(df["score_diff"], errors="coerce").fillna(0).abs() == 0,
            pd.to_numeric(df["score_diff"], errors="coerce").fillna(0).abs() == 1,
        ],
        ["draw", "one_goal"],
        default="multi_goal",
    )
    df["markov_v2_state"] = (
        df["score_bucket"].astype(str)
        + "|"
        + df["time_bucket"].astype(str)
        + "|"
        + df["period_bucket"].astype(str)
        + "|"
        + df["momentum_bucket"].astype(str)
        + "|"
        + df["pressure_bucket"].astype(str)
        + "|"
        + df["price_zone_bucket"].astype(str)
        + "|"
        + df["event_group"].astype(str)
        + "|"
        + df["confidence_bucket"].astype(str)
    )
    return df, edges


def weighted_mean(values: pd.Series, weights: pd.Series, default: float):
    v = pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.2).clip(0.05, 1.0).to_numpy(dtype=float)
    den = w.sum()
    if den <= 0:
        return float(default)
    return float(np.dot(v, w) / den)


def learn_tables(train: pd.DataFrame, target: str):
    global_mean = weighted_mean(train[target], train["confidence_grade"], float(train[target].mean()))
    state_table = {}
    event_table = {}
    for state, grp in train.groupby("markov_v2_state"):
        if len(grp) >= 15:
            state_table[str(state)] = weighted_mean(grp[target], grp["confidence_grade"], global_mean)
    for key, grp in train.groupby(["event_group", "time_bucket", "score_context_bucket"]):
        if len(grp) >= 25:
            event_table[tuple(map(str, key))] = weighted_mean(grp[target], grp["confidence_grade"], global_mean)
    return global_mean, state_table, event_table


def learn_transitions(train: pd.DataFrame):
    edge_weight = defaultdict(Counter)
    state_weight = Counter()
    for _, grp in train.sort_values(["fixture_id", "timestamp"]).groupby("fixture_id"):
        states = grp["markov_v2_state"].to_numpy()
        conf = pd.to_numeric(grp["confidence_grade"], errors="coerce").fillna(0.2).clip(0.05, 1.0).to_numpy()
        density = pd.to_numeric(grp.get("event_density_60s", 0), errors="coerce").fillna(0).to_numpy()
        if len(states) < 2:
            continue
        density_scale = 1.0 + np.clip(density / 600.0, 0, 1.5)
        for i in range(len(states) - 1):
            w = float(conf[i] * density_scale[i])
            s0, s1 = str(states[i]), str(states[i + 1])
            edge_weight[s0][s1] += w
            state_weight[s0] += w
    transitions = {}
    for s0, targets in edge_weight.items():
        total = float(sum(targets.values())) or 1.0
        transitions[s0] = [(s1, float(w) / total) for s1, w in targets.items()]
    return transitions, {s: float(w) for s, w in state_weight.items()}


def value_iteration(transitions, reward, global_reward, gamma=0.68, max_iter=70, tol=1e-6):
    states = set(reward) | set(transitions)
    for targets in transitions.values():
        states.update(s for s, _ in targets)
    values = {s: reward.get(s, global_reward) for s in states}
    for _ in range(max_iter):
        delta = 0.0
        new_values = {}
        for s in states:
            future = sum(p * values.get(s1, global_reward) for s1, p in transitions.get(s, []))
            nv = reward.get(s, global_reward) + gamma * future
            new_values[s] = nv
            delta = max(delta, abs(nv - values.get(s, 0.0)))
        values = new_values
        if delta < tol:
            break
    return values


def add_markov_v2_features(df, tables):
    df = df.copy()
    abs_global, abs_state, abs_event = tables["abs"]
    signed_global, signed_state, signed_event = tables["signed"]
    transitions_weight = tables["transition_weight"]
    abs_value = tables["abs_value"]
    signed_value = tables["signed_value"]

    event_keys = list(zip(df["event_group"], df["time_bucket"], df["score_context_bucket"]))
    df["markov_v2_state_abs_move"] = df["markov_v2_state"].map(abs_state).fillna(abs_global).astype(float)
    df["markov_v2_event_abs_move"] = [abs_event.get(tuple(map(str, k)), abs_global) for k in event_keys]
    df["markov_v2_value_abs_move"] = df["markov_v2_state"].map(abs_value).fillna(abs_global).astype(float)
    df["markov_v2_state_signed_move"] = df["markov_v2_state"].map(signed_state).fillna(signed_global).astype(float)
    df["markov_v2_event_signed_move"] = [signed_event.get(tuple(map(str, k)), signed_global) for k in event_keys]
    df["markov_v2_value_signed_move"] = df["markov_v2_state"].map(signed_value).fillna(signed_global).astype(float)
    df["markov_v2_transition_weight"] = df["markov_v2_state"].map(transitions_weight).fillna(0.0).astype(float)

    abs_norm = df["markov_v2_value_abs_move"].clip(0, 0.30) / 0.30
    event_norm = pd.Series(df["markov_v2_event_abs_move"], index=df.index).clip(0, 0.30) / 0.30
    conf = pd.to_numeric(df["confidence_grade"], errors="coerce").fillna(0.2).clip(0, 1)
    density = pd.to_numeric(df.get("event_density_60s", 0), errors="coerce").fillna(0).clip(0, 800) / 800
    df["markov_v2_path_abs_score"] = (0.45 * abs_norm + 0.30 * event_norm + 0.15 * density + 0.10 * conf).clip(0, 1)
    return df


def prepare_data():
    df = pd.read_parquet(DATA_FILE)
    df = engineer_features(df)
    df["fixture_id"] = df["fixture_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # FEATURE_COLS was originally built for the binary volatility classifier and
    # includes price_change_1m. That is the signed regression target here, so
    # keep it out of every continuous model to avoid look-ahead leakage.
    base_features = [c for c in FEATURE_COLS if c in df.columns and c != SIGNED_TARGET]
    needed = base_features + [ABS_TARGET, SIGNED_TARGET, "timestamp", "fixture_id", "incident_name", "mid_price"]
    df = df.dropna(subset=needed)
    df = df.sort_values(["fixture_id", "timestamp"]).reset_index(drop=True)
    train, val, test = chronological_split(df)

    train, edges = add_markov_v2_state(train)
    val, _ = add_markov_v2_state(val, edges)
    test, _ = add_markov_v2_state(test, edges)
    transitions, transition_weight = learn_transitions(train)
    abs_tables = learn_tables(train, ABS_TARGET)
    signed_tables = learn_tables(train, SIGNED_TARGET)
    abs_value = value_iteration(transitions, abs_tables[1], abs_tables[0], gamma=0.70)
    signed_value = value_iteration(transitions, signed_tables[1], signed_tables[0], gamma=0.55)
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
    features = base_features + MARKOV_V2_FEATURES
    return train, val, test, features, tables


def regression_metrics(y_true, y_pred, label):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    top_n = max(1, int(len(y_pred) * 0.10))
    top_idx = np.argsort(y_pred)[-top_n:]
    top_capture = float(y_true[top_idx].mean() / (y_true.mean() or 1.0))
    return {
        f"{label}_mae": round(float(mae), 5),
        f"{label}_rmse": round(float(rmse), 5),
        f"{label}_r2": round(float(r2_score(y_true, y_pred)), 4),
        f"{label}_spearman": round(float(spearman if pd.notna(spearman) else 0.0), 4),
        f"{label}_top_decile_lift": round(top_capture, 4),
    }


def fit_regressor(X_train, y_train, sample_weight, cfg):
    model = HistGradientBoostingRegressor(
        loss=cfg.get("loss", "squared_error"),
        learning_rate=cfg.get("learning_rate", 0.05),
        max_iter=cfg.get("max_iter", 220),
        max_leaf_nodes=cfg.get("max_leaf_nodes", 31),
        min_samples_leaf=cfg.get("min_samples_leaf", 35),
        l2_regularization=cfg.get("l2_regularization", 0.02),
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def choose_abs_model(train, val, features):
    configs = [
        {"name": "HGB abs squared conservative", "learning_rate": 0.045, "max_iter": 220, "max_leaf_nodes": 31, "min_samples_leaf": 40},
        {"name": "HGB abs squared deeper", "learning_rate": 0.035, "max_iter": 280, "max_leaf_nodes": 63, "min_samples_leaf": 35},
        {"name": "HGB abs absolute", "loss": "absolute_error", "learning_rate": 0.05, "max_iter": 180, "max_leaf_nodes": 31, "min_samples_leaf": 45},
        {"name": "HGB abs poisson-like", "loss": "poisson", "learning_rate": 0.04, "max_iter": 220, "max_leaf_nodes": 31, "min_samples_leaf": 35},
    ]
    X_train = train[features].to_numpy(dtype=float)
    X_val = val[features].to_numpy(dtype=float)
    y_train = train[ABS_TARGET].clip(lower=0).to_numpy(dtype=float)
    y_val = val[ABS_TARGET].clip(lower=0).to_numpy(dtype=float)
    sample_weight = pd.to_numeric(train["confidence_grade"], errors="coerce").fillna(0.2).clip(0.1, 1.0).to_numpy(dtype=float)
    best = None
    rows = []
    for cfg in configs:
        print(f"Training continuous abs model: {cfg['name']}")
        model = fit_regressor(X_train, y_train, sample_weight, cfg)
        pred = np.clip(model.predict(X_val), 0, 1)
        metrics = regression_metrics(y_val, pred, "abs_move")
        row = {"model": cfg["name"], **metrics}
        rows.append(row)
        # Prefer rank quality, then MAE; the UI uses this for finding large moves.
        score = (metrics["abs_move_spearman"], metrics["abs_move_top_decile_lift"], -metrics["abs_move_mae"])
        if best is None or score > best["score"]:
            best = {"score": score, "model": model, "row": row, "config": cfg}
    return best, rows


def fit_signed_model(train, val, test, features):
    X_train = train[features].to_numpy(dtype=float)
    y_train = train[SIGNED_TARGET].to_numpy(dtype=float)
    sample_weight = pd.to_numeric(train["confidence_grade"], errors="coerce").fillna(0.2).clip(0.1, 1.0).to_numpy(dtype=float)
    cfg = {"learning_rate": 0.035, "max_iter": 260, "max_leaf_nodes": 31, "min_samples_leaf": 45, "l2_regularization": 0.05}
    model = fit_regressor(X_train, y_train, sample_weight, cfg)
    val_pred = model.predict(val[features].to_numpy(dtype=float))
    test_pred = model.predict(test[features].to_numpy(dtype=float))
    return model, regression_metrics(val[SIGNED_TARGET], val_pred, "signed_move"), test_pred


def direction_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= 0.02
    if mask.sum() == 0:
        return {"direction_accuracy_2c": None, "direction_rows_2c": 0}
    return {
        "direction_accuracy_2c": round(float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()), 4),
        "direction_rows_2c": int(mask.sum()),
    }


def promote_to_main_predictions(continuous_pred: pd.DataFrame):
    main_path = OUT_DIR / "test_predictions.parquet"
    if not main_path.exists():
        return False
    main = pd.read_parquet(main_path)
    main["timestamp"] = pd.to_datetime(main["timestamp"], utc=True, errors="coerce")
    cont = continuous_pred.copy()
    cont["timestamp"] = pd.to_datetime(cont["timestamp"], utc=True, errors="coerce")
    main_sorted = main.sort_values(["fixture_id", "timestamp", "incident_name"]).reset_index()
    cont_sorted = cont.sort_values(["fixture_id", "timestamp", "incident_name"]).reset_index(drop=True)
    if len(main_sorted) != len(cont_sorted):
        return False
    key_cols = ["fixture_id", "timestamp", "incident_name"]
    if not main_sorted[key_cols].astype(str).equals(cont_sorted[key_cols].astype(str)):
        return False
    cols = [
        "predicted_abs_move_120s",
        "predicted_price_change_1m",
        "predicted_price_after_1m",
        "continuous_risk_score",
        "markov_v2_state_abs_move",
        "markov_v2_event_abs_move",
        "markov_v2_value_abs_move",
        "markov_v2_state_signed_move",
        "markov_v2_event_signed_move",
        "markov_v2_value_signed_move",
        "markov_v2_transition_weight",
        "markov_v2_path_abs_score",
        "markov_v2_state",
        "price_zone_bucket",
        "pressure_bucket",
        "realized_vol_bucket",
        "score_context_bucket",
    ]
    for col in cols:
        if col in cont_sorted.columns:
            main_sorted[col] = cont_sorted[col].to_numpy()
    restored = main_sorted.sort_values("index").drop(columns=["index"])
    restored.to_parquet(main_path, index=False)
    return True


def main():
    train, val, test, features, tables = prepare_data()
    print(f"Rows train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"Continuous features: {len(features)}")

    abs_best, abs_rows = choose_abs_model(train, val, features)
    signed_model, signed_val_metrics, signed_test_pred = fit_signed_model(train, val, test, features)

    abs_model = abs_best["model"]
    abs_test_pred = np.clip(abs_model.predict(test[features].to_numpy(dtype=float)), 0, 1)
    signed_test_pred = np.clip(signed_test_pred, -1, 1)
    abs_test_metrics = regression_metrics(test[ABS_TARGET], abs_test_pred, "abs_move")
    signed_test_metrics = regression_metrics(test[SIGNED_TARGET], signed_test_pred, "signed_move")
    dir_metrics = direction_metrics(test[SIGNED_TARGET], signed_test_pred)

    out_cols = list(dict.fromkeys([c for c in VIZ_COLS + [
        "event_group", "score_bucket", "time_bucket", "period_bucket", "momentum_bucket",
        "confidence_bucket", "pressure_bucket", "realized_vol_bucket", "price_zone_bucket",
        "score_context_bucket", "markov_v2_state", *MARKOV_V2_FEATURES,
    ] if c in test.columns]))
    pred = test[out_cols].copy()
    pred["predicted_abs_move_120s"] = abs_test_pred
    pred["predicted_price_change_1m"] = signed_test_pred
    pred["predicted_price_after_1m"] = (pd.to_numeric(pred["mid_price"], errors="coerce").fillna(0.5) + signed_test_pred).clip(0, 1)
    pred["continuous_risk_score"] = np.clip(abs_test_pred / 0.15, 0, 1)
    pred.to_parquet(OUT_DIR / "continuous_pricing_predictions.parquet", index=False)

    with open(MODEL_DIR / "continuous_pricing_models.pkl", "wb") as f:
        pickle.dump({"abs_model": abs_model, "signed_model": signed_model, "features": features}, f)

    promoted = promote_to_main_predictions(pred)
    metrics = {
        "model": "Markov v2 + HistGradientBoosting continuous pricing",
        "objective": "continuous_price_path_forecast",
        "rows": int(len(train) + len(val) + len(test)),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "features": features,
        "markov_v2_features": MARKOV_V2_FEATURES,
        "markov_v2_states": int(len(tables["transition_weight"])),
        "best_abs_validation_model": abs_best["row"],
        "abs_validation_candidates": abs_rows,
        "signed_validation": signed_val_metrics,
        "test": {**abs_test_metrics, **signed_test_metrics, **dir_metrics},
        "promoted_to_test_predictions": promoted,
        "outputs": {
            "predictions": "outputs/continuous_pricing_predictions.parquet",
            "metrics": "outputs/continuous_pricing_metrics.json",
            "model": "outputs/models/continuous_pricing_models.pkl",
        },
    }
    (OUT_DIR / "continuous_pricing_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
