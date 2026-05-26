"""
Markov xT price-path model for LSports -> Polymarket volatility discovery.

This is a deliberately interpretable companion to the recall-first classifier:
it learns confidence-weighted state transitions, state/event price impact, value
iteration scores, event-time heatmaps, and top event paths.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from project_paths import OUTPUTS_DIR, PROJECT_ROOT

BASE = PROJECT_ROOT
OUT = OUTPUTS_DIR
DATA_FILE = OUT / "real_dataset_v4.parquet"
TARGET = "high_volatility"

HIGH_IMPACT_EVENTS = {
    "Score",
    "PlayerGoals",
    "Player Goals Inside the Penalty Area (Raw)",
    "Player Left Foot Goal",
    "Player Right Foot Goals",
    "Player Header Goals",
    "Header Goals (Raw)",
    "Right Foot Goals",
    "Left foot Goals",
    "VAR",
    "Penalties",
    "MissedPenalty",
    "Penalties Saved (Raw)",
    "RedCard",
    "Player Red Card",
    "ShotsOnTarget",
    "PlayerShotsOnTarget",
    "Player Header Shots On Target",
    "Player Shots On Target From Outside Penalty Area",
    "DangerousFreeKicks",
}

EVENT_GROUPS = [
    ("goal", ["Score", "Goal"]),
    ("var_penalty", ["VAR", "Penalt", "Penalty"]),
    ("red_card", ["RedCard", "Red Card"]),
    ("shot_on_target", ["ShotsOnTarget", "Shots On Target", "ShotOnTarget"]),
    ("shot", ["Shot", "Total Shots", "BlockedShots"]),
    ("danger", ["Dangerous", "Attack"]),
    ("set_piece", ["Corner", "FreeKick", "Free Kick"]),
    ("discipline", ["YellowCard", "Yellow Card", "Foul"]),
    ("substitution", ["Substitution"]),
    ("score_state", ["FixtureStatus", "Period", "Timer"]),
]

TIME_LABELS = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"]
CONF_LABELS = ["low", "mid", "high", "confirmed"]
MOMENTUM_LABELS = ["low", "medium", "high"]


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def event_group(name: str) -> str:
    text = str(name or "")
    for group, needles in EVENT_GROUPS:
        if any(n in text for n in needles):
            return group
    return "other"


def score_bucket(v: float) -> str:
    if pd.isna(v):
        return "0"
    return str(int(np.clip(round(float(v)), -4, 4)))


def time_bucket(v: float) -> str:
    if pd.isna(v):
        return "90+"
    x = float(v)
    if x < 0:
        return "0-15"
    for lo, hi, label in [
        (0, 15, "0-15"),
        (15, 30, "15-30"),
        (30, 45, "30-45"),
        (45, 60, "45-60"),
        (60, 75, "60-75"),
        (75, 90, "75-90"),
    ]:
        if lo <= x < hi:
            return label
    return "90+"


def confidence_bucket(v: float) -> str:
    if pd.isna(v):
        return "low"
    x = float(v)
    if x >= 0.995:
        return "confirmed"
    if x >= 0.8:
        return "high"
    if x >= 0.5:
        return "mid"
    return "low"


def period_bucket(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    try:
        x = int(v)
    except Exception:
        return "unknown"
    if x in {1, 10}:
        return "1H"
    if x in {2, 20}:
        return "2H"
    if x in {3, 30}:
        return "ET1"
    if x in {4, 40}:
        return "ET2"
    if x in {5, 50}:
        return "PEN"
    return str(x)


def add_state_columns(df: pd.DataFrame, momentum_edges: tuple[float, float] | None = None) -> tuple[pd.DataFrame, tuple[float, float]]:
    df = df.copy()
    density = pd.to_numeric(df.get("event_density_60s", 0), errors="coerce").fillna(0)
    if momentum_edges is None:
        lo, hi = np.quantile(density, [0.33, 0.66])
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            lo, hi = 120.0, 360.0
        momentum_edges = (float(lo), float(hi))

    lo, hi = momentum_edges
    df["event_group"] = df["incident_name"].map(event_group)
    df["score_bucket"] = df["score_diff"].map(score_bucket)
    df["time_bucket"] = df["minutes_remaining"].map(time_bucket)
    df["period_bucket"] = df["period_id"].map(period_bucket)
    df["confidence_bucket"] = df["confidence_grade"].map(confidence_bucket)
    df["momentum_bucket"] = np.select(
        [density <= lo, density <= hi],
        ["low", "medium"],
        default="high",
    )
    df["markov_state"] = (
        df["score_bucket"].astype(str)
        + "|"
        + df["time_bucket"].astype(str)
        + "|"
        + df["period_bucket"].astype(str)
        + "|"
        + df["momentum_bucket"].astype(str)
        + "|"
        + df["event_group"].astype(str)
        + "|"
        + df["confidence_bucket"].astype(str)
    )
    return df, momentum_edges


def chronological_split(df: pd.DataFrame) -> SplitData:
    order = df.groupby("fixture_id")["timestamp"].min().sort_values().index.to_numpy()
    n = len(order)
    n_train = max(1, int(n * 0.60))
    n_val = max(1, int(n * 0.20))
    train_ids = set(order[:n_train])
    val_ids = set(order[n_train : n_train + n_val])
    test_ids = set(order[n_train + n_val :])
    return SplitData(
        train=df[df["fixture_id"].isin(train_ids)].copy(),
        val=df[df["fixture_id"].isin(val_ids)].copy(),
        test=df[df["fixture_id"].isin(test_ids)].copy(),
    )


def weighted_rate(values: pd.Series, weights: pd.Series, default: float) -> float:
    v = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0.05).to_numpy(dtype=float)
    den = w.sum()
    if den <= 0:
        return default
    return float(np.dot(v, w) / den)


def build_transition_model(train: pd.DataFrame) -> tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
    edge_weight: dict[str, Counter] = defaultdict(Counter)
    state_weight: Counter = Counter()

    for _, grp in train.sort_values(["fixture_id", "timestamp"]).groupby("fixture_id"):
        states = grp["markov_state"].to_numpy()
        conf = pd.to_numeric(grp["confidence_grade"], errors="coerce").fillna(0.2).clip(0.05, 1.0).to_numpy()
        if len(states) < 2:
            continue
        for i in range(len(states) - 1):
            s0 = str(states[i])
            s1 = str(states[i + 1])
            w = float(conf[i])
            edge_weight[s0][s1] += w
            state_weight[s0] += w

    transitions: dict[str, list[tuple[str, float]]] = {}
    for s0, targets in edge_weight.items():
        total = float(sum(targets.values())) or 1.0
        transitions[s0] = [(s1, float(w) / total) for s1, w in targets.items()]

    priors = {s: float(w) for s, w in state_weight.items()}
    return transitions, priors


def build_impact_tables(train: pd.DataFrame) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[tuple[str, str], float], float]:
    global_rate = weighted_rate(train[TARGET], train["confidence_grade"], default=float(train[TARGET].mean()))

    state_risk: dict[str, float] = {}
    for state, grp in train.groupby("markov_state"):
        if len(grp) >= 20:
            state_risk[str(state)] = weighted_rate(grp[TARGET], grp["confidence_grade"], global_rate)

    event_time_risk: dict[tuple[str, str], float] = {}
    for key, grp in train.groupby(["event_group", "time_bucket"]):
        if len(grp) >= 30:
            event_time_risk[(str(key[0]), str(key[1]))] = weighted_rate(grp[TARGET], grp["confidence_grade"], global_rate)

    event_time_move: dict[tuple[str, str], float] = {}
    for key, grp in train.groupby(["event_group", "time_bucket"]):
        if len(grp) >= 30:
            event_time_move[(str(key[0]), str(key[1]))] = weighted_rate(
                grp["max_abs_move_120s"], grp["confidence_grade"], float(train["max_abs_move_120s"].mean())
            )

    return state_risk, event_time_risk, event_time_move, global_rate


def value_iteration(
    transitions: dict[str, list[tuple[str, float]]],
    state_risk: dict[str, float],
    global_rate: float,
    gamma: float = 0.72,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> dict[str, float]:
    states = set(state_risk) | set(transitions)
    for targets in transitions.values():
        states.update(s for s, _ in targets)
    values = {s: state_risk.get(s, global_rate) for s in states}
    for _ in range(max_iter):
        delta = 0.0
        new_values = {}
        for s in states:
            future = 0.0
            for s1, p in transitions.get(s, []):
                future += p * values.get(s1, global_rate)
            nv = state_risk.get(s, global_rate) + gamma * future
            new_values[s] = nv
            delta = max(delta, abs(nv - values.get(s, 0.0)))
        values = new_values
        if delta < tol:
            break
    return values


def normalize_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    lo, hi = s.quantile([0.02, 0.98])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or lo >= hi:
        lo, hi = float(s.min()), float(s.max())
    if lo >= hi:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def add_markov_scores(
    df: pd.DataFrame,
    state_risk: dict[str, float],
    event_time_risk: dict[tuple[str, str], float],
    event_time_move: dict[tuple[str, str], float],
    state_value: dict[str, float],
    global_rate: float,
) -> pd.DataFrame:
    df = df.copy()
    default_move = float(pd.to_numeric(df["max_abs_move_120s"], errors="coerce").fillna(0).mean())
    df["markov_state_risk"] = df["markov_state"].map(state_risk).fillna(global_rate).astype(float)
    df["markov_event_risk"] = [
        event_time_risk.get((str(e), str(t)), global_rate) for e, t in zip(df["event_group"], df["time_bucket"])
    ]
    df["markov_event_move"] = [
        event_time_move.get((str(e), str(t)), default_move)
        for e, t in zip(df["event_group"], df["time_bucket"])
    ]
    df["markov_state_value_raw"] = df["markov_state"].map(state_value).fillna(global_rate).astype(float)
    value_norm = normalize_score(df["markov_state_value_raw"])
    move_norm = normalize_score(df["markov_event_move"])

    conf = pd.to_numeric(df["confidence_grade"], errors="coerce").fillna(0.2).clip(0, 1)
    xt_conf = normalize_score(pd.to_numeric(df.get("xt_x_conf", conf), errors="coerce").fillna(0))
    pressure = normalize_score(pd.to_numeric(df.get("pressure_score", 0), errors="coerce").fillna(0))
    recency = normalize_score(90 - pd.to_numeric(df["minutes_remaining"], errors="coerce").fillna(90).clip(0, 90))

    df["markov_path_score"] = (
        0.38 * value_norm
        + 0.22 * df["markov_state_risk"].clip(0, 1)
        + 0.18 * pd.Series(df["markov_event_risk"], index=df.index).clip(0, 1)
        + 0.12 * move_norm
        + 0.05 * xt_conf
        + 0.03 * pressure
        + 0.02 * recency
    )
    df["markov_path_confidence"] = (df["markov_path_score"] * (0.35 + 0.65 * conf)).clip(0, 1)
    df["markov_jump_level"] = pd.cut(
        df["markov_event_move"],
        bins=[-0.001, 0.02, 0.05, 0.15, 0.30, 1.0],
        labels=["L1", "L2", "L3", "L4", "L5"],
    ).astype(str)
    return df


def select_threshold(y_true: np.ndarray, score: np.ndarray, precision_floor: float = 0.30) -> tuple[float, dict]:
    best = None
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (score >= t).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        f3 = ((1 + 9) * precision * recall / (9 * precision + recall)) if (precision + recall) else 0.0
        alert_rate = float(pred.mean())
        ok = precision >= precision_floor
        sort_key = (1 if ok else 0, recall if ok else precision, f3, f1, -alert_rate)
        if best is None or sort_key > best[0]:
            best = (sort_key, float(t), precision, recall, f1, f3, alert_rate)
    _, t, precision, recall, f1, f3, alert_rate = best
    return t, {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f3": float(f3),
        "alert_rate": float(alert_rate),
    }


def evaluate(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    out = {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "actual_high_volatility_intervals": int(y_true.sum()),
        "predicted_intervals": int(pred.sum()),
        "missed_intervals": fn,
        "false_alarm_intervals": fp,
    }
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_true, score)), 4)
    except ValueError:
        out["roc_auc"] = None
    return out


def export_heatmap(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    rows = []
    for key, grp in train.groupby(["event_group", "time_bucket"]):
        if len(grp) < 30:
            continue
        rows.append(
            {
                "event_group": str(key[0]),
                "time_bucket": str(key[1]),
                "train_rows": int(len(grp)),
                "avg_move": round(float(grp["max_abs_move_120s"].mean()), 4),
                "vol_rate": round(float(grp[TARGET].mean()), 4),
                "avg_confidence": round(float(grp["confidence_grade"].mean()), 4),
            }
        )
    present = {(r["event_group"], r["time_bucket"]): r for r in rows}
    for key, grp in test.groupby(["event_group", "time_bucket"]):
        item = present.get((str(key[0]), str(key[1])))
        if item is not None:
            item["test_rows"] = int(len(grp))
            item["test_vol_rate"] = round(float(grp[TARGET].mean()), 4)
    return sorted(rows, key=lambda x: (x["event_group"], TIME_LABELS.index(x["time_bucket"]) if x["time_bucket"] in TIME_LABELS else 99))


def export_top_paths(df: pd.DataFrame, top_n: int = 30) -> list[dict]:
    path_stats: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0, 0, 0.0, 0.0])
    event_col = "event_group"
    for fid, grp in df.sort_values(["fixture_id", "timestamp"]).groupby("fixture_id"):
        events = grp[event_col].to_numpy()
        confs = pd.to_numeric(grp["confidence_grade"], errors="coerce").fillna(0.2).clip(0.05, 1.0).to_numpy()
        moves = pd.to_numeric(grp["max_abs_move_120s"], errors="coerce").fillna(0).to_numpy()
        labels = grp[TARGET].to_numpy()
        scores = pd.to_numeric(grp["markov_path_score"], errors="coerce").fillna(0).to_numpy()
        if len(events) < 3:
            continue
        for i in range(2, len(events)):
            path = tuple(events[i - 2 : i + 1])
            if len(set(path)) == 1 and path[0] in {"score_state", "other"}:
                continue
            stat = path_stats[path]
            stat[0] += 1
            stat[1] += int(labels[i])
            stat[2] += float(moves[i])
            stat[3] += float(np.prod(confs[i - 2 : i + 1]) * scores[i] * max(moves[i], 0.01))
    out = []
    for path, (n, hit, move_sum, conf_sum) in path_stats.items():
        if n < 20:
            continue
        out.append(
            {
                "path": " -> ".join(path),
                "n": int(n),
                "hit_rate": round(float(hit / n), 4),
                "avg_move": round(float(move_sum / n), 4),
                "path_confidence": round(float(conf_sum / n), 5),
            }
        )
    return sorted(out, key=lambda x: (x["path_confidence"], x["hit_rate"], x["avg_move"]), reverse=True)[:top_n]


def main() -> dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}")

    print("Loading aligned dataset...")
    df = pd.read_parquet(DATA_FILE)
    df["fixture_id"] = df["fixture_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", TARGET, "mid_price", "minutes_remaining", "confidence_grade"])
    df = df.sort_values(["fixture_id", "timestamp"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows across {df['fixture_id'].nunique()} fixtures.")
    split0 = chronological_split(df)
    print("Discretizing Markov states...")
    train, edges = add_state_columns(split0.train)
    val, _ = add_state_columns(split0.val, edges)
    test, _ = add_state_columns(split0.test, edges)

    print("Learning confidence-weighted transitions...")
    transitions, state_priors = build_transition_model(train)
    print("Learning state/event price impact tables...")
    state_risk, event_time_risk, event_time_move, global_rate = build_impact_tables(train)
    print("Running value iteration...")
    values = value_iteration(transitions, state_risk, global_rate)

    print("Scoring train/validation/test rows...")
    train_scored = add_markov_scores(train, state_risk, event_time_risk, event_time_move, values, global_rate)
    val_scored = add_markov_scores(val, state_risk, event_time_risk, event_time_move, values, global_rate)
    test_scored = add_markov_scores(test, state_risk, event_time_risk, event_time_move, values, global_rate)

    print("Selecting validation threshold...")
    threshold, val_selection = select_threshold(
        val_scored[TARGET].to_numpy(dtype=int),
        val_scored["markov_path_score"].to_numpy(dtype=float),
        precision_floor=0.30,
    )
    print("Evaluating test split...")
    test_metrics = evaluate(
        test_scored[TARGET].to_numpy(dtype=int),
        test_scored["markov_path_score"].to_numpy(dtype=float),
        threshold,
    )

    test_out_cols = [
        "fixture_id",
        "timestamp",
        "incident_name",
        "mid_price",
        "minutes_remaining",
        "minutes_elapsed",
        "score_diff",
        "total_goals",
        "period_id",
        "xt_weight",
        "confidence_grade",
        "conf_trend",
        "event_density_30s",
        "event_density_60s",
        "event_density_120s",
        "event_density_300s",
        "risk_event_count_30s",
        "risk_event_count_60s",
        "high_impact_60s",
        "high_impact_300s",
        "xt_sum_60s",
        "xt_sum_300s",
        "price_velocity",
        "price_realized_vol",
        "price_change_1m",
        "high_volatility",
        "max_abs_move_120s",
        "next_60s_high_impact",
        "event_group",
        "score_bucket",
        "time_bucket",
        "period_bucket",
        "momentum_bucket",
        "confidence_bucket",
        "markov_state_risk",
        "markov_event_risk",
        "markov_event_move",
        "markov_state_value_raw",
        "markov_path_score",
        "markov_path_confidence",
        "markov_jump_level",
    ]
    pred = test_scored[[c for c in test_out_cols if c in test_scored.columns]].copy()
    pred["markov_predicted_volatile"] = (pred["markov_path_score"] >= threshold).astype(int)
    pred.to_parquet(OUT / "markov_xt_predictions.parquet", index=False)

    heatmap = export_heatmap(train_scored, test_scored)
    print("Mining top 3-event paths...")
    top_paths = export_top_paths(test_scored)
    state_values = pd.DataFrame(
        [
            {
                "markov_state": state,
                "state_value": float(value),
                "state_risk": float(state_risk.get(state, global_rate)),
                "transition_weight": float(state_priors.get(state, 0.0)),
            }
            for state, value in values.items()
        ]
    ).sort_values("state_value", ascending=False)
    state_values.to_parquet(OUT / "markov_state_values.parquet", index=False)

    metrics = {
        "model": "Markov xT v1",
        "objective": "recall_first_with_explainability",
        "data_file": str(DATA_FILE.relative_to(BASE)),
        "rows": int(len(df)),
        "fixtures": int(df["fixture_id"].nunique()),
        "train_rows": int(len(train_scored)),
        "val_rows": int(len(val_scored)),
        "test_rows": int(len(test_scored)),
        "train_fixtures": int(train_scored["fixture_id"].nunique()),
        "val_fixtures": int(val_scored["fixture_id"].nunique()),
        "test_fixtures": int(test_scored["fixture_id"].nunique()),
        "global_train_volatility_rate": round(float(global_rate), 4),
        "momentum_edges": [round(float(x), 4) for x in edges],
        "transition_states": int(len(transitions)),
        "valued_states": int(len(values)),
        "validation_selection": {k: round(float(v), 4) for k, v in val_selection.items()},
        "test": test_metrics,
        "outputs": {
            "predictions": "outputs/markov_xt_predictions.parquet",
            "metrics": "outputs/markov_xt_metrics.json",
            "heatmap": "outputs/markov_event_heatmap.json",
            "top_paths": "outputs/markov_top_paths.json",
            "state_values": "outputs/markov_state_values.parquet",
        },
    }

    (OUT / "markov_xt_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (OUT / "markov_event_heatmap.json").write_text(json.dumps(heatmap, ensure_ascii=False, indent=2))
    (OUT / "markov_top_paths.json").write_text(json.dumps(top_paths, ensure_ascii=False, indent=2))

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
