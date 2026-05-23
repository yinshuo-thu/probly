"""
Feature engineering for Polymarket volatility prediction.

Extracts Markov-state features, xT features, momentum, and temporal features
from the LSports Hyper event stream.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Markov state encoding ─────────────────────────────────────────────────────

SCORE_DIFF_BINS = [-10, -3, -2, -1, 0, 1, 2, 3, 10]  # bucket score differences
TIME_BINS = [0, 15, 30, 45, 60, 75, 90, 120]          # minute buckets


def encode_score_diff(home: int, away: int) -> int:
    """Bin score difference into discrete state."""
    diff = home - away
    for i, edge in enumerate(SCORE_DIFF_BINS[1:]):
        if diff < edge:
            return i
    return len(SCORE_DIFF_BINS) - 2


def encode_time_bucket(elapsed_minutes: float) -> int:
    """Bin elapsed time into period buckets."""
    for i, edge in enumerate(TIME_BINS[1:]):
        if elapsed_minutes < edge:
            return i
    return len(TIME_BINS) - 2


def build_markov_state(row: pd.Series) -> tuple:
    """Build a discrete Markov state tuple from a row."""
    score_home = row.get("score_home", row.get("ScoreHome", 0)) or 0
    score_away = row.get("score_away", row.get("ScoreAway", 0)) or 0
    elapsed = row.get("elapsed_time", row.get("ElapsedTime", row.get("Minute", 0))) or 0
    period = row.get("period", row.get("Period", 1)) or 1
    return (
        encode_score_diff(int(score_home), int(score_away)),
        int(period),
        encode_time_bucket(float(elapsed)),
    )


# ── xT (Expected Threat) grid ─────────────────────────────────────────────────

XT_GRID_X = 12  # pitch horizontal cells
XT_GRID_Y = 8   # pitch vertical cells

# Standard xT grid values (adapted from Karun Singh 2019, approximate)
_XT_BASE = np.array([
    [0.00, 0.00, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03],
    [0.00, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04],
    [0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.06],
    [0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.06, 0.10],
    [0.01, 0.01, 0.02, 0.03, 0.04, 0.06, 0.10, 0.15],
    [0.01, 0.02, 0.02, 0.03, 0.05, 0.08, 0.14, 0.20],
    [0.02, 0.02, 0.03, 0.05, 0.07, 0.12, 0.20, 0.30],
    [0.02, 0.02, 0.04, 0.06, 0.09, 0.15, 0.25, 0.35],
    [0.03, 0.03, 0.05, 0.08, 0.12, 0.20, 0.32, 0.42],
    [0.04, 0.04, 0.07, 0.10, 0.16, 0.26, 0.38, 0.47],
    [0.05, 0.06, 0.09, 0.14, 0.22, 0.34, 0.45, 0.52],
    [0.08, 0.10, 0.14, 0.20, 0.30, 0.43, 0.56, 0.65],
])  # shape (12, 8)

XT_GRID = _XT_BASE  # attacking direction = increasing x


def get_xt_value(x_norm: float, y_norm: float) -> float:
    """Get xT value for normalized pitch coordinates [0,1]."""
    xi = min(int(x_norm * XT_GRID_X), XT_GRID_X - 1)
    yi = min(int(y_norm * XT_GRID_Y), XT_GRID_Y - 1)
    return float(XT_GRID[xi, yi])


# ── Rolling feature extraction ────────────────────────────────────────────────

HIGH_RISK_EVENTS = {
    "GOAL", "PENALTY", "RED_CARD", "YELLOW_RED_CARD",
    "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "CORNER",
    "DANGEROUS_ATTACK", "ATTACK",
}

EVENT_WEIGHT = {
    "GOAL": 5.0,
    "PENALTY": 4.0,
    "RED_CARD": 4.0,
    "YELLOW_RED_CARD": 3.5,
    "SHOT_ON_TARGET": 2.0,
    "SHOT_OFF_TARGET": 1.2,
    "CORNER": 1.5,
    "DANGEROUS_ATTACK": 1.0,
    "ATTACK": 0.5,
    "POSSESSION": 0.1,
}


def extract_match_features(df: pd.DataFrame, windows_sec: list = [30, 60, 120]) -> pd.DataFrame:
    """
    Extract rolling features per event for a single match DataFrame.

    Input df must have columns: timestamp (datetime), event_type, confidence,
    score_home, score_away, elapsed_time, period.
    Output: df with new feature columns appended.
    """
    df = df.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ts_sec"] = df["timestamp"].astype(np.int64) // 1_000_000_000

    # Normalize event_type
    if "event_type" in df.columns:
        ev_col = "event_type"
    elif "EventType" in df.columns:
        ev_col = "EventType"
    else:
        ev_col = None

    # Markov state
    df["markov_state"] = df.apply(build_markov_state, axis=1)
    df["markov_score_bucket"] = df["markov_state"].apply(lambda s: s[0])
    df["markov_period"] = df["markov_state"].apply(lambda s: s[1])
    df["markov_time_bucket"] = df["markov_state"].apply(lambda s: s[2])

    # Event weight
    if ev_col:
        df["event_weight"] = df[ev_col].apply(
            lambda e: EVENT_WEIGHT.get(str(e).upper(), 0.1)
        )
        df["is_high_risk_event"] = df[ev_col].apply(
            lambda e: int(str(e).upper() in HIGH_RISK_EVENTS)
        )
    else:
        df["event_weight"] = 0.1
        df["is_high_risk_event"] = 0

    # xT value per event (if position data available)
    for xc, yc in [("x_norm", "y_norm"), ("XNorm", "YNorm"), ("PosX", "PosY")]:
        if xc in df.columns and yc in df.columns:
            df["xt_value"] = df.apply(
                lambda r: get_xt_value(r[xc] / 100.0, r[yc] / 100.0)
                if pd.notnull(r[xc]) else 0.0, axis=1
            )
            break
    else:
        df["xt_value"] = df["event_weight"] * 0.05  # fallback proxy

    # Confidence
    conf_col = "confidence" if "confidence" in df.columns else "Confidence"
    if conf_col in df.columns:
        df["confidence_val"] = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.5)
    else:
        df["confidence_val"] = 0.5

    # Rolling features over time windows
    ts = df["ts_sec"].values
    weights = df["event_weight"].values
    xt = df["xt_value"].values
    conf = df["confidence_val"].values
    risk = df["is_high_risk_event"].values

    for w in windows_sec:
        event_density = np.zeros(len(df))
        xt_sum = np.zeros(len(df))
        conf_mean = np.zeros(len(df))
        risk_count = np.zeros(len(df))

        for i in range(len(df)):
            mask = (ts >= ts[i] - w) & (ts <= ts[i])
            event_density[i] = mask.sum()
            xt_sum[i] = xt[mask].sum()
            conf_mean[i] = conf[mask].mean() if mask.sum() > 0 else 0.5
            risk_count[i] = risk[mask].sum()

        df[f"event_density_{w}s"] = event_density
        df[f"xt_sum_{w}s"] = xt_sum
        df[f"conf_mean_{w}s"] = conf_mean
        df[f"risk_event_count_{w}s"] = risk_count

    # Score-based features
    sh = pd.to_numeric(df.get("score_home", df.get("ScoreHome", 0)), errors="coerce").fillna(0)
    sa = pd.to_numeric(df.get("score_away", df.get("ScoreAway", 0)), errors="coerce").fillna(0)
    df["score_diff"] = (sh - sa).astype(int)
    df["total_goals"] = (sh + sa).astype(int)

    # Time remaining (approx for 90-min match)
    elapsed = pd.to_numeric(
        df.get("elapsed_time", df.get("ElapsedTime", df.get("Minute", 0))),
        errors="coerce"
    ).fillna(0)
    df["minutes_remaining"] = (90 - elapsed).clip(lower=0)
    df["game_intensity"] = df["total_goals"] / (elapsed + 1)

    return df


def label_volatility(df: pd.DataFrame,
                     price_col: str = "polymarket_price",
                     lookahead_sec: int = 120,
                     threshold_pct: float = 0.03) -> pd.DataFrame:
    """
    Label each event row with whether high Polymarket volatility occurs
    in the next `lookahead_sec` seconds.

    threshold_pct: minimum fractional price move to count as high-volatility.
    """
    df = df.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ts = df["timestamp"].values.astype(np.int64) // 1_000_000_000
    prices = df[price_col].values

    labels = np.zeros(len(df), dtype=int)
    max_move = np.zeros(len(df), dtype=float)

    for i in range(len(df)):
        mask = (ts > ts[i]) & (ts <= ts[i] + lookahead_sec)
        if mask.sum() > 0 and prices[i] > 0:
            future_prices = prices[mask]
            move = np.abs(future_prices - prices[i]).max() / prices[i]
            max_move[i] = move
            if move >= threshold_pct:
                labels[i] = 1

    df["high_volatility_label"] = labels
    df["max_price_move_pct"] = max_move
    return df


def volatility_grade(move_pct: float) -> int:
    """Assign L1-L5 volatility grade from price move percentage."""
    if move_pct < 0.01:
        return 0
    elif move_pct < 0.02:
        return 1
    elif move_pct < 0.05:
        return 2
    elif move_pct < 0.10:
        return 3
    elif move_pct < 0.20:
        return 4
    else:
        return 5
