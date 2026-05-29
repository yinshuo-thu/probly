"""Utilities for evaluating LSports pre-goal alert signals.

The alert score used in the prototype is not an official LSports feed field.
It is a model score computed from rolling event-flow features:

    y_t = 1{there is a goal in (t, t + horizon_sec]}
    hazard_t = sigmoid(beta_0 + sum_j beta_j * zscore(x_{t,j}))

Repeated threshold crossings are collapsed into alert episodes with a cooldown.
An episode is a true alert if a goal occurs within the prediction horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def alert_episodes(frame: pd.DataFrame, hazard: np.ndarray, threshold: float,
                   cooldown_sec: int = 90) -> pd.DataFrame:
    """Collapse threshold crossings into alert episodes.

    An episode starts on an upward crossing from below threshold to above
    threshold. Staying above threshold does not create repeated alerts. The
    cooldown only prevents immediate re-alerts after a brief dip below the
    threshold.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["alert_ts"])
    ts = pd.to_datetime(frame["ts"], utc=True, format="mixed").reset_index(drop=True)
    h = pd.Series(hazard).reset_index(drop=True)
    rows, last = [], pd.NaT
    was_above = False
    for t, score in zip(ts, h):
        is_above = bool(score >= threshold)
        if is_above and not was_above:
            if pd.isna(last) or (t - last).total_seconds() >= cooldown_sec:
                rows.append({"alert_ts": t})
                last = t
        was_above = is_above
    return pd.DataFrame(rows)


def score_threshold(frame: pd.DataFrame, hazard: np.ndarray,
                    goals: pd.DataFrame, bbo_reaction: pd.DataFrame,
                    threshold: float, warning_sec: int = 180,
                    cooldown_sec: int = 90) -> dict:
    """Score one threshold against future goals and BBO reaction timestamps."""
    alerts = alert_episodes(frame, hazard, threshold, cooldown_sec=cooldown_sec)
    goals_ts = pd.to_datetime(goals["ts"], utc=True, format="mixed") \
        if goals is not None and not goals.empty else pd.Series(dtype="datetime64[ns, UTC]")
    bbo_ts = pd.to_datetime(bbo_reaction["reaction_ts"], utc=True, format="mixed").dropna() \
        if bbo_reaction is not None and not bbo_reaction.empty else pd.Series(dtype="datetime64[ns, UTC]")

    true_alerts = 0
    bbo_preempt = 0
    covered_goals = set()
    covered_bbo = set()
    lead_secs = []
    for _, a in alerts.iterrows():
        t = a["alert_ts"]
        future_goals = goals_ts[(goals_ts >= t) &
                                (goals_ts <= t + pd.Timedelta(seconds=warning_sec))]
        if not future_goals.empty:
            true_alerts += 1
            g = future_goals.iloc[0]
            covered_goals.add(str(g))
            lead_secs.append((g - t).total_seconds())
            future_bbo = bbo_ts[(bbo_ts >= t) &
                                (bbo_ts <= g + pd.Timedelta(seconds=10))]
            if not future_bbo.empty and t < future_bbo.iloc[0]:
                bbo_preempt += 1
                covered_bbo.add(str(future_bbo.iloc[0]))

    n_alerts = len(alerts)
    precision = true_alerts / n_alerts if n_alerts else np.nan
    recall = len(covered_goals) / len(goals_ts) if len(goals_ts) else np.nan
    return {
        "threshold": threshold,
        "n_alerts": n_alerts,
        "true_alerts": true_alerts,
        "false_alerts": n_alerts - true_alerts,
        "precision": precision,
        "goal_recall": recall,
        "bbo_preempt_alerts": bbo_preempt,
        "unique_bbo_moves_preempted": len(covered_bbo),
        "mean_goal_lead_sec": np.mean(lead_secs) if lead_secs else np.nan,
    }


def markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table formatter."""
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)
