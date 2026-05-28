"""
Phase 3: Lead-Time Signal Detection
Question: In the 0-120 seconds BEFORE a goal, do LSports event patterns
(DangerousAttacks, ShotsOnTarget, etc.) diverge from baseline?
If yes → we have a lead-time signal for price moves.

Method:
1. Identify all confirmed goals (confidence→1.0) per fixture
2. For each goal, count events in windows: [-120s,-60s], [-60s,-30s], [-30s,0s]
3. Compare to background rate (same event type, outside goal windows)
4. Compute lift ratio: goal_window_rate / background_rate

Also builds the price-move prediction dataset:
- Feature: event density in last Ns
- Label: price move in next Xs

Outputs:
- outputs/lead_signal_stats.json: lift ratios per event type
- outputs/prediction_dataset.parquet: ML-ready feature/label pairs
"""

from __future__ import annotations
from typing import Optional, List, Dict, Tuple

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import glob
import json
import os
from pathlib import Path

DATA_ROOT = Path("data")
OUT_DIR = Path("outputs")

SIGNAL_EVENTS = [
    "DangerousAttacks", "Attacks", "ShotsOnTarget", "ShotsOffTarget",
    "BlockedShots", "Corners", "FreeKicks",
]

GOAL_WINDOWS = [-120, -60, -30, -10]  # seconds before goal


def load_lsports(fixture_id: int) -> Optional[pd.DataFrame]:
    files = glob.glob(str(DATA_ROOT / f"hyper/football/*/{fixture_id}/messages.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[0])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def extract_confirmed_goals(msgs: pd.DataFrame) -> List[Dict]:
    """
    Extract goals that are definitively confirmed (confidence reaches 1.0).
    Returns list of dicts with goal metadata.
    """
    score_events = msgs[msgs["incident_name"] == "Score"].copy()
    score_events["home_score"] = pd.to_numeric(score_events["home_value"], errors="coerce").fillna(0).astype(int)
    score_events["away_score"] = pd.to_numeric(score_events["away_value"], errors="coerce").fillna(0).astype(int)
    score_events = score_events.sort_values("timestamp_utc").reset_index(drop=True)

    goals = []
    seen_scores = {}
    for _, row in score_events.iterrows():
        h, a = row["home_score"], row["away_score"]
        score_key = (h, a)
        conf = row["confidence_grade"]
        t = row["timestamp_utc"]
        elapsed = float(pd.to_numeric(row.get("seconds", 0), errors="coerce") or 0)

        if score_key not in seen_scores:
            seen_scores[score_key] = {
                "first_ts": t,
                "first_conf": conf,
                "confirmed_ts": t if conf == 1.0 else None,
                "home_score": h,
                "away_score": a,
                "elapsed_sec": elapsed,
            }
        else:
            if conf == 1.0 and seen_scores[score_key]["confirmed_ts"] is None:
                seen_scores[score_key]["confirmed_ts"] = t

    # Keep only score transitions (actual goals)
    sorted_scores = sorted(seen_scores.keys(), key=lambda k: (k[0] + k[1], k[0]))
    prev_total = 0
    for score_key in sorted_scores:
        h, a = score_key
        total = h + a
        if total > prev_total and seen_scores[score_key]["confirmed_ts"] is not None:
            info = seen_scores[score_key]
            goals.append({
                "home_score": h,
                "away_score": a,
                "total_goals": total,
                "goal_ts": info["first_ts"],  # when LSports first reported
                "confirmed_ts": info["confirmed_ts"],
                "first_conf": info["first_conf"],
                "elapsed_sec": info["elapsed_sec"],
            })
        prev_total = max(prev_total, total)

    return goals


def count_events_in_window(msgs: pd.DataFrame, t_start: pd.Timestamp,
                            t_end: pd.Timestamp, event_types: List[str]) -> Dict[str, int]:
    """Count events of each type in a time window."""
    window = msgs[
        (msgs["timestamp_utc"] >= t_start) &
        (msgs["timestamp_utc"] < t_end) &
        (msgs["incident_name"].isin(event_types))
    ]
    counts = window["incident_name"].value_counts()
    return {ev: int(counts.get(ev, 0)) for ev in event_types}


def compute_event_rates(msgs: pd.DataFrame, goals: List[Dict],
                         event_types: List[str]) -> Tuple[Dict, List[Dict]]:
    """
    Compute:
    1. Background event rates (events/minute outside goal windows)
    2. Goal-window rates at various lead times

    Returns: (background_rates, goal_window_records)
    """
    if not goals:
        return {}, []

    game_start = msgs["timestamp_utc"].min()
    game_end = msgs["timestamp_utc"].max()
    total_min = max(1, (game_end - game_start).total_seconds() / 60)

    # Mark goal-adjacent windows (120s before each goal) to exclude from background
    goal_windows = [(g["goal_ts"] - pd.Timedelta(seconds=150),
                     g["goal_ts"] + pd.Timedelta(seconds=60))
                    for g in goals]

    # Filter non-goal events
    def in_goal_window(t):
        return any(ws <= t <= we for ws, we in goal_windows)

    bg_events = msgs[
        msgs["incident_name"].isin(event_types) &
        ~msgs["timestamp_utc"].apply(in_goal_window)
    ]

    bg_duration_min = total_min - len(goals) * 3.5  # approximate: 3.5 min per goal window
    bg_duration_min = max(1, bg_duration_min)

    background_rates = {}
    for ev in event_types:
        n = (bg_events["incident_name"] == ev).sum()
        background_rates[ev] = float(n) / bg_duration_min

    # Per-goal pre-window counts
    goal_records = []
    for goal in goals:
        t_goal = goal["goal_ts"]
        record = {
            "goal_ts": t_goal,
            "home_score": goal["home_score"],
            "away_score": goal["away_score"],
            "elapsed_sec": goal["elapsed_sec"],
            "first_conf": goal["first_conf"],
        }

        for window_s in [120, 90, 60, 30, 10]:
            t_start = t_goal - pd.Timedelta(seconds=window_s)
            counts = count_events_in_window(msgs, t_start, t_goal, event_types)
            for ev, cnt in counts.items():
                rate = float(cnt) / (window_s / 60)
                record[f"{ev}_{window_s}s_rate"] = rate
                record[f"{ev}_{window_s}s_count"] = cnt
                bg_rate = background_rates.get(ev, 0.001)
                record[f"{ev}_{window_s}s_lift"] = rate / max(bg_rate, 0.01)

        goal_records.append(record)

    return background_rates, goal_records


def build_prediction_features(fixture_id: int, msgs: pd.DataFrame,
                               prices: Optional[pd.DataFrame] = None) -> List[Dict]:
    """
    Build sliding-window feature rows for prediction.
    At each timer tick (every ~30 seconds), compute:
    - Event density features (last 30/60/120s)
    - Game state (score, elapsed, period)
    - Target: will there be a price move >3¢ in the next 60 seconds?
    """
    event_types = SIGNAL_EVENTS

    # Sample at every timer event (≈1 per minute)
    timer_events = msgs[msgs["incident_name"] == "Timer"].copy()
    timer_events = timer_events.sort_values("timestamp_utc").reset_index(drop=True)

    # Get confirmed goals for labeling
    goals = extract_confirmed_goals(msgs)
    goal_times = [g["goal_ts"] for g in goals]

    # Price data as int64 ns arrays for fast lookup
    pr_ts = None
    pr_mid = None
    if prices is not None and len(prices) > 0:
        pr_ts = prices["timestamp"].values.astype("datetime64[ns]").astype("int64")
        pr_mid = prices["mid"].values

    # Forward-fill score
    score_ev = msgs[msgs["incident_name"] == "Score"].copy()
    score_ev["hs"] = pd.to_numeric(score_ev["home_value"], errors="coerce").fillna(0).astype(int)
    score_ev["as_"] = pd.to_numeric(score_ev["away_value"], errors="coerce").fillna(0).astype(int)
    score_ev = score_ev[score_ev["confidence_grade"] == 1.0].sort_values("timestamp_utc")

    score_ts_int = score_ev["timestamp_utc"].values.astype("datetime64[ns]").astype("int64")
    hs_arr = score_ev["hs"].values
    as_arr = score_ev["as_"].values

    rows = []
    for _, ev in timer_events.iterrows():
        t = ev["timestamp_utc"]
        elapsed_s = float(pd.to_numeric(ev.get("seconds", 0), errors="coerce") or 0)
        period = ev.get("period_name", "")

        if period not in ("1st Half", "2nd Half", "1st half", "2nd half"):
            continue

        # Current score
        t_int = t.value
        idx = np.searchsorted(score_ts_int, t_int, side="right") - 1
        if idx >= 0:
            h, a = int(hs_arr[idx]), int(as_arr[idx])
        else:
            h, a = 0, 0

        row = {
            "fixture_id": fixture_id,
            "timestamp_utc": t,
            "elapsed_sec": elapsed_s,
            "elapsed_min": elapsed_s / 60,
            "time_remaining_min": max(0, (90 * 60 - elapsed_s) / 60),
            "period": period,
            "home_score": h,
            "away_score": a,
            "total_goals": h + a,
            "score_diff": h - a,
        }

        # Event density features
        for window_s in [30, 60, 120]:
            t_start = t - pd.Timedelta(seconds=window_s)
            counts = count_events_in_window(msgs, t_start, t, event_types)
            for ev_type, cnt in counts.items():
                row[f"{ev_type}_{window_s}s"] = cnt

        # Danger score: weighted combination
        da_60 = row.get("DangerousAttacks_60s", 0)
        sot_60 = row.get("ShotsOnTarget_60s", 0)
        corner_60 = row.get("Corners_60s", 0)
        row["danger_score_60s"] = da_60 * 1.0 + sot_60 * 2.0 + corner_60 * 1.5

        # Label: is there a goal in next 0-120 seconds?
        for horizon_s in [30, 60, 120]:
            t_end_goal = t + pd.Timedelta(seconds=horizon_s)
            has_goal = any(g >= t and g <= t_end_goal for g in goal_times)
            row[f"goal_in_{horizon_s}s"] = int(has_goal)

        # Label: price move in next 60 seconds (if prices available)
        if pr_ts is not None:
            t_int = t.value  # int64 ns
            # Ensure it's int64 ns
            if hasattr(t, 'value'):
                t_int = t.value
            else:
                t_int = int(np.datetime64(t, 'ns').astype('int64'))

            idx_before = np.searchsorted(pr_ts, t_int, side="right") - 1
            if idx_before >= 0:
                mid_now = pr_mid[idx_before]
                for horizon_s in [30, 60, 120]:
                    t_future = t_int + int(horizon_s * 1e9)
                    idx_future = np.searchsorted(pr_ts, t_future, side="right") - 1
                    if idx_future > idx_before:
                        mid_future = pr_mid[idx_future]
                        row[f"price_delta_{horizon_s}s"] = mid_future - mid_now
                        row[f"abs_price_delta_{horizon_s}s"] = abs(mid_future - mid_now)
                    else:
                        row[f"price_delta_{horizon_s}s"] = np.nan
                        row[f"abs_price_delta_{horizon_s}s"] = np.nan

        rows.append(row)

    return rows


def main():
    fm = pd.read_parquet(DATA_ROOT / "polymarket/fixture_market_matches.parquet")
    fi = pd.read_parquet(DATA_ROOT / "fixture_index.parquet").set_index("fixture_id")

    lsports_ids = {
        int(Path(f).parent.name)
        for f in glob.glob(str(DATA_ROOT / "hyper/football/*/*/messages.parquet"))
    }
    fixture_cids = fm[fm["market_type"] == "O/U 2.5"].groupby("fixture_id")["condition_id"].first().to_dict()
    overlap = sorted(lsports_ids & set(fm["fixture_id"].unique()))

    print(f"Analyzing {len(overlap)} fixtures for lead-time signals")

    # === Part 1: Pre-goal event lift analysis ===
    all_bg_rates = []
    all_goal_records = []

    for fid in overlap:
        msgs = load_lsports(fid)
        if msgs is None:
            continue
        goals = extract_confirmed_goals(msgs)
        if not goals:
            continue
        bg_rates, goal_records = compute_event_rates(msgs, goals, SIGNAL_EVENTS)
        for rec in goal_records:
            rec["fixture_id"] = fid
        all_bg_rates.append(bg_rates)
        all_goal_records.extend(goal_records)

    # Aggregate lift ratios
    if all_goal_records:
        goal_df = pd.DataFrame(all_goal_records)
        print(f"\n=== PRE-GOAL LIFT RATIOS (n={len(goal_df)} goals) ===")
        print("Event rate lift in windows before goal (vs background):\n")

        summary_rows = []
        for ev in SIGNAL_EVENTS:
            for window_s in [120, 60, 30, 10]:
                col = f"{ev}_{window_s}s_lift"
                if col in goal_df.columns:
                    vals = goal_df[col].dropna()
                    if len(vals) > 0:
                        summary_rows.append({
                            "event": ev,
                            "window_s": window_s,
                            "mean_lift": vals.mean(),
                            "median_lift": vals.median(),
                            "n": len(vals),
                        })

        lift_df = pd.DataFrame(summary_rows)
        if not lift_df.empty:
            pivot = lift_df.pivot(index="event", columns="window_s", values="mean_lift")
            pivot.columns = [f"{c}s before goal" for c in pivot.columns]
            print(pivot.round(2).to_string())

        lift_summary = {}
        for ev in SIGNAL_EVENTS:
            lift_summary[ev] = {}
            for window_s in [120, 60, 30, 10]:
                col = f"{ev}_{window_s}s_lift"
                if col in goal_df.columns:
                    lift_summary[ev][f"{window_s}s"] = float(goal_df[col].mean())

        with open(OUT_DIR / "lead_signal_stats.json", "w") as f:
            json.dump(lift_summary, f, indent=2)

    # === Part 2: Build ML prediction dataset ===
    print("\n=== BUILDING PREDICTION DATASET ===")

    all_feature_rows = []
    for fid in overlap:
        msgs = load_lsports(fid)
        if msgs is None:
            continue

        # Load O/U 2.5 price if available
        cid = fixture_cids.get(fid)
        prices = None
        if cid:
            from fair_value_model import load_prices_for_condition, select_yes_asset, poisson_ou_fair_price
            from fair_value_model import get_lambda, LEAGUE_LAMBDA
            league = fi.loc[fid, "league_name"] if fid in fi.index else ""
            lam = get_lambda(league)
            pregame_fair = poisson_ou_fair_price(2.5, 0, 90, lam)
            raw_prices = load_prices_for_condition(cid)
            if raw_prices is not None:
                prices = select_yes_asset(raw_prices, pregame_fair)

        rows = build_prediction_features(fid, msgs, prices)
        if rows:
            league = fi.loc[fid, "league_name"] if fid in fi.index else ""
            for r in rows:
                r["league_name"] = league
            all_feature_rows.extend(rows)
            print(f"  {fid}: {len(rows)} feature rows")

    if all_feature_rows:
        pred_df = pd.DataFrame(all_feature_rows)
        out_path = OUT_DIR / "prediction_dataset.parquet"
        pred_df.to_parquet(out_path, index=False)

        print(f"\nPrediction dataset: {len(pred_df):,} rows")
        print(f"Goal in next 60s: {pred_df['goal_in_60s'].mean():.1%} base rate")
        print(f"Goal in next 30s: {pred_df['goal_in_30s'].mean():.1%} base rate")
        print(f"Goal in next 120s: {pred_df['goal_in_120s'].mean():.1%} base rate")

        if "abs_price_delta_60s" in pred_df.columns:
            valid = pred_df["abs_price_delta_60s"].dropna()
            print(f"Price move >3¢ in 60s: {(valid > 0.03).mean():.1%} base rate")
            print(f"Price move >5¢ in 60s: {(valid > 0.05).mean():.1%} base rate")

        # Simple analysis: danger score vs goal probability
        print("\nGoal rate by danger score quintile (next 60s):")
        pred_df["danger_q"] = pd.qcut(pred_df["danger_score_60s"].clip(0, 20),
                                       q=5, labels=["Q1\n(low)", "Q2", "Q3", "Q4", "Q5\n(high)"],
                                       duplicates="drop")
        goal_by_danger = pred_df.groupby("danger_q")["goal_in_60s"].agg(["mean", "count"])
        goal_by_danger.columns = ["goal_rate", "n"]
        print(goal_by_danger.round(3).to_string())

        print(f"\nSaved: {out_path}, outputs/lead_signal_stats.json")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent.parent)
    main()
