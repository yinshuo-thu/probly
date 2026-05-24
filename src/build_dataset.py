"""
Task 4 - Part 1: Build training dataset by joining LSports events with Polymarket prices.

For each matched fixture:
1. Load LSports messages (compute rolling features)
2. Load Polymarket price data (compute volatility labels)
3. Join on timestamp
4. Output one big dataset for ML training
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

MATCHES_PATH = "data/polymarket/fixture_market_matches.parquet"
PRICES_DIR = "data/polymarket/prices"
HYPER_DIR = "data/hyper/football"
OUTPUT_PATH = "outputs/training_dataset.parquet"
MIN_SCORE = 0.75

# Key incident types we care about for features
KEY_INCIDENTS = {
    "Score": "score",
    "DangerousAttacks": "dangerous_attack",
    "Attacks": "attack",
    "PlayerShotsOnTarget": "shot_on_target",
    "PlayerShotsOffTarget": "shot_off_target",
    "Player Yellow Card": "yellow_card",
    "Player Red Card": "red_card",
    "Corner Kicks": "corner",
    "Goal": "goal",
    "Timer": "timer",
    "Period": "period",
}

INCIDENT_WEIGHT = {
    "goal": 5.0,
    "red_card": 4.0,
    "shot_on_target": 2.0,
    "shot_off_target": 1.2,
    "corner": 1.5,
    "dangerous_attack": 1.0,
    "attack": 0.5,
    "yellow_card": 0.8,
    "timer": 0.1,
}


def load_messages(fixture_id, event_date):
    """Load LSports messages for a fixture, return cleaned DataFrame."""
    path = os.path.join(HYPER_DIR, event_date, str(fixture_id), "messages.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    if len(df) == 0:
        return None

    # Parse timestamps
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df = df.sort_values("timestamp_utc")

    # Map incident names
    df["incident_key"] = df["incident_name"].map(KEY_INCIDENTS).fillna("other")
    df["event_weight"] = df["incident_key"].map(INCIDENT_WEIGHT).fillna(0.1)

    # Parse score
    score_df = df[df["incident_name"] == "Score"].copy()
    if len(score_df) > 0:
        score_df["score_home"] = pd.to_numeric(score_df["home_value"], errors="coerce").fillna(0)
        score_df["score_away"] = pd.to_numeric(score_df["away_value"], errors="coerce").fillna(0)
        # Forward-fill scores onto all events
        score_df = score_df[["timestamp_utc", "score_home", "score_away"]].sort_values("timestamp_utc")
        df = df.merge(score_df.drop_duplicates("timestamp_utc"), on="timestamp_utc", how="left")
        df["score_home"] = df["score_home"].ffill().fillna(0)
        df["score_away"] = df["score_away"].ffill().fillna(0)
    else:
        df["score_home"] = 0
        df["score_away"] = 0

    # Period
    df["period_id"] = pd.to_numeric(df["period_id"], errors="coerce").ffill().fillna(10)

    # Confidence
    df["confidence_grade"] = pd.to_numeric(df["confidence_grade"], errors="coerce").fillna(0.5)

    # Seconds elapsed
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce").fillna(0)

    return df


def compute_rolling_features(df, ts_ns, windows_sec=[30, 60, 120]):
    """Compute rolling features for each row in df using pandas rolling on a temp frame."""
    df2 = df.copy()
    df2["_ts"] = pd.to_datetime(ts_ns, utc=True)
    df2 = df2.set_index("_ts")

    is_shot = (df["incident_key"] == "shot_on_target").astype(float)
    is_danger = (df["incident_key"] == "dangerous_attack").astype(float)

    features = {}
    for w in windows_sec:
        ws = f"{w}s"
        roll = df2["event_weight"].rolling(ws)
        features[f"event_density_{w}s"] = roll.count().values
        features[f"weight_sum_{w}s"] = roll.sum().values
        conf_roll = df2["confidence_grade"].rolling(ws)
        features[f"conf_mean_{w}s"] = conf_roll.mean().fillna(0.5).values
        features[f"conf_min_{w}s"] = conf_roll.min().fillna(1.0).values
        features[f"shot_on_target_{w}s"] = is_shot.set_axis(df2.index).rolling(ws).sum().fillna(0).values
        features[f"dangerous_attack_{w}s"] = is_danger.set_axis(df2.index).rolling(ws).sum().fillna(0).values

    scores_home = df["score_home"].values
    scores_away = df["score_away"].values
    seconds = df["seconds"].values
    period = df["period_id"].values
    conf = df["confidence_grade"].values

    features["score_home"] = scores_home
    features["score_away"] = scores_away
    features["score_diff"] = scores_home - scores_away
    features["total_goals"] = scores_home + scores_away
    features["seconds_elapsed"] = seconds
    features["period_id"] = period
    features["minutes_remaining"] = np.clip(90 - seconds / 60.0, 0, 120)
    features["confidence_grade"] = conf

    return pd.DataFrame(features, index=df.index)


def load_prices(condition_id):
    """Load PMXT price data for a condition_id.

    Uses best_bid/best_ask mid for the YES asset_id (the one with higher mid price
    at start), resampled to 10-second bars to track market probability.
    """
    safe_cid = condition_id.replace("0x", "")[:32]
    path = os.path.join(PRICES_DIR, f"{safe_cid}.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    if len(df) == 0:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["best_bid"] = pd.to_numeric(df["best_bid"], errors="coerce").fillna(0)
    df["best_ask"] = pd.to_numeric(df["best_ask"], errors="coerce").fillna(0)
    df["mid"] = (df["best_bid"] + df["best_ask"]) / 2.0
    df = df.sort_values("timestamp")

    # Pick the YES asset (the one that starts with mid > 0.5, or just take the first)
    if "asset_id" in df.columns and df["asset_id"].notna().any():
        asset_mids = df.groupby("asset_id")["mid"].first().sort_values(ascending=False)
        yes_asset = asset_mids.index[0]  # highest starting mid = YES
        df = df[df["asset_id"] == yes_asset]

    # Keep mid > 0 rows
    df = df[df["mid"] > 0]
    if len(df) < 10:
        return None

    # Deduplicate by timestamp (keep last)
    df = df.drop_duplicates("timestamp", keep="last")

    return df[["timestamp", "mid"]].rename(columns={"mid": "price"})


def label_volatility(price_df, ts_ns_prices, lookahead_sec=120, threshold_pct=0.03):
    """For each price point, compute forward volatility label."""
    prices = price_df["price"].values.astype(float)
    labels = np.zeros(len(price_df), dtype=int)
    max_moves = np.zeros(len(price_df), dtype=float)

    for i in range(len(price_df)):
        p0 = prices[i]
        if p0 <= 0:
            continue
        t0 = ts_ns_prices[i]
        mask = (ts_ns_prices > t0) & (ts_ns_prices <= t0 + lookahead_sec * 1_000_000_000)
        if mask.sum() > 0:
            future = prices[mask]
            move = np.abs(future - p0).max() / p0
            max_moves[i] = move
            labels[i] = int(move >= threshold_pct)

    return labels, max_moves


def join_lsports_to_prices(events_df, prices_df, lookahead_sec=120):
    """
    For each LSports event, find the nearest prior price and compute
    forward volatility. Uses vectorized merge_asof for efficiency.
    """
    # Build price series indexed by time
    price_series = prices_df[["timestamp", "price"]].sort_values("timestamp").copy()
    price_series["price"] = price_series["price"].astype(float)

    events_sorted = events_df[["timestamp_utc"]].copy()
    events_sorted["timestamp"] = events_sorted["timestamp_utc"]
    events_sorted = events_sorted.sort_values("timestamp")

    # Normalize timestamps to same resolution
    events_sorted["timestamp"] = events_sorted["timestamp"].dt.as_unit("us")
    price_series["timestamp"] = price_series["timestamp"].dt.as_unit("us")

    # merge_asof: nearest prior price for each event
    merged = pd.merge_asof(
        events_sorted,
        price_series.rename(columns={"price": "current_price"}),
        on="timestamp",
        direction="backward",
    )
    current_price = merged["current_price"].reindex(events_df.index).fillna(0).values

    # Forward volatility: use vectorized rolling on the price series
    pr_ts = price_series["timestamp"].values.astype(np.int64)
    prices = price_series["price"].values
    ev_ts = events_df["timestamp_utc"].dt.as_unit("us").values.astype(np.int64)

    labels = np.zeros(len(events_df), dtype=int)
    max_moves = np.zeros(len(events_df), dtype=float)
    price_std_arr = np.zeros(len(events_df), dtype=float)

    la_ns = lookahead_sec * 1_000_000  # in microseconds

    # Use searchsorted for fast window indexing
    for i, t0 in enumerate(ev_ts):
        p0 = current_price[i]
        if p0 <= 0:
            continue
        # Find future prices in window
        lo = np.searchsorted(pr_ts, t0 + 1)
        hi = np.searchsorted(pr_ts, t0 + la_ns, side="right")
        if hi > lo:
            future = prices[lo:hi]
            move = np.abs(future - p0).max() / p0
            max_moves[i] = move
            price_std_arr[i] = future.std()
            labels[i] = int(move >= 0.03)

    return current_price, labels, max_moves, price_std_arr


def build_fixture_dataset(fixture_id, event_date, condition_id, market_type):
    """Build feature-label dataset for one fixture-market pair."""
    msgs = load_messages(fixture_id, event_date)
    if msgs is None or len(msgs) < 50:
        return None

    prices = load_prices(condition_id)
    if prices is None or len(prices) < 50:
        return None

    # Only keep in-game events (during match hours)
    match_start = prices["timestamp"].min()
    match_end = prices["timestamp"].max()
    msgs = msgs[(msgs["timestamp_utc"] >= match_start) & (msgs["timestamp_utc"] <= match_end)]
    if len(msgs) < 10:
        return None

    # Compute rolling features
    ts_ns = msgs["timestamp_utc"].values.astype(np.int64)
    feat_df = compute_rolling_features(msgs, ts_ns)
    feat_df["timestamp_utc"] = msgs["timestamp_utc"].values

    # Join prices → labels
    cur_price, labels, max_moves, price_std = join_lsports_to_prices(msgs, prices)

    feat_df["current_price"] = cur_price
    feat_df["high_volatility_label"] = labels
    feat_df["max_price_move_pct"] = max_moves
    feat_df["price_std_120s"] = price_std
    feat_df["fixture_id"] = fixture_id
    feat_df["condition_id"] = condition_id
    feat_df["market_type"] = market_type
    feat_df["event_date"] = event_date

    # Drop rows with no price data
    feat_df = feat_df[feat_df["current_price"] > 0]

    return feat_df


def main():
    os.makedirs("outputs", exist_ok=True)

    print("Loading matches...")
    matches = pd.read_parquet(MATCHES_PATH)
    top = matches[matches["match_score"] >= MIN_SCORE].copy()
    print(f"Processing {len(top)} high-confidence markets across {top['fixture_id'].nunique()} fixtures")

    all_datasets = []
    processed = 0
    skipped_no_msg = 0
    skipped_no_price = 0

    for i, (_, row) in enumerate(top.iterrows()):
        fixture_id = int(row["fixture_id"])
        event_date = str(row["event_date"])
        condition_id = row["condition_id"]
        market_type = row["market_type"]
        home = row["home"]
        away = row["away"]

        print(f"\n[{i+1}/{len(top)}] {home} vs {away} | {market_type[:30]} | {condition_id[:20]}...")

        ds = build_fixture_dataset(fixture_id, event_date, condition_id, market_type)
        if ds is None:
            # Diagnose why
            if load_messages(fixture_id, event_date) is None:
                print(f"  SKIP: no messages data")
                skipped_no_msg += 1
            elif load_prices(condition_id) is None:
                print(f"  SKIP: no price data")
                skipped_no_price += 1
            else:
                print(f"  SKIP: insufficient data after filtering")
            continue

        label_rate = ds["high_volatility_label"].mean()
        print(f"  OK: {len(ds)} rows, label_rate={label_rate:.3f}, "
              f"price_range=[{ds['current_price'].min():.3f}, {ds['current_price'].max():.3f}]")
        all_datasets.append(ds)
        processed += 1

    print(f"\n\nSummary:")
    print(f"  Processed: {processed} markets")
    print(f"  Skipped (no messages): {skipped_no_msg}")
    print(f"  Skipped (no prices): {skipped_no_price}")

    if not all_datasets:
        print("ERROR: No datasets built!")
        return

    final_df = pd.concat(all_datasets, ignore_index=True)
    print(f"\nFull dataset: {len(final_df)} rows, {len(all_datasets)} market-fixture pairs")
    print(f"Label distribution: {final_df['high_volatility_label'].value_counts().to_dict()}")
    print(f"Unique fixtures: {final_df['fixture_id'].nunique()}")

    final_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Print feature column list
    feature_cols = [c for c in final_df.columns if c not in [
        "timestamp_utc", "fixture_id", "condition_id", "market_type",
        "event_date", "high_volatility_label", "max_price_move_pct", "price_std_120s"
    ]]
    print(f"\nFeature columns ({len(feature_cols)}):")
    print(feature_cols)


if __name__ == "__main__":
    main()
