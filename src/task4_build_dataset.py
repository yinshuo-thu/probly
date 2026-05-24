#!/usr/bin/env python3
"""Task 4: Build aligned dataset from LSports messages and PMXT prices."""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=== Task 4: Building aligned dataset ===")

matched_df = pd.read_parquet('/Volumes/T7/probly/data/matched_fixtures.parquet')
MESSAGES_BASE = Path('/Volumes/T7/probly/data/hyper/football')
PRICES_DIR = Path('/Volumes/T7/probly/data/polymarket/prices')
OUTPUT_DIR = Path('/Volumes/T7/probly/outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# xT threat weights by incident type
THREAT_WEIGHTS = {
    'goal': 1.0,
    'shot on target': 0.6,
    'shot off target': 0.3,
    'penalty': 0.9,
    'red card': 0.8,
    'yellow card': 0.3,
    'corner kick': 0.25,
    'free kick': 0.2,
    'dangerous attack': 0.35,
    'offside': 0.1,
    'substitution': 0.05,
    'injury': 0.4,
    'var': 0.5,
    'score': 0.7,
    'period': 0.1,
}

RISK_INCIDENTS = {'goal', 'penalty', 'red card', 'shot on target', 'var', 'injury', 'score'}

def get_threat_weight(incident_name):
    if pd.isna(incident_name):
        return 0.0
    name_lower = str(incident_name).lower()
    for k, v in THREAT_WEIGHTS.items():
        if k in name_lower:
            return v
    return 0.05

def is_risk_event(incident_name):
    if pd.isna(incident_name):
        return False
    name_lower = str(incident_name).lower()
    return any(r in name_lower for r in RISK_INCIDENTS)

def compute_price_features(price_df):
    """Compute mid-price and price changes."""
    df = price_df.copy()
    # Convert to UTC, handling any timezone
    ts = pd.to_datetime(df['timestamp'], errors='coerce')
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize('UTC')
    else:
        ts = ts.dt.tz_convert('UTC')
    df['timestamp'] = ts
    df = df.dropna(subset=['timestamp'])
    df['best_bid'] = pd.to_numeric(df['best_bid'], errors='coerce').fillna(0)
    df['best_ask'] = pd.to_numeric(df['best_ask'], errors='coerce').fillna(1)
    df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
    df = df.sort_values('timestamp').reset_index(drop=True)
    # Compute per-condition_id price changes
    df['price_change'] = df.groupby('condition_id')['mid_price'].diff().fillna(0)
    return df

def label_volatility(price_df, event_ts_s, window_s=120, threshold=0.03):
    """Label: 1 if max |price_change| in next window_s seconds > threshold.
    event_ts_s: Unix seconds float
    """
    end_ts_s = event_ts_s + window_s
    mask = (price_df['ts_s'] >= event_ts_s) & (price_df['ts_s'] <= end_ts_s)
    future = price_df.loc[mask, 'price_change'].abs()
    if len(future) == 0:
        return 0
    return 1 if future.max() > threshold else 0

def build_features_for_fixture(messages_df, price_df, fixture_id):
    """Build feature rows by aligning messages to prices."""
    # Parse timestamps
    messages_df = messages_df.copy()
    messages_df['timestamp_utc'] = pd.to_datetime(
        messages_df['timestamp_utc'], utc=True, errors='coerce'
    )
    messages_df = messages_df.dropna(subset=['timestamp_utc'])
    messages_df = messages_df.sort_values('timestamp_utc').reset_index(drop=True)

    if len(messages_df) == 0 or len(price_df) == 0:
        return pd.DataFrame()

    price_df = compute_price_features(price_df)
    # Precompute Unix seconds for fast lookup
    price_df['ts_s'] = np.array([t.timestamp() for t in price_df['timestamp']])

    # Use numpy for fast nearest-timestamp lookup
    # Use pd.Timestamp.timestamp() for correct Unix seconds regardless of resolution
    price_ts_s = price_df['ts_s'].values

    rows = []
    for i, msg_row in messages_df.iterrows():
        ts = msg_row['timestamp_utc']
        # Ensure ts is UTC
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        ts_s = ts.timestamp()  # Unix seconds (float)

        # Find nearest price within 5 minutes (300s)
        diffs = np.abs(price_ts_s - ts_s)
        nearest_idx = diffs.argmin()
        min_diff_s = diffs[nearest_idx]

        if min_diff_s > 300:
            continue  # No price within 5 minutes

        nearest_price = price_df.iloc[nearest_idx]

        # Label: high volatility in next 120s
        label = label_volatility(price_df, ts_s, window_s=120, threshold=0.03)

        # Rolling event density features (last 30s, 60s)
        ts30 = ts - pd.Timedelta(seconds=30)
        ts60 = ts - pd.Timedelta(seconds=60)
        mask30 = (messages_df['timestamp_utc'] >= ts30) & (messages_df['timestamp_utc'] <= ts)
        mask60 = (messages_df['timestamp_utc'] >= ts60) & (messages_df['timestamp_utc'] <= ts)

        window30 = messages_df[mask30]
        window60 = messages_df[mask60]

        event_density_30s = len(window30)
        event_density_60s = len(window60)
        risk_event_count_30s = window30['incident_name'].apply(is_risk_event).sum()
        risk_event_count_60s = window60['incident_name'].apply(is_risk_event).sum()

        # Markov state: score
        home_val = msg_row.get('home_value', None)
        away_val = msg_row.get('away_value', None)
        try:
            score_home = int(float(home_val)) if pd.notna(home_val) else 0
        except:
            score_home = 0
        try:
            score_away = int(float(away_val)) if pd.notna(away_val) else 0
        except:
            score_away = 0
        score_diff = score_home - score_away

        # Period and time
        period_id = float(msg_row.get('period_id', 0)) if pd.notna(msg_row.get('period_id')) else 0.0
        seconds = float(msg_row.get('seconds', 0)) if pd.notna(msg_row.get('seconds')) else 0.0
        minutes = seconds / 60.0
        # Map period to total minutes played
        if period_id == 10:  # 1st half
            total_minutes = minutes
        elif period_id == 20:  # 2nd half
            total_minutes = 45 + minutes
        else:
            total_minutes = minutes
        minutes_remaining = max(0, 90 - total_minutes)

        # Confidence
        confidence_grade = float(msg_row.get('confidence_grade', 5.0)) if pd.notna(msg_row.get('confidence_grade')) else 5.0

        # Confidence trend (last 3 events)
        prev_rows = messages_df.iloc[max(0, i-3):i]
        prev_conf = pd.to_numeric(prev_rows['confidence_grade'], errors='coerce').dropna()
        conf_trend = confidence_grade - prev_conf.mean() if len(prev_conf) > 0 else 0.0

        # xT proxy
        incident_name = str(msg_row.get('incident_name', ''))
        xt_weight = get_threat_weight(incident_name)

        # Price at event
        mid_price = float(nearest_price.get('mid_price', 0.5))
        best_bid = float(nearest_price.get('best_bid', 0))
        best_ask = float(nearest_price.get('best_ask', 1))
        spread = best_ask - best_bid

        rows.append({
            'fixture_id': fixture_id,
            'timestamp': ts,
            'incident_name': incident_name,
            'event_density_30s': event_density_30s,
            'event_density_60s': event_density_60s,
            'risk_event_count_30s': int(risk_event_count_30s),
            'risk_event_count_60s': int(risk_event_count_60s),
            'score_diff': score_diff,
            'minutes_remaining': round(minutes_remaining, 2),
            'period_id': period_id,
            'confidence_grade': confidence_grade,
            'conf_trend': float(conf_trend) if pd.notna(conf_trend) else 0.0,
            'xt_weight': xt_weight,
            'mid_price': mid_price,
            'spread': spread,
            'high_volatility': label
        })

    return pd.DataFrame(rows)

# Process all matched fixtures with both messages and prices
all_feature_rows = []
processed = 0
skipped_no_prices = 0
skipped_no_messages = 0

for _, row in matched_df.iterrows():
    fixture_id = str(row['fixture_id'])
    event_date = str(row['event_date'])

    msg_path = MESSAGES_BASE / event_date / fixture_id / 'messages.parquet'
    price_path = PRICES_DIR / f"{fixture_id}.parquet"

    if not msg_path.exists():
        skipped_no_messages += 1
        continue
    if not price_path.exists():
        skipped_no_prices += 1
        continue

    try:
        messages_df = pd.read_parquet(msg_path)
        price_df = pd.read_parquet(price_path)

        if len(messages_df) == 0 or len(price_df) == 0:
            print(f"  Skipping {fixture_id}: empty data")
            continue

        print(f"Processing {fixture_id}: {row['home']} vs {row['away']} | msgs={len(messages_df)} prices={len(price_df)}")

        features = build_features_for_fixture(messages_df, price_df, fixture_id)
        if len(features) > 0:
            all_feature_rows.append(features)
            vol_rate = features['high_volatility'].mean()
            print(f"  -> {len(features)} rows, volatility rate: {vol_rate:.3f}")
        else:
            print(f"  -> No aligned rows")
        processed += 1

    except Exception as e:
        import traceback
        print(f"  Error for {fixture_id}: {e}")
        traceback.print_exc()

print(f"\n=== Summary ===")
print(f"Processed: {processed}")
print(f"No messages: {skipped_no_messages}")
print(f"No prices: {skipped_no_prices}")

if all_feature_rows:
    final_df = pd.concat(all_feature_rows, ignore_index=True)
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)

    out_path = OUTPUT_DIR / 'aligned_dataset.parquet'
    final_df.to_parquet(out_path, index=False)

    print(f"\nDataset shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")
    print(f"Volatility rate: {final_df['high_volatility'].mean():.3f}")
    print(f"Positive labels: {int(final_df['high_volatility'].sum())}")
    print(f"Saved to {out_path}")
else:
    print("No features built! Check data alignment.")

print("\n=== Task 4 Complete ===")
