#!/usr/bin/env python3.13
"""Task 3: Download PMXT price data for matched Polymarket markets."""

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import time

print("=== Task 3: Downloading PMXT price data ===")

# Load matched fixtures
matched_df = pd.read_parquet('/Volumes/T7/probly/data/matched_fixtures.parquet')
print(f"Loaded {len(matched_df)} matched fixtures")

# Limit to 30 fixtures for initial run (as noted in instructions if slow)
# But let's try all first - prioritize real football matches
real_football = matched_df[~matched_df['league_name'].str.contains('E-Football', na=False)]
e_football = matched_df[matched_df['league_name'].str.contains('E-Football', na=False)]
print(f"Real football fixtures: {len(real_football)}")
print(f"E-football fixtures: {len(e_football)}")

# Prioritize real football, then e-football up to 30 total
target_fixtures = pd.concat([real_football, e_football]).head(50)
print(f"Processing {len(target_fixtures)} fixtures")

OUTPUT_DIR = Path('/Volumes/T7/probly/data/polymarket/prices')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PMXT = "https://r2v2.pmxt.dev/polymarket_orderbook_{HOUR}.parquet"

def get_hours_range(start_utc_str, n_before=1, n_after=3):
    """Get list of hour strings to query."""
    # Parse ISO format like "2026-05-09T17:00:00Z"
    dt = datetime.fromisoformat(start_utc_str.replace('Z', '+00:00'))
    hours = []
    for delta_h in range(-n_before, n_after + 1):
        h = dt + timedelta(hours=delta_h)
        hours.append(h.strftime('%Y-%m-%dT%H'))
    return hours

def format_condition_ids(cids):
    """Format condition IDs for SQL IN clause."""
    return ', '.join(f"'{cid}'" for cid in cids)

processed = 0
skipped = 0
errors = 0

for idx, row in target_fixtures.iterrows():
    fixture_id = row['fixture_id']
    out_path = OUTPUT_DIR / f"{fixture_id}.parquet"

    if out_path.exists():
        skipped += 1
        continue

    condition_ids = row['condition_ids']
    if condition_ids is None or (hasattr(condition_ids, '__len__') and len(condition_ids) == 0):
        continue

    start_utc = row['start_date_utc']
    hours = get_hours_range(start_utc, n_before=1, n_after=3)
    cids_sql = format_condition_ids(condition_ids)

    print(f"\n[{idx}] {row['home']} vs {row['away']} ({row['league_name']})")
    print(f"  Start: {start_utc}, Hours: {hours[0]}..{hours[-1]}")
    print(f"  Markets: {len(condition_ids)}")

    all_dfs = []
    for hour in hours:
        url = BASE_PMXT.replace('{HOUR}', hour)
        query = f"""
        SELECT
            timestamp_received,
            timestamp,
            decode(market) as condition_id,
            event_type,
            price,
            side,
            best_bid,
            best_ask,
            size
        FROM read_parquet('{url}')
        WHERE event_type = 'price_change'
          AND decode(market) IN ({cids_sql})
        """
        try:
            df = duckdb.sql(query).df()
            if len(df) > 0:
                all_dfs.append(df)
                print(f"  Hour {hour}: {len(df)} rows")
        except Exception as e:
            print(f"  Hour {hour}: Error - {str(e)[:80]}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('timestamp')
        combined.to_parquet(out_path, index=False)
        print(f"  Saved {len(combined)} total price events")
        processed += 1
    else:
        print(f"  No data found")
        errors += 1

print(f"\n=== Task 3 Summary ===")
print(f"Processed: {processed}")
print(f"Skipped (already exist): {skipped}")
print(f"No data: {errors}")

# List what we have
existing = list(OUTPUT_DIR.glob('*.parquet'))
print(f"Total fixture price files: {len(existing)}")
print("\n=== Task 3 Complete ===")
