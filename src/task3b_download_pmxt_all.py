#!/usr/bin/env python3.13
"""Task 3b: Download PMXT price data for ALL matched Polymarket markets (fixture_id named files)."""

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

print("=== Task 3b: Downloading PMXT price data for all matched fixtures ===")

# Load matched fixtures
matched_df = pd.read_parquet('/Volumes/T7/probly/data/matched_fixtures.parquet')
print(f"Loaded {len(matched_df)} matched fixtures")

OUTPUT_DIR = Path('/Volumes/T7/probly/data/polymarket/prices')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PMXT = "https://r2v2.pmxt.dev/polymarket_orderbook_{HOUR}.parquet"

def get_hours_range(start_utc_str, n_before=1, n_after=3):
    """Get list of hour strings to query."""
    dt = datetime.fromisoformat(start_utc_str.replace('Z', '+00:00'))
    hours = []
    for delta_h in range(-n_before, n_after + 1):
        h = dt + timedelta(hours=delta_h)
        hours.append(h.strftime('%Y-%m-%dT%H'))
    return hours

def format_condition_ids(cids):
    return ', '.join(f"'{cid}'" for cid in cids)

# Prioritize real football fixtures
real_football = matched_df[~matched_df['league_name'].str.contains('E-Football', na=False)]
e_football = matched_df[matched_df['league_name'].str.contains('E-Football', na=False)]
print(f"Real football: {len(real_football)}, E-football: {len(e_football)}")

# Process real football first, then e-football, limit to 40 total
target_fixtures = pd.concat([real_football, e_football]).head(40)
print(f"Processing {len(target_fixtures)} fixtures")

processed = 0
skipped = 0
no_data = 0

for idx, row in target_fixtures.iterrows():
    fixture_id = str(row['fixture_id'])
    out_path = OUTPUT_DIR / f"{fixture_id}.parquet"

    if out_path.exists():
        skipped += 1
        print(f"[SKIP] {fixture_id}: {row['home']} vs {row['away']}")
        continue

    condition_ids = list(row['condition_ids'])
    if not condition_ids:
        no_data += 1
        continue

    start_utc = str(row['start_date_utc'])
    hours = get_hours_range(start_utc, n_before=1, n_after=3)
    cids_sql = format_condition_ids(condition_ids)

    print(f"\n[{processed+skipped+1}/{len(target_fixtures)}] {row['home']} vs {row['away']} ({row['league_name']})")
    print(f"  fixture_id={fixture_id}, start={start_utc}")
    print(f"  {len(condition_ids)} markets, hours: {hours[0]}..{hours[-1]}")

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
            else:
                print(f"  Hour {hour}: 0 rows")
        except Exception as e:
            print(f"  Hour {hour}: Error - {str(e)[:80]}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('timestamp')
        combined.to_parquet(out_path, index=False)
        print(f"  Saved {len(combined)} total rows to {out_path.name}")
        processed += 1
    else:
        print(f"  No data found for this fixture")
        no_data += 1

print(f"\n=== Task 3b Summary ===")
print(f"Newly processed: {processed}")
print(f"Skipped (already exist): {skipped}")
print(f"No data found: {no_data}")

# Count valid fixture price files
valid = [f for f in OUTPUT_DIR.glob('*.parquet')
         if f.stem.isdigit() or (f.stem.replace('_','').isalnum() and len(f.stem) <= 12)]
print(f"\nFixture price files (numeric IDs): {sum(1 for f in OUTPUT_DIR.glob('*.parquet') if f.stem.lstrip('0123456789').strip() == '' or (len(f.stem) < 12))}")

all_files = list(OUTPUT_DIR.glob('*.parquet'))
print(f"Total price files: {len(all_files)}")
print("\n=== Task 3b Complete ===")
