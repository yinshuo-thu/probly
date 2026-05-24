"""
Tasks 1+3: Download PMXT price data for matched Polymarket markets.

For each top matched fixture, downloads price data from PMXT for the
relevant hours and saves as parquet files.
"""

import sys
import os
import json
import pandas as pd
import subprocess
from datetime import datetime, timezone, timedelta

MATCHES_PATH = "data/polymarket/fixture_market_matches.parquet"
PRICES_DIR = "data/polymarket/prices"
MIN_SCORE = 0.75  # only process high-confidence matches

os.makedirs(PRICES_DIR, exist_ok=True)


def get_match_hours(start_date_utc: str) -> list[str]:
    """Return list of PMXT hour strings to query for a match (kickoff ± 2h)."""
    dt = datetime.fromisoformat(start_date_utc.replace("Z", "+00:00"))
    hours = []
    for delta_h in range(-1, 3):  # 1 hour before, up to 2 hours after kickoff
        h = dt + timedelta(hours=delta_h)
        hours.append(h.strftime("%Y-%m-%dT%H"))
    return hours


def download_condition_prices(condition_id: str, hours: list[str], out_path: str) -> int:
    """Download price data for a condition_id across multiple PMXT hour files."""
    # Write a temp python3.13 script to do the DuckDB queries
    script = f"""
import duckdb
import pandas as pd
import sys

con = duckdb.connect()
con.execute('SET http_timeout=60000')
con.execute('SET enable_progress_bar=false')

condition_id = '{condition_id}'
hours = {hours!r}
all_dfs = []

for hour in hours:
    url = f'https://r2v2.pmxt.dev/polymarket_orderbook_{{hour}}.parquet'
    try:
        df = con.execute(f'''
            SELECT
                timestamp,
                event_type,
                price,
                size,
                side,
                best_bid,
                best_ask,
                asset_id
            FROM read_parquet('{{url}}')
            WHERE decode(market)::varchar = '{{condition_id}}'
            ORDER BY timestamp
        ''').df()
        if len(df) > 0:
            print(f"Hour {{hour}}: {{len(df)}} rows", flush=True)
            all_dfs.append(df)
        else:
            print(f"Hour {{hour}}: no data", flush=True)
    except Exception as e:
        print(f"Hour {{hour}}: error - {{str(e)[:80]}}", flush=True)

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp')
    combined.to_parquet('{out_path}', index=False)
    print(f"Saved {{len(combined)}} rows to {out_path}", flush=True)
else:
    print("No data found", flush=True)
"""
    tmp_script = f"/tmp/pmxt_query_{condition_id[:16]}.py"
    with open(tmp_script, "w") as f:
        f.write(script)

    result = subprocess.run(
        ["python3.13", tmp_script],
        capture_output=True, text=True, timeout=300
    )
    os.unlink(tmp_script)

    output = result.stdout + result.stderr
    for line in output.strip().split("\n"):
        if line.strip():
            print(f"    {line}")

    if os.path.exists(out_path):
        df = pd.read_parquet(out_path)
        return len(df)
    return 0


def main():
    print("Loading matches...")
    matches = pd.read_parquet(MATCHES_PATH)
    top_matches = matches[matches["match_score"] >= MIN_SCORE].copy()

    print(f"Processing {len(top_matches)} high-confidence markets across "
          f"{top_matches['fixture_id'].nunique()} fixtures")

    # Group by fixture to download all condition_ids at once per fixture
    # (they share the same time window)
    fixture_groups = top_matches.groupby("fixture_id")

    total_downloaded = 0
    fixture_count = 0

    for fixture_id, group in fixture_groups:
        start_utc = group["start_date_utc"].iloc[0]
        home = group["home"].iloc[0]
        away = group["away"].iloc[0]
        hours = get_match_hours(start_utc)

        print(f"\n[{fixture_count+1}/{len(fixture_groups)}] {home} vs {away} ({start_utc})")
        print(f"  Hours to scan: {hours}")

        for _, row in group.iterrows():
            cid = row["condition_id"]
            safe_cid = cid.replace("0x", "")[:32]
            out_path = os.path.join(PRICES_DIR, f"{safe_cid}.parquet")

            if os.path.exists(out_path):
                existing = pd.read_parquet(out_path)
                print(f"  Already have {cid[:20]}... ({len(existing)} rows)")
                total_downloaded += len(existing)
                continue

            print(f"  Downloading {cid[:20]}... ({row['question'][:60]})")
            n = download_condition_prices(cid, hours, out_path)
            total_downloaded += n

        fixture_count += 1

    print(f"\nDone! Total price rows downloaded: {total_downloaded}")
    print(f"Files in {PRICES_DIR}: {len(os.listdir(PRICES_DIR))}")


if __name__ == "__main__":
    main()
