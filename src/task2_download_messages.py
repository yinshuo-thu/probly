#!/usr/bin/env python3
"""Task 2: Download messages.parquet for matched fixtures from HuggingFace."""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=== Task 2: Downloading messages.parquet for matched fixtures ===")

# Load matched fixtures
matched_df = pd.read_parquet('/Volumes/T7/probly/data/matched_fixtures.parquet')
print(f"Loaded {len(matched_df)} matched fixtures")

# HuggingFace token
token_path = os.path.expanduser('~/.cache/huggingface/token')
with open(token_path) as f:
    TOKEN = f.read().strip()
print(f"Token loaded: {TOKEN[:8]}...")

BASE_URL = "https://huggingface.co/datasets/probly/lsports-hyper-dataset/resolve/main"
OUTPUT_BASE = Path('/Volumes/T7/probly/data/hyper/football')

def download_messages(row):
    fixture_id = row['fixture_id']
    event_date = str(row['event_date'])

    out_dir = OUTPUT_BASE / event_date / str(fixture_id)
    out_path = out_dir / 'messages.parquet'

    if out_path.exists():
        return fixture_id, 'skipped', None

    # Construct URL
    url = f"{BASE_URL}/sport_id=6046__sport=Football/event_date={event_date}/fixture_id={fixture_id}/messages.parquet"

    headers = {'Authorization': f'Bearer {TOKEN}'}

    try:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 200:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            return fixture_id, 'downloaded', len(resp.content)
        elif resp.status_code == 404:
            return fixture_id, '404', None
        else:
            return fixture_id, f'error_{resp.status_code}', None
    except Exception as e:
        return fixture_id, f'exception_{str(e)[:50]}', None

rows = [row for _, row in matched_df.iterrows()]
print(f"\nDownloading {len(rows)} fixtures with 8 workers...")

results = {'downloaded': 0, 'skipped': 0, 'not_found': 0, 'error': 0}
downloaded_fixtures = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(download_messages, row): row['fixture_id'] for row in rows}
    for i, future in enumerate(as_completed(futures)):
        fixture_id, status, size = future.result()
        if status == 'downloaded':
            results['downloaded'] += 1
            downloaded_fixtures.append(fixture_id)
            if results['downloaded'] % 10 == 0:
                print(f"  [{i+1}/{len(rows)}] Downloaded {results['downloaded']} so far...")
        elif status == 'skipped':
            results['skipped'] += 1
            downloaded_fixtures.append(fixture_id)
        elif status == '404':
            results['not_found'] += 1
        else:
            results['error'] += 1
            if i < 20:  # Show first errors
                print(f"  Error for {fixture_id}: {status}")

print(f"\nResults:")
print(f"  Downloaded: {results['downloaded']}")
print(f"  Skipped (already exist): {results['skipped']}")
print(f"  Not found (404): {results['not_found']}")
print(f"  Errors: {results['error']}")

# Verify what we have
existing = []
for _, row in matched_df.iterrows():
    fixture_id = row['fixture_id']
    event_date = str(row['event_date'])
    out_path = OUTPUT_BASE / event_date / str(fixture_id) / 'messages.parquet'
    if out_path.exists():
        existing.append(fixture_id)

print(f"\nTotal fixtures with messages.parquet: {len(existing)}")
print("\n=== Task 2 Complete ===")
