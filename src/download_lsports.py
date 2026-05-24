"""
Task 3: Download LSports messages.parquet for matched fixtures from HuggingFace.
"""

import os
import sys
import requests
import pandas as pd

MATCHES_PATH = "data/polymarket/fixture_market_matches.parquet"
HYPER_DIR = "data/hyper/football"
HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
HF_BASE = "https://huggingface.co/datasets/probly/lsports-hyper-dataset/resolve/main"
MIN_SCORE = 0.75

def get_token():
    with open(HF_TOKEN_PATH) as f:
        return f.read().strip()

def download_messages(fixture_id, event_date, token):
    """Download messages.parquet for a fixture. Returns local path or None."""
    out_dir = os.path.join(HYPER_DIR, event_date, str(fixture_id))
    out_path = os.path.join(out_dir, "messages.parquet")

    if os.path.exists(out_path):
        sz = os.path.getsize(out_path)
        if sz > 1000:
            print(f"  Already have {fixture_id} ({sz} bytes)")
            return out_path
        else:
            print(f"  Removing tiny file for {fixture_id} ({sz} bytes)")
            os.remove(out_path)

    os.makedirs(out_dir, exist_ok=True)

    url = (
        f"{HF_BASE}/sport_id=6046__sport=Football"
        f"/event_date={event_date}"
        f"/fixture_id={fixture_id}"
        f"/messages.parquet"
    )

    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=120, stream=True)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            sz = os.path.getsize(out_path)
            print(f"  Downloaded {fixture_id}: {sz:,} bytes → {out_path}")
            return out_path
        elif resp.status_code == 404:
            print(f"  NOT FOUND: {fixture_id} (date={event_date})")
            return None
        else:
            print(f"  ERROR {resp.status_code} for {fixture_id}: {resp.text[:100]}")
            return None
    except Exception as e:
        print(f"  EXCEPTION for {fixture_id}: {e}")
        return None


def main():
    token = get_token()
    print("Loaded HF token")

    matches = pd.read_parquet(MATCHES_PATH)
    top = matches[matches["match_score"] >= MIN_SCORE].drop_duplicates("fixture_id")
    print(f"Downloading LSports messages for {len(top)} fixtures...")

    downloaded = 0
    failed = 0

    for i, (_, row) in enumerate(top.iterrows()):
        fixture_id = int(row["fixture_id"])
        event_date = str(row["event_date"])
        home = row["home"]
        away = row["away"]
        print(f"\n[{i+1}/{len(top)}] {home} vs {away} (fixture_id={fixture_id}, date={event_date})")

        path = download_messages(fixture_id, event_date, token)
        if path:
            # Verify it's a valid parquet
            try:
                df = pd.read_parquet(path)
                print(f"  OK: {len(df)} rows, columns: {list(df.columns)[:5]}")
                downloaded += 1
            except Exception as e:
                print(f"  Invalid parquet: {e}")
                os.remove(path)
                failed += 1
        else:
            failed += 1

    print(f"\nDone! Downloaded: {downloaded}, Failed/missing: {failed}")


if __name__ == "__main__":
    main()
