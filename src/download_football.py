"""
Download all football fixture metadata and messages for major leagues.

Phase 1: Download fixtures.parquet for all dates → build index
Phase 2: Filter by league → find major league fixtures
Phase 3: Download messages.parquet for filtered fixtures

Usage:
  python src/download_football.py --index           # Build fixture index only
  python src/download_football.py --download        # Download all major league messages
  python src/download_football.py --all             # Both
"""

import os, sys, json, time, argparse, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

TOKEN = open(Path.home() / ".cache/huggingface/token").read().strip()
REPO = "probly/lsports-hyper-dataset"
BASE_URL = "https://huggingface.co/datasets/probly/lsports-hyper-dataset/resolve/main"
API_BASE = "https://huggingface.co/api/datasets/probly/lsports-hyper-dataset"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

DATA_DIR = Path(__file__).parent.parent / "data" / "hyper" / "football"
INDEX_FILE = Path(__file__).parent.parent / "data" / "fixture_index.parquet"

# Major leagues likely covered by Polymarket
MAJOR_LEAGUES = {
    "UEFA Champions League", "UEFA Europa League", "UEFA Conference League",
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Copa Libertadores", "Copa Sudamericana",
    "MLS", "Liga MX",
    "Champions League", "Europa League",
}

FOOTBALL_DATES = [
    "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10",
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
    "2026-05-16", "2026-05-17", "2026-05-18", "2026-05-19", "2026-05-20",
    "2026-05-21", "2026-05-22",
]


def list_fixtures_for_date(date: str) -> list:
    """Get all fixture IDs for a given date."""
    url = f"{API_BASE}/tree/main/sport_id%3D6046__sport%3DFootball/event_date%3D{date}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if not r.ok:
        print(f"  Error listing {date}: {r.status_code}")
        return []
    entries = r.json()
    return [
        int(e["path"].split("fixture_id=")[-1])
        for e in entries
        if e.get("type") == "directory" and "fixture_id=" in e["path"]
    ]


def download_fixture_meta(date: str, fixture_id: int):
    """Download and parse fixtures.parquet for one fixture."""
    path = f"sport_id=6046__sport=Football/event_date={date}/fixture_id={fixture_id}/fixtures.parquet"
    url = f"{BASE_URL}/{path}"
    local = DATA_DIR / date / str(fixture_id) / "fixtures.parquet"
    local.parent.mkdir(parents=True, exist_ok=True)

    if not local.exists():
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        if not r.ok:
            return None
        with open(local, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    try:
        df = pd.read_parquet(local)
        if len(df) == 0:
            return None
        row = df.iloc[0]
        return {
            "fixture_id": fixture_id,
            "event_date": date,
            "home": row.get("home", ""),
            "away": row.get("away", ""),
            "league_id": row.get("league_id"),
            "league_name": row.get("league_name", ""),
            "league_type": row.get("league_type"),
            "location_name": row.get("location_name", ""),
            "start_date_utc": str(row.get("start_date_utc", "")),
            "status": row.get("status"),
        }
    except Exception as e:
        return None


def build_index(max_workers: int = 8) -> pd.DataFrame:
    """Download all fixture metadata and build a searchable index."""
    print(f"Building fixture index for {len(FOOTBALL_DATES)} dates...")
    all_meta = []

    for date in FOOTBALL_DATES:
        print(f"  {date}: listing fixtures...")
        fixture_ids = list_fixtures_for_date(date)
        print(f"    Found {len(fixture_ids)} fixtures")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(download_fixture_meta, date, fid): fid for fid in fixture_ids}
            for fut in as_completed(futures):
                meta = fut.result()
                if meta:
                    all_meta.append(meta)

        print(f"    Downloaded {len([m for m in all_meta if m['event_date']==date])} fixture metas")

    index = pd.DataFrame(all_meta)
    index.to_parquet(INDEX_FILE, index=False)
    print(f"\nFixture index saved: {len(index)} fixtures → {INDEX_FILE}")
    return index


def download_messages(fixture_id: int, date: str, force: bool = False):
    """Download messages.parquet for one fixture."""
    path = f"sport_id=6046__sport=Football/event_date={date}/fixture_id={fixture_id}/messages.parquet"
    url = f"{BASE_URL}/{path}"
    local = DATA_DIR / date / str(fixture_id) / "messages.parquet"
    local.parent.mkdir(parents=True, exist_ok=True)

    if local.exists() and not force:
        return local

    r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
    if not r.ok:
        return None
    with open(local, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    return local


def download_major_leagues(index: pd.DataFrame, max_workers: int = 6) -> None:
    """Download messages for all major league fixtures."""
    # Filter by league name
    major = index[
        index["league_name"].apply(
            lambda n: any(ml.lower() in str(n).lower() for ml in MAJOR_LEAGUES)
        )
    ].copy()

    print(f"\nMajor league fixtures: {len(major)}")
    print(major.groupby("league_name").size().sort_values(ascending=False).head(20).to_string())
    print()

    rows = list(major.itertuples())
    done, failed = 0, 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(download_messages, row.fixture_id, row.event_date): row
            for row in rows
        }
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result:
                done += 1
            else:
                failed += 1
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(rows)} done={done} failed={failed}")

    print(f"Done: {done} downloaded, {failed} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Build fixture index")
    parser.add_argument("--download", action="store_true", help="Download major league messages")
    parser.add_argument("--all", action="store_true", help="Build index + download")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.index:
        index = build_index(max_workers=args.workers)
    elif INDEX_FILE.exists():
        index = pd.read_parquet(INDEX_FILE)
        print(f"Loaded existing index: {len(index)} fixtures")
    else:
        print("No index found. Run with --index first.")
        return

    if args.all or args.download:
        download_major_leagues(index, max_workers=args.workers)

    if not any([args.index, args.download, args.all]):
        print("Specify --index, --download, or --all")
        print("Example: python src/download_football.py --all --workers 8")


if __name__ == "__main__":
    main()
