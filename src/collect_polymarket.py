"""
Collect Polymarket prediction market data for football matches.

Uses:
  - Gamma Markets API: find football markets, get metadata
  - Polymarket CLOB API: get price history / trades for each market

Usage:
  python src/collect_polymarket.py --search          # Find football markets
  python src/collect_polymarket.py --prices          # Download price history
  python src/collect_polymarket.py --all             # Both

Output:
  data/polymarket/markets.parquet   - market metadata
  data/polymarket/prices.parquet    - price time series
  data/polymarket/trades.parquet    - trade history
"""

import json, time, argparse, requests
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"
DATA_DIR  = Path(__file__).parent.parent / "data" / "polymarket"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Date range matching our LSports data
DATE_FROM = "2026-05-06"
DATE_TO   = "2026-05-22"

FOOTBALL_KEYWORDS = [
    "football", "soccer", "match", "goal", "win", "draw",
    "premier league", "la liga", "champions league", "serie a",
    "bundesliga", "ligue 1", "europa league", "copa",
    "real madrid", "barcelona", "manchester", "liverpool",
    "arsenal", "chelsea", "psg", "juventus", "milan", "inter",
    "bayern", "dortmund", "atletico", "napoli",
]


def get_markets(keyword: str, limit: int = 100, offset: int = 0) -> list:
    """Search Gamma API for markets matching keyword."""
    params = {
        "q": keyword,
        "limit": limit,
        "offset": offset,
        "active": "true",
    }
    try:
        r = requests.get(f"{GAMMA_API}/markets", params=params, timeout=15)
        if r.ok:
            return r.json()
    except Exception as e:
        print(f"  Error fetching markets for '{keyword}': {e}")
    return []


def get_all_football_markets() -> pd.DataFrame:
    """Search for all football-related markets on Polymarket."""
    print("Searching Polymarket for football markets...")
    all_markets = {}

    for kw in FOOTBALL_KEYWORDS:
        markets = get_markets(kw)
        for m in markets:
            mid = m.get("id") or m.get("conditionId") or m.get("condition_id")
            if mid and mid not in all_markets:
                all_markets[mid] = m
        time.sleep(0.1)

    # Also get recent resolved markets
    try:
        r = requests.get(f"{GAMMA_API}/markets", params={
            "limit": 500,
            "tag_id": "7",  # Sports tag
        }, timeout=15)
        if r.ok:
            for m in r.json():
                mid = m.get("id") or m.get("conditionId")
                if mid:
                    all_markets[mid] = m
    except Exception as e:
        print(f"  Error fetching sports markets: {e}")

    df = pd.DataFrame(list(all_markets.values()))
    print(f"Found {len(df)} unique football/sports markets")
    return df


def get_price_history(condition_id: str, resolution: int = 60) -> pd.DataFrame:
    """
    Get price time series for a market from CLOB.
    resolution: candle size in seconds (60 = 1-min candles)
    """
    params = {
        "market": condition_id,
        "resolution": resolution,
        "startTs": int(pd.Timestamp(DATE_FROM).timestamp()),
        "endTs": int(pd.Timestamp(DATE_TO).timestamp()),
    }
    try:
        r = requests.get(f"{CLOB_API}/prices-history", params=params, timeout=15)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and "history" in data:
                history = data["history"]
                if history:
                    df = pd.DataFrame(history)
                    df["condition_id"] = condition_id
                    return df
    except Exception as e:
        pass
    return pd.DataFrame()


def get_trades(condition_id: str) -> pd.DataFrame:
    """Get recent trades for a market."""
    try:
        r = requests.get(f"{CLOB_API}/trades", params={
            "market": condition_id,
            "limit": 500,
        }, timeout=15)
        if r.ok:
            data = r.json()
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                df["condition_id"] = condition_id
                return df
    except Exception as e:
        pass
    return pd.DataFrame()


def get_market_detail(condition_id: str) -> dict:
    """Get detailed market info including tokens."""
    try:
        r = requests.get(f"{CLOB_API}/markets/{condition_id}", timeout=15)
        if r.ok:
            return r.json()
    except Exception as e:
        pass
    return {}


def collect_all_prices(markets: pd.DataFrame) -> pd.DataFrame:
    """Download price history for all markets."""
    print(f"\nDownloading price history for {len(markets)} markets...")
    all_prices = []

    # Identify condition_id column
    id_col = None
    for col in ["conditionId", "condition_id", "id"]:
        if col in markets.columns:
            id_col = col
            break

    if id_col is None:
        print("No condition_id column found. Columns:", list(markets.columns))
        return pd.DataFrame()

    for i, (_, row) in enumerate(markets.iterrows()):
        cid = row.get(id_col)
        if not cid:
            continue
        df = get_price_history(str(cid))
        if not df.empty:
            all_prices.append(df)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(markets)} markets processed, {len(all_prices)} with data")
        time.sleep(0.05)

    if not all_prices:
        print("No price history found")
        return pd.DataFrame()

    result = pd.concat(all_prices, ignore_index=True)
    print(f"Price history: {len(result)} rows for {result['condition_id'].nunique()} markets")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="Find football markets")
    parser.add_argument("--prices", action="store_true", help="Download price history")
    parser.add_argument("--all", action="store_true", help="Search + prices")
    args = parser.parse_args()

    markets_file = DATA_DIR / "markets.parquet"
    prices_file  = DATA_DIR / "prices.parquet"

    if args.all or args.search:
        markets = get_all_football_markets()
        if not markets.empty:
            markets.to_parquet(markets_file, index=False)
            print(f"Markets saved: {markets_file}")
            print("\nMarket sample:")
            show_cols = [c for c in ["question","title","description","endDate","active"] if c in markets.columns]
            print(markets[show_cols].head(10).to_string() if show_cols else markets.head(5).to_string())
    elif markets_file.exists():
        markets = pd.read_parquet(markets_file)
        print(f"Loaded {len(markets)} markets from {markets_file}")
    else:
        print("No markets file. Run with --search first.")
        return

    if args.all or args.prices:
        prices = collect_all_prices(markets)
        if not prices.empty:
            prices.to_parquet(prices_file, index=False)
            print(f"Prices saved: {prices_file}")

    if not any([args.search, args.prices, args.all]):
        print("Specify --search, --prices, or --all")
        print("Example: python src/collect_polymarket.py --all")


if __name__ == "__main__":
    main()
