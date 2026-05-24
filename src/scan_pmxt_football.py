#!/usr/bin/env python3.13
"""
Scan PMXT v2 archive for football/soccer markets from May 6-22, 2026.
Queries evening hours (17-21 UTC) across the date range, decodes market BLOBs,
looks up each condition_id via CLOB API, and filters for football markets.
"""

import json
import time
import duckdb
import urllib.request
import urllib.error
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_PATH = "/Volumes/T7/probly/data/polymarket/football_markets.json"
PMXT_BASE   = "https://r2v2.pmxt.dev/polymarket_orderbook_{dt}T{hh}.parquet"
CLOB_BASE   = "https://clob.polymarket.com/markets/{cid}"

START_DATE  = date(2026, 5, 6)
END_DATE    = date(2026, 5, 22)
HOURS       = [17, 18, 19, 20, 21]          # UTC evening hours
TOP_N       = 200                            # most-active markets per file
MAX_WORKERS = 12                             # concurrent CLOB threads

# Football keywords (English + major European leagues)
FOOTBALL_KEYWORDS = [
    # generic
    "vs", " v ", "draw", " win", "goal", "score", "soccer", "football",
    "match", "league", "cup", "ucl", "champions league", "europa league",
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    # Spanish clubs
    "real madrid", "barcelona", "atletico", "sevilla", "valencia",
    "athletic bilbao", "real sociedad", "villarreal", "betis", "osasuna",
    # English clubs
    "manchester", "arsenal", "chelsea", "liverpool", "tottenham", "newcastle",
    "aston villa", "brighton", "west ham", "everton", "brentford",
    # Italian clubs
    "juventus", "milan", "inter", "napoli", "roma", "lazio", "fiorentina",
    "atalanta", "bologna", "torino",
    # German clubs
    "bayern", "dortmund", "leverkusen", "rb leipzig", "frankfurt",
    "wolfsburg", "borussia",
    # French clubs
    "psg", "paris saint", "marseille", "lyon", "monaco", "lille", "nice",
    "rennes", "lens",
    # Misc
    "premier", "championship", "fa cup", "copa del rey", "dfb pokal",
    "coupe de france", "world cup", "euro 2026", "nations league",
    "cl final", "el final",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
seen_cids:  set[str] = set()
results:    list[dict] = []
lock:       Lock = Lock()


def is_football(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in FOOTBALL_KEYWORDS)


def fetch_clob(cid: str) -> dict | None:
    """Fetch market details from CLOB API. Returns dict or None on failure."""
    url = CLOB_BASE.format(cid=cid)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "pmxt-football-scanner/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError,
            TimeoutError, ConnectionResetError):
        pass
    return None


def query_pmxt_file(url: str, label: str) -> list[str]:
    """
    Query a single PMXT parquet file via DuckDB HTTPS.
    Returns list of condition_id strings (top N most active markets).
    """
    sql = f"""
    SELECT
        decode(market) AS condition_id,
        COUNT(*) AS cnt
    FROM read_parquet('{url}')
    GROUP BY market
    ORDER BY cnt DESC
    LIMIT {TOP_N}
    """
    try:
        con = duckdb.connect()
        con.execute("SET http_timeout=60000")
        rows = con.execute(sql).fetchall()
        con.close()
        cids = [str(row[0]).strip() for row in rows if row[0]]
        print(f"  [{label}] {len(cids)} markets fetched from PMXT")
        return cids
    except Exception as e:
        print(f"  [{label}] PMXT query failed: {e}")
        return []


def process_cid(cid: str, label: str) -> dict | None:
    """Look up a condition_id and return a result dict if it's a football market."""
    data = fetch_clob(cid)
    if not data:
        return None
    question = data.get("question") or data.get("market_slug") or ""
    if not is_football(question):
        return None
    return {
        "condition_id": cid,
        "question":     question,
        "date":         label,
        "event_type":   "football",
        "clob_data":    {
            "market_slug":   data.get("market_slug"),
            "description":   data.get("description", "")[:200],
            "end_date_iso":  data.get("end_date_iso"),
            "active":        data.get("active"),
            "closed":        data.get("closed"),
            "volume":        data.get("volume"),
        },
    }


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------
def main():
    global seen_cids, results

    current = START_DATE
    all_dates = []
    while current <= END_DATE:
        all_dates.append(current)
        current += timedelta(days=1)

    total_files = len(all_dates) * len(HOURS)
    print(f"Scanning {len(all_dates)} dates × {len(HOURS)} hours = {total_files} files")
    print(f"Date range: {START_DATE} – {END_DATE}")
    print(f"Hours (UTC): {HOURS}")
    print("-" * 60)

    new_cids_to_lookup: list[tuple[str, str]] = []  # (cid, label)

    # Phase 1: collect condition_ids from PMXT
    for d in all_dates:
        for h in HOURS:
            dt_str  = d.strftime("%Y-%m-%d")
            hh_str  = f"{h:02d}"
            url     = PMXT_BASE.format(dt=dt_str, hh=hh_str)
            label   = f"{dt_str}T{hh_str}"

            cids = query_pmxt_file(url, label)

            added = 0
            for cid in cids:
                if cid and cid not in seen_cids:
                    seen_cids.add(cid)
                    new_cids_to_lookup.append((cid, label))
                    added += 1
            print(f"  [{label}] {added} new condition_ids queued (total unique: {len(seen_cids)})")

    print("\n" + "=" * 60)
    print(f"Phase 1 complete. Total unique condition_ids to look up: {len(new_cids_to_lookup)}")
    print("=" * 60)

    # Phase 2: concurrent CLOB lookups
    print(f"\nPhase 2: CLOB lookup with {MAX_WORKERS} workers...")
    football_count = 0
    done = 0
    total = len(new_cids_to_lookup)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(process_cid, cid, label): (cid, label)
            for cid, label in new_cids_to_lookup
        }
        for future in as_completed(future_map):
            done += 1
            result = future.result()
            if result:
                with lock:
                    # Avoid duplicate condition_ids in results
                    existing_cids = {r["condition_id"] for r in results}
                    if result["condition_id"] not in existing_cids:
                        results.append(result)
                        football_count += 1
                        print(f"  FOOTBALL [{done}/{total}] {result['condition_id'][:20]}... "
                              f"=> {result['question'][:60]}")
            if done % 100 == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (total - done) / rate if rate > 0 else 0
                print(f"  Progress: {done}/{total} ({100*done/total:.1f}%) "
                      f"| {rate:.1f} req/s | ETA {eta:.0f}s "
                      f"| Football found: {football_count}")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    results.sort(key=lambda x: (x["date"], x["question"]))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"SCAN COMPLETE in {elapsed_total:.0f}s")
    print(f"Total unique condition_ids scanned : {len(seen_cids)}")
    print(f"Football markets found             : {football_count}")
    print(f"Results saved to                   : {OUTPUT_PATH}")
    print("=" * 60)

    # Print summary table
    if results:
        print("\nFootball markets found:")
        for r in results:
            print(f"  {r['date']:18s}  {r['condition_id'][:20]}...  {r['question'][:60]}")


if __name__ == "__main__":
    main()
