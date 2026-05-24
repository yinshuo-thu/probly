"""
Build real aligned dataset - optimized batch version.

Strategy:
1. Group all condition_ids by PMXT hour (95 unique hours vs 1104 sequential)
2. Parallel DuckDB queries (4 workers) → price cache
3. Assemble feature+label rows per fixture
"""

import sys, os, json, warnings, subprocess, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
sys.path.insert(0, '/Volumes/T7/probly/src')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/Volumes/T7/probly')
MATCHED_FILE = BASE / 'data/matched_fixtures.parquet'
FMM_FILE     = BASE / 'data/polymarket/fixture_market_matches.parquet'
MSGS_DIR     = BASE / 'data/hyper/football'
OUT_FILE     = BASE / 'outputs/real_dataset.parquet'
CACHE_DIR    = BASE / 'data/price_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PMXT_URL = 'https://r2v2.pmxt.dev/polymarket_orderbook_{hour}.parquet'
MAX_WORKERS = 2

XT_WEIGHT = {
    'Score': 5.0, 'PlayerGoals': 5.0,
    'Penalties': 4.0, 'MissedPenalty': 3.0,
    'RedCard': 4.0, 'PlayerRedCard': 4.0,
    'YellowCard': 1.5, 'PlayerYellowCard': 1.5,
    'PlayerShotsOnTarget': 2.0, 'ShotsOnTarget': 2.0,
    'PlayerShotsOffTarget': 1.0, 'ShotsOffTarget': 1.0,
    'Corners': 1.2, 'DangerousAttacks': 0.8,
    'Attacks': 0.4, 'Possession': 0.1,
    'Timer': 0.0, 'Period': 0.0,
}
HIGH_RISK = {'Score','PlayerGoals','Penalties','RedCard','PlayerRedCard',
             'YellowCard','PlayerShotsOnTarget','ShotsOnTarget','MissedPenalty'}


def fetch_hour_prices(hour_str: str, condition_ids: set) -> pd.DataFrame:
    """Fetch ALL price_change events for given condition_ids from one PMXT hour file."""
    cache_file = CACHE_DIR / f"{hour_str.replace(':', '-')}.parquet"

    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    # Filter to valid hex condition_ids
    valid_cids = [c for c in condition_ids if c and c.startswith('0x')]
    if not valid_cids:
        return pd.DataFrame()

    url = PMXT_URL.format(hour=hour_str)
    # decode(market) gives the hex string like '0x...' directly
    cid_list = "', '".join(valid_cids)

    code = f"""
import duckdb, pandas as pd, sys
con = duckdb.connect()
con.execute("SET http_timeout=300000; SET threads=4;")
try:
    df = con.execute('''
        SELECT
            timestamp,
            decode(market) as condition_id_raw,
            event_type,
            price,
            size,
            side,
            best_bid,
            best_ask
        FROM read_parquet('{url}')
        WHERE event_type = 'price_change'
          AND decode(market) IN ('{cid_list}')
        ORDER BY timestamp
    ''').df()
    print(df.to_json(orient='records'))
except Exception as e:
    sys.stderr.write(str(e) + '\\n')
    print('[]')
"""
    try:
        result = subprocess.run(
            ['python3.13', '-c', code],
            capture_output=True, text=True, timeout=600
        )
        raw = result.stdout.strip()
        if not raw or raw == '[]':
            df = pd.DataFrame()
        else:
            records = json.loads(raw)
            if not records:
                df = pd.DataFrame()
            else:
                df = pd.DataFrame(records)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                # Map condition_id_raw back to hex
                df['condition_id'] = df['condition_id_raw'].apply(
                    lambda x: hex(int(str(x).encode('latin1').hex(), 16)) if x else None
                )
        if not df.empty:
            df.to_parquet(cache_file, index=False)
        return df
    except Exception as e:
        print(f"  ERROR fetching {hour_str}: {e}")
        return pd.DataFrame()


def build_price_cache(hour_to_cids: dict) -> dict:
    """Parallel fetch of all PMXT hours."""
    print(f"Fetching {len(hour_to_cids)} PMXT hours (parallel, {MAX_WORKERS} workers)...")
    hour_prices = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(fetch_hour_prices, hour, cids): hour
            for hour, cids in hour_to_cids.items()
        }
        done = 0
        for fut in as_completed(futures):
            hour = futures[fut]
            try:
                df = fut.result()
                hour_prices[hour] = df
                rows = len(df) if not df.empty else 0
                done += 1
                print(f"  [{done}/{len(hour_to_cids)}] {hour}: {rows} price rows")
            except Exception as e:
                hour_prices[hour] = pd.DataFrame()
                done += 1
                print(f"  [{done}/{len(hour_to_cids)}] {hour}: ERROR {e}")

    return hour_prices


def prepare_messages(msgs: pd.DataFrame) -> pd.DataFrame:
    msgs = msgs.copy()
    msgs['timestamp_utc'] = pd.to_datetime(msgs['timestamp_utc'], utc=True)
    msgs['score_home'] = pd.to_numeric(msgs['home_value'], errors='coerce').fillna(0)
    msgs['score_away'] = pd.to_numeric(msgs['away_value'], errors='coerce').fillna(0)
    msgs['score_diff']  = (msgs['score_home'] - msgs['score_away']).astype(int)
    elapsed_sec = pd.to_numeric(msgs['seconds'], errors='coerce').fillna(0)
    msgs['minutes_remaining'] = (90*60 - elapsed_sec).clip(lower=0) / 60
    msgs['period_id'] = pd.to_numeric(msgs['period_id'], errors='coerce').fillna(10.0)
    return msgs.sort_values('timestamp_utc')


def label_and_features(msgs: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    msgs = msgs.sort_values('timestamp_utc').copy()
    msgs['ts'] = pd.to_datetime(msgs['timestamp_utc'], utc=True)

    if prices.empty:
        return pd.DataFrame()

    prices = prices.sort_values('timestamp').copy()
    prices['ts'] = pd.to_datetime(prices['timestamp'], utc=True)
    prices['mid'] = (prices['best_bid'] + prices['best_ask']) / 2

    # Filter prices to match time window (±2h around messages)
    t_min = msgs['ts'].min() - pd.Timedelta(hours=1)
    t_max = msgs['ts'].max() + pd.Timedelta(hours=2)
    prices = prices[(prices['ts'] >= t_min) & (prices['ts'] <= t_max)].copy()

    if prices.empty:
        return pd.DataFrame()

    price_ts = prices['ts'].values
    mids = prices['mid'].values

    rows = []
    for row in msgs.itertuples():
        t0 = row.ts
        t0_ns = np.datetime64(t0)

        idx_before = np.searchsorted(price_ts, t0_ns, side='right') - 1
        if idx_before < 0:
            continue
        current_price = mids[idx_before]
        if current_price <= 0 or current_price >= 1:
            continue

        t_end_ns = np.datetime64(t0 + pd.Timedelta(seconds=120))
        future_mask = (price_ts > t0_ns) & (price_ts <= t_end_ns)
        future_mids = mids[future_mask]

        max_abs_move = float(np.abs(future_mids - current_price).max()) if len(future_mids) > 0 else 0.0
        high_vol = int(max_abs_move > 0.03)

        t30  = t0 - pd.Timedelta(seconds=30)
        t60  = t0 - pd.Timedelta(seconds=60)
        t120 = t0 - pd.Timedelta(seconds=120)

        ev30  = msgs[(msgs['ts'] >= t30)  & (msgs['ts'] < t0)]
        ev60  = msgs[(msgs['ts'] >= t60)  & (msgs['ts'] < t0)]
        ev120 = msgs[(msgs['ts'] >= t120) & (msgs['ts'] < t0)]

        recent_conf = msgs[(msgs['ts'] >= t60) & (msgs['ts'] < t0)]['confidence_grade'].dropna().values[-5:]
        conf_trend = float(np.polyfit(range(len(recent_conf)), recent_conf, 1)[0]) if len(recent_conf) > 1 else 0.0

        incident = str(row.incident_name)
        rows.append({
            'fixture_id':           str(row.fixture_id),
            'timestamp':            str(t0),
            'incident_name':        incident,
            'confidence_grade':     float(row.confidence_grade) if pd.notnull(row.confidence_grade) else 0.5,
            'conf_trend':           conf_trend,
            'xt_weight':            XT_WEIGHT.get(incident, 0.2),
            'event_density_30s':    len(ev30),
            'event_density_60s':    len(ev60),
            'event_density_120s':   len(ev120),
            'risk_event_count_30s': int(ev30['incident_name'].isin(HIGH_RISK).sum()),
            'risk_event_count_60s': int(ev60['incident_name'].isin(HIGH_RISK).sum()),
            'score_diff':           int(row.score_diff) if pd.notnull(row.score_diff) else 0,
            'minutes_remaining':    float(row.minutes_remaining) if pd.notnull(row.minutes_remaining) else 45.0,
            'period_id':            float(row.period_id) if pd.notnull(row.period_id) else 10.0,
            'mid_price':            current_price,
            'spread':               float(prices.iloc[idx_before]['best_ask'] - prices.iloc[idx_before]['best_bid']),
            'max_abs_move_120s':    max_abs_move,
            'high_volatility':      high_vol,
        })

    return pd.DataFrame(rows)


def main():
    # Load fixture-market matches (163 fixtures × multiple condition_ids)
    fmm = pd.read_parquet(FMM_FILE)
    mf  = pd.read_parquet(MATCHED_FILE)
    print(f"fixture_market_matches: {len(fmm)} rows, {fmm['fixture_id'].nunique()} fixtures")

    # Build hour → condition_ids mapping
    hour_to_cids = defaultdict(set)
    fixture_to_cids = defaultdict(set)

    for _, row in fmm.iterrows():
        base_hour = int(row['market_hour'])
        date_str  = str(row['market_date'])
        cid       = str(row['condition_id'])
        fid       = str(row['fixture_id'])

        base_dt = datetime.datetime.strptime(f"{date_str}T{base_hour:02d}", '%Y-%m-%dT%H')
        for h_offset in range(-1, 3):
            t = base_dt + datetime.timedelta(hours=h_offset)
            hour_key = t.strftime('%Y-%m-%dT%H')
            hour_to_cids[hour_key].add(cid)
        fixture_to_cids[fid].add(cid)

    print(f"Unique PMXT hours: {len(hour_to_cids)}")

    # Build price cache (parallel)
    hour_prices = build_price_cache(hour_to_cids)

    # Consolidate: per fixture, merge all relevant prices
    # condition_id_raw from DuckDB decode(market) gives the raw hex string like '0x...'
    # We match directly against the fixture's condition_ids set
    fixture_price_cache = {}
    for fid, cids in fixture_to_cids.items():
        cids_lower = {c.lower() for c in cids}
        all_dfs = []
        for hour_df in hour_prices.values():
            if hour_df.empty or 'condition_id_raw' not in hour_df.columns:
                continue
            sub = hour_df[hour_df['condition_id_raw'].str.lower().isin(cids_lower)]
            if not sub.empty:
                all_dfs.append(sub)
        if all_dfs:
            fixture_price_cache[fid] = pd.concat(all_dfs, ignore_index=True).sort_values('timestamp')

    print(f"Fixtures with price data: {len(fixture_price_cache)}")

    # Build feature dataset
    all_chunks = []
    for idx, mf_row in mf.iterrows():
        fid  = str(mf_row['fixture_id'])
        date = str(mf_row['event_date'])

        msg_path = MSGS_DIR / date / fid / 'messages.parquet'
        if not msg_path.exists():
            continue
        try:
            msgs = pd.read_parquet(msg_path)
        except Exception:
            continue
        if len(msgs) < 20:
            continue

        msgs = prepare_messages(msgs)

        prices = fixture_price_cache.get(fid, pd.DataFrame())
        if prices.empty:
            print(f"  [{idx+1}] {mf_row['home']} vs {mf_row['away']} — no price data")
            continue

        chunk = label_and_features(msgs, prices)
        if chunk.empty:
            continue

        chunk['fixture_id'] = fid
        all_chunks.append(chunk)

        vol_rate = chunk['high_volatility'].mean()
        print(f"  [{idx+1}] {mf_row['home']} vs {mf_row['away']} "
              f"— {len(chunk)} events, vol={vol_rate:.3f}, "
              f"price=[{prices['mid'].min():.2f},{prices['mid'].max():.2f}]")

    if not all_chunks:
        print("No data collected!")
        return

    dataset = pd.concat(all_chunks, ignore_index=True)
    dataset.to_parquet(OUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"Saved: {len(dataset)} rows, {dataset['fixture_id'].nunique()} fixtures")
    print(f"Volatility rate: {dataset['high_volatility'].mean():.3f}")
    print(f"Columns: {list(dataset.columns)}")


if __name__ == '__main__':
    main()
