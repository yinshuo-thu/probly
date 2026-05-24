"""
Build the real aligned dataset from LSports messages + PMXT Polymarket prices.

Steps:
1. For each matched fixture, load messages.parquet
2. For each fixture's condition_ids, fetch PMXT price data via DuckDB
3. Align events to prices by timestamp
4. Label high_volatility (absolute price move > 3 cents in next 120s)
5. Extract features
6. Save to outputs/real_dataset.parquet
"""

import sys, os, json, warnings
sys.path.insert(0, '/Volumes/T7/probly/src')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

BASE = Path('/Volumes/T7/probly')
MATCHED_FILE  = BASE / 'data/matched_fixtures.parquet'
MSGS_DIR      = BASE / 'data/hyper/football'
OUT_FILE      = BASE / 'outputs/real_dataset.parquet'
PMXT_BASE_V2  = 'https://r2v2.pmxt.dev/polymarket_orderbook_{hour}.parquet'

# ── Incident → xT threat weight ─────────────────────────────────────────────
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


def fetch_pmxt_prices(condition_ids: list, start_utc: pd.Timestamp, hours: int = 4) -> pd.DataFrame:
    """Fetch price_change events from PMXT for given condition_ids and time window."""
    # Convert condition_id hex to integer for comparison
    try:
        cid_ints = [str(int(c, 16)) for c in condition_ids if c and c.startswith('0x')]
    except Exception:
        return pd.DataFrame()

    if not cid_ints:
        return pd.DataFrame()

    # Build list of PMXT hours to query
    hour_urls = []
    for h in range(-1, hours + 1):
        t = start_utc + pd.Timedelta(hours=h)
        hour_str = t.strftime('%Y-%m-%dT%H')
        hour_urls.append(PMXT_BASE_V2.format(hour=hour_str))

    cid_list = "', '".join(cid_ints)
    all_dfs = []

    for url in hour_urls:
        try:
            result = subprocess.run(
                ['python3.13', '-c', f"""
import duckdb, pandas as pd, sys
con = duckdb.connect()
con.execute("SET http_timeout=45000;")
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
          AND CAST(unhex(substr(hex(market), 5)) AS HUGEINT)::VARCHAR IN ('{cid_list}')
        ORDER BY timestamp
    ''').df()
    print(df.to_json(orient='records'))
except Exception as e:
    print('[]')
"""],
                capture_output=True, text=True, timeout=60
            )
            if result.stdout and result.stdout.strip() not in ('[]', ''):
                records = json.loads(result.stdout.strip())
                if records:
                    df = pd.DataFrame(records)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    all_dfs.append(df)
        except Exception:
            pass

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True).sort_values('timestamp')


def label_and_features(msgs: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Align LSports events with Polymarket prices and build feature+label rows."""
    msgs = msgs.sort_values('timestamp_utc').copy()
    msgs['ts'] = pd.to_datetime(msgs['timestamp_utc'], utc=True)

    if prices.empty:
        return pd.DataFrame()

    prices = prices.sort_values('timestamp').copy()
    prices['ts'] = pd.to_datetime(prices['timestamp'], utc=True)
    prices['mid'] = (prices['best_bid'] + prices['best_ask']) / 2

    rows = []
    msg_ts = msgs['ts'].values
    price_ts = prices['ts'].values
    mids = prices['mid'].values

    for i, row in enumerate(msgs.itertuples()):
        t0 = row.ts
        t0_ns = np.datetime64(t0)

        # Nearest Polymarket price at or before event
        idx_before = np.searchsorted(price_ts, t0_ns, side='right') - 1
        if idx_before < 0:
            continue
        current_price = mids[idx_before]
        if current_price <= 0 or current_price >= 1:
            continue

        # Future prices in next 120s
        t_end = t0 + pd.Timedelta(seconds=120)
        t_end_ns = np.datetime64(t_end)
        future_mask = (price_ts > t0_ns) & (price_ts <= t_end_ns)
        future_mids = mids[future_mask]

        if len(future_mids) == 0:
            max_abs_move = 0.0
        else:
            max_abs_move = float(np.abs(future_mids - current_price).max())

        # High volatility: absolute move > 0.03 (3 cents on a 0-1 scale)
        high_vol = int(max_abs_move > 0.03)

        # Rolling features (look-back windows)
        t30 = t0 - pd.Timedelta(seconds=30)
        t60 = t0 - pd.Timedelta(seconds=60)
        t120 = t0 - pd.Timedelta(seconds=120)

        mask30  = (msgs['ts'] >= t30) & (msgs['ts'] < t0)
        mask60  = (msgs['ts'] >= t60) & (msgs['ts'] < t0)
        mask120 = (msgs['ts'] >= t120) & (msgs['ts'] < t0)

        ev30  = msgs[mask30]
        ev60  = msgs[mask60]
        ev120 = msgs[mask120]

        def risk_count(subset):
            return subset['incident_name'].isin(HIGH_RISK).sum()

        # Confidence trend (slope of last 5 confidence values)
        recent_conf = msgs[mask60]['confidence_grade'].dropna().values[-5:]
        conf_trend = float(np.polyfit(range(len(recent_conf)), recent_conf, 1)[0]) if len(recent_conf) > 1 else 0.0

        incident = str(row.incident_name)
        rows.append({
            'fixture_id':          str(row.fixture_id),
            'timestamp':           str(t0),
            'incident_name':       incident,
            'confidence_grade':    float(row.confidence_grade) if pd.notnull(row.confidence_grade) else 0.5,
            'conf_trend':          conf_trend,
            'xt_weight':           XT_WEIGHT.get(incident, 0.2),
            'event_density_30s':   len(ev30),
            'event_density_60s':   len(ev60),
            'event_density_120s':  len(ev120),
            'risk_event_count_30s': int(risk_count(ev30)),
            'risk_event_count_60s': int(risk_count(ev60)),
            'score_diff':          int(row.score_diff) if hasattr(row, 'score_diff') and pd.notnull(row.score_diff) else 0,
            'minutes_remaining':   float(row.minutes_remaining) if hasattr(row, 'minutes_remaining') and pd.notnull(row.minutes_remaining) else 45.0,
            'period_id':           float(row.period_id) if pd.notnull(row.period_id) else 10.0,
            'mid_price':           current_price,
            'spread':              float(prices.iloc[idx_before]['best_ask'] - prices.iloc[idx_before]['best_bid']),
            'max_abs_move_120s':   max_abs_move,
            'high_volatility':     high_vol,
        })

    return pd.DataFrame(rows)


def prepare_messages(msgs: pd.DataFrame) -> pd.DataFrame:
    """Add score_diff and minutes_remaining to messages dataframe."""
    msgs = msgs.copy()
    msgs['timestamp_utc'] = pd.to_datetime(msgs['timestamp_utc'], utc=True)

    # Score
    msgs['score_home'] = pd.to_numeric(msgs['home_value'], errors='coerce').fillna(0)
    msgs['score_away'] = pd.to_numeric(msgs['away_value'], errors='coerce').fillna(0)
    msgs['score_diff']  = (msgs['score_home'] - msgs['score_away']).astype(int)

    # Time
    elapsed_sec = pd.to_numeric(msgs['seconds'], errors='coerce').fillna(0)
    msgs['minutes_remaining'] = (90*60 - elapsed_sec).clip(lower=0) / 60

    # Period
    msgs['period_id'] = pd.to_numeric(msgs['period_id'], errors='coerce').fillna(10.0)
    return msgs.sort_values('timestamp_utc')


def main():
    mf = pd.read_parquet(MATCHED_FILE)
    print(f"Matched fixtures: {len(mf)}")

    all_chunks = []
    failed = 0

    for idx, row in mf.iterrows():
        fid  = str(row['fixture_id'])
        date = str(row['event_date'])
        cids = list(row['condition_ids']) if row['condition_ids'] is not None else []

        if not cids:
            continue

        # Load messages
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
        start_utc = msgs['timestamp_utc'].min()

        # Fetch PMXT prices
        prices = fetch_pmxt_prices(cids, start_utc, hours=4)

        if prices.empty:
            failed += 1
            print(f"  [{idx+1}/{len(mf)}] {row['home']} vs {row['away']} — no price data")
            continue

        # Build features
        chunk = label_and_features(msgs, prices)
        if chunk.empty:
            continue

        chunk['fixture_id'] = fid
        all_chunks.append(chunk)

        vol_rate = chunk['high_volatility'].mean()
        print(f"  [{idx+1}/{len(mf)}] {row['home']} vs {row['away']} "
              f"— {len(chunk)} events, vol_rate={vol_rate:.3f}, "
              f"price_range=[{prices['mid'].min():.2f},{prices['mid'].max():.2f}]")

    if not all_chunks:
        print("No data collected!")
        return

    dataset = pd.concat(all_chunks, ignore_index=True)
    dataset.to_parquet(OUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"Dataset saved: {len(dataset)} rows, {dataset['fixture_id'].nunique()} fixtures")
    print(f"Volatility rate: {dataset['high_volatility'].mean():.3f}")
    print(f"Columns: {list(dataset.columns)}")
    print(f"Failed (no price data): {failed}/{len(mf)}")


if __name__ == '__main__':
    main()
