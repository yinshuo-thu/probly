"""
Build real aligned dataset using Polymarket CLOB API price history.

For each matched fixture:
1. Get token_ids from CLOB markets API
2. Fetch 1-minute price candles from CLOB prices-history
3. Align with LSports event stream
4. Label high_volatility (absolute price move > 3 cents in next 120s)
5. Extract features

Much faster than PMXT parquet downloads (compact JSON vs 200MB files).
"""

import sys, json, warnings, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '/Volumes/T7/probly/src')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

BASE         = Path('/Volumes/T7/probly')
FMM_FILE     = BASE / 'data/polymarket/fixture_market_matches.parquet'
MATCHED_FILE = BASE / 'data/matched_fixtures.parquet'
MSGS_DIR     = BASE / 'data/hyper/football'
OUT_FILE     = BASE / 'outputs/real_dataset.parquet'
CLOB_API     = 'https://clob.polymarket.com'
MAX_WORKERS  = 8

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


def get_token_ids(condition_id: str) -> list:
    """Get YES/NO token_ids for a market condition_id."""
    try:
        r = requests.get(f'{CLOB_API}/markets/{condition_id}', timeout=15)
        if r.ok:
            return [t['token_id'] for t in r.json().get('tokens', [])]
    except Exception:
        pass
    return []


def get_price_history(token_id: str, start_ts: int, end_ts: int, resolution: int = 60) -> list:
    """Get price candles for a token."""
    try:
        r = requests.get(f'{CLOB_API}/prices-history', params={
            'market': token_id,
            'resolution': resolution,
            'startTs': start_ts,
            'endTs': end_ts,
        }, timeout=15)
        if r.ok:
            return r.json().get('history', [])
    except Exception:
        pass
    return []


def fetch_fixture_prices(condition_id: str, start_utc: pd.Timestamp) -> pd.DataFrame:
    """Fetch price time series for one market (picks the more active token)."""
    token_ids = get_token_ids(condition_id)
    if not token_ids:
        return pd.DataFrame()

    # Time window: 1 hour before match to 3 hours after
    t_start = int((start_utc - pd.Timedelta(hours=1)).timestamp())
    t_end   = int((start_utc + pd.Timedelta(hours=3)).timestamp())

    best_df = pd.DataFrame()
    for tid in token_ids:
        candles = get_price_history(tid, t_start, t_end, resolution=60)
        if not candles:
            continue
        df = pd.DataFrame(candles, columns=['t', 'p'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='s', utc=True)
        df['mid'] = df['p'].astype(float)
        df = df[['timestamp', 'mid']].dropna()
        # Keep the token that trades in [0.05, 0.95] range (more interesting)
        mid_mean = df['mid'].mean()
        if 0.05 < mid_mean < 0.95 and len(df) > len(best_df):
            best_df = df

    return best_df.sort_values('timestamp') if not best_df.empty else pd.DataFrame()


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


def label_and_features(msgs: pd.DataFrame, prices: pd.DataFrame, fixture_id: str) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    msgs = msgs.sort_values('timestamp_utc').copy()
    msgs['ts'] = pd.to_datetime(msgs['timestamp_utc'], utc=True)
    prices = prices.sort_values('timestamp').copy()

    price_ts = prices['timestamp'].values
    mids = prices['mid'].values

    rows = []
    for row in msgs.itertuples():
        t0 = row.ts
        t0_ns = np.datetime64(t0)

        idx_before = np.searchsorted(price_ts, t0_ns, side='right') - 1
        if idx_before < 0:
            continue
        current_price = float(mids[idx_before])
        if current_price <= 0.02 or current_price >= 0.98:
            continue

        # Look ahead 2 minutes (120s)
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
            'fixture_id':           fixture_id,
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
            'spread':               0.01,  # CLOB 1-min candles don't have bid/ask spread
            'max_abs_move_120s':    max_abs_move,
            'high_volatility':      high_vol,
        })

    return pd.DataFrame(rows)


def process_one_fixture(fid, date, cids, home, away):
    """Load messages + fetch prices + build features for one fixture."""
    msg_path = MSGS_DIR / date / fid / 'messages.parquet'
    if not msg_path.exists():
        return None, f"no messages: {home} vs {away}"

    try:
        msgs = pd.read_parquet(msg_path)
    except Exception as e:
        return None, f"read error: {e}"

    if len(msgs) < 20:
        return None, f"too few messages ({len(msgs)}): {home} vs {away}"

    msgs = prepare_messages(msgs)
    start_utc = msgs['timestamp_utc'].min()

    # Try all condition_ids until we get prices
    for cid in cids:
        prices = fetch_fixture_prices(cid, start_utc)
        if not prices.empty and len(prices) > 5:
            chunk = label_and_features(msgs, prices, fid)
            if not chunk.empty:
                return chunk, f"OK: {home} vs {away} — {len(chunk)} events, vol={chunk['high_volatility'].mean():.3f}"

    return None, f"no price data: {home} vs {away}"


def main():
    fmm = pd.read_parquet(FMM_FILE)
    mf  = pd.read_parquet(MATCHED_FILE)
    print(f"Matched fixtures: {mf['fixture_id'].nunique()} fixtures to process")

    # Build fixture → condition_ids map (from matched_fixtures, not fmm)
    fixture_cids = {}
    for _, row in mf.iterrows():
        fid  = str(row['fixture_id'])
        date = str(row['event_date'])
        cids = list(row['condition_ids']) if row['condition_ids'] is not None else []
        if cids:
            fixture_cids[fid] = (date, cids, str(row['home']), str(row['away']))

    print(f"Fixtures with condition_ids: {len(fixture_cids)}")

    # Process in parallel
    all_chunks = []
    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_one_fixture, fid, date, cids, home, away): fid
            for fid, (date, cids, home, away) in fixture_cids.items()
        }
        for fut in as_completed(futures):
            fid = futures[fut]
            try:
                chunk, msg = fut.result()
            except Exception as e:
                chunk, msg = None, str(e)

            done += 1
            if chunk is not None and not chunk.empty:
                all_chunks.append(chunk)
                print(f"  [{done}/{len(fixture_cids)}] {msg}")
            else:
                failed += 1
                if done % 10 == 0 or failed <= 5:
                    print(f"  [{done}/{len(fixture_cids)}] SKIP — {msg}")

    if not all_chunks:
        print("No data collected!")
        return

    dataset = pd.concat(all_chunks, ignore_index=True)
    dataset.to_parquet(OUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"Saved: {len(dataset)} rows, {dataset['fixture_id'].nunique()} fixtures")
    print(f"Volatility rate: {dataset['high_volatility'].mean():.3f}")
    print(f"Failed fixtures: {failed}/{len(fixture_cids)}")
    print(f"Columns: {list(dataset.columns)}")


if __name__ == '__main__':
    main()
