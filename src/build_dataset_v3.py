"""
Build real aligned dataset - v3, using fixture_market_matches.parquet.

All 163 fixtures here have verified Polymarket condition_ids.
Fetches price history from CLOB API (fast JSON, not 200MB parquet files).
Vectorized O(N log N) label_and_features instead of O(N^2).
"""

import sys, warnings, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '/Volumes/T7/probly/src')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

BASE     = Path('/Volumes/T7/probly')
FMM_FILE = BASE / 'data/polymarket/fixture_market_matches.parquet'
MSGS_DIR = BASE / 'data/hyper/football'
OUT_FILE = BASE / 'outputs/real_dataset.parquet'
CLOB_API = 'https://clob.polymarket.com'
MAX_WORKERS = 6

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
    try:
        r = requests.get(f'{CLOB_API}/markets/{condition_id}', timeout=15)
        if r.ok:
            return [t['token_id'] for t in r.json().get('tokens', [])]
    except Exception:
        pass
    return []


def get_price_candles(token_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    try:
        r = requests.get(f'{CLOB_API}/prices-history', params={
            'market': token_id, 'resolution': 60,
            'startTs': start_ts, 'endTs': end_ts,
        }, timeout=15)
        if r.ok:
            hist = r.json().get('history', [])
            if hist:
                df = pd.DataFrame(hist, columns=['t', 'p'])
                # Multiply by 10^9 to force nanosecond precision (avoids datetime64[s] vs [ns] mismatch)
                df['timestamp'] = pd.to_datetime(df['t'].astype('int64') * 10**9, unit='ns', utc=True)
                df['mid'] = df['p'].astype(float)
                return df[['timestamp', 'mid']].sort_values('timestamp')
    except Exception:
        pass
    return pd.DataFrame()


def fetch_best_prices(condition_ids: list, start_utc: pd.Timestamp) -> pd.DataFrame:
    t_start = int((start_utc - pd.Timedelta(hours=1)).timestamp())
    t_end   = int((start_utc + pd.Timedelta(hours=4)).timestamp())

    best_df = pd.DataFrame()
    best_score = -1

    for cid in condition_ids:
        tokens = get_token_ids(cid)
        for tid in tokens:
            df = get_price_candles(tid, t_start, t_end)
            if df.empty:
                continue
            mid_mean = df['mid'].mean()
            score = len(df) * (1 - abs(mid_mean - 0.5) * 2)
            if score > best_score and 0.05 < mid_mean < 0.95:
                best_score = score
                best_df = df

    return best_df


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
    """Vectorized O(N log N) feature extraction using numpy searchsorted."""
    if prices.empty:
        return pd.DataFrame()

    msgs = msgs.sort_values('timestamp_utc').copy()
    msgs['ts'] = pd.to_datetime(msgs['timestamp_utc'], utc=True)
    prices = prices.sort_values('timestamp').copy()

    price_ts = prices['timestamp'].values.astype('int64')
    mids = prices['mid'].values

    # Pre-build numpy arrays for vectorized lookups
    msg_ts_ns = msgs['ts'].values.astype('int64')
    msg_is_risk = msgs['incident_name'].isin(HIGH_RISK).values

    # Window sizes in nanoseconds
    NS30  = 30  * 10**9
    NS60  = 60  * 10**9
    NS120 = 120 * 10**9

    rows = []
    for i, row in enumerate(msgs.itertuples()):
        t0_ns = msg_ts_ns[i]

        # Find current price via searchsorted
        idx_before = np.searchsorted(price_ts, t0_ns, side='right') - 1
        if idx_before < 0:
            continue
        current_price = float(mids[idx_before])
        if current_price <= 0.02 or current_price >= 0.98:
            continue

        # Future price move (label)
        t_end_ns = t0_ns + NS120
        future_mask = (price_ts > t0_ns) & (price_ts <= t_end_ns)
        future_mids = mids[future_mask]
        max_abs_move = float(np.abs(future_mids - current_price).max()) if len(future_mids) > 0 else 0.0
        high_vol = int(max_abs_move > 0.03)

        # Event density using searchsorted on sorted msg_ts_ns (O(log N) each)
        idx_now   = np.searchsorted(msg_ts_ns, t0_ns,        side='left')
        idx_30    = np.searchsorted(msg_ts_ns, t0_ns - NS30,  side='left')
        idx_60    = np.searchsorted(msg_ts_ns, t0_ns - NS60,  side='left')
        idx_120   = np.searchsorted(msg_ts_ns, t0_ns - NS120, side='left')

        ev30_count  = idx_now - idx_30
        ev60_count  = idx_now - idx_60
        ev120_count = idx_now - idx_120

        risk_30  = int(msg_is_risk[idx_30:idx_now].sum())
        risk_60  = int(msg_is_risk[idx_60:idx_now].sum())

        # Confidence trend (last 5 conf values in past 60s)
        recent_conf = msgs['confidence_grade'].values[idx_60:idx_now]
        recent_conf = recent_conf[~np.isnan(recent_conf.astype(float))][-5:]
        conf_trend = float(np.polyfit(range(len(recent_conf)), recent_conf.astype(float), 1)[0]) if len(recent_conf) > 1 else 0.0

        incident = str(row.incident_name)
        rows.append({
            'fixture_id':           fixture_id,
            'timestamp':            str(row.ts),
            'incident_name':        incident,
            'confidence_grade':     float(row.confidence_grade) if pd.notnull(row.confidence_grade) else 0.5,
            'conf_trend':           conf_trend,
            'xt_weight':            XT_WEIGHT.get(incident, 0.2),
            'event_density_30s':    ev30_count,
            'event_density_60s':    ev60_count,
            'event_density_120s':   ev120_count,
            'risk_event_count_30s': risk_30,
            'risk_event_count_60s': risk_60,
            'score_diff':           int(row.score_diff) if pd.notnull(row.score_diff) else 0,
            'minutes_remaining':    float(row.minutes_remaining) if pd.notnull(row.minutes_remaining) else 45.0,
            'period_id':            float(row.period_id) if pd.notnull(row.period_id) else 10.0,
            'mid_price':            current_price,
            'spread':               0.01,
            'max_abs_move_120s':    max_abs_move,
            'high_volatility':      high_vol,
        })

    return pd.DataFrame(rows)


def process_fixture(fid, date, condition_ids, home, away):
    msg_path = MSGS_DIR / date / str(fid) / 'messages.parquet'
    if not msg_path.exists():
        return None, f"no messages: {home} vs {away}"
    try:
        msgs = pd.read_parquet(msg_path)
    except Exception as e:
        return None, f"read error: {e}"
    if len(msgs) < 20:
        return None, f"too few msgs: {home} vs {away}"

    msgs = prepare_messages(msgs)
    start_utc = msgs['timestamp_utc'].min()

    prices = fetch_best_prices(condition_ids, start_utc)
    if prices.empty:
        return None, f"no prices: {home} vs {away}"

    chunk = label_and_features(msgs, prices, str(fid))
    if chunk.empty:
        return None, f"no rows: {home} vs {away}"

    return chunk, f"OK: {home} vs {away} — {len(chunk)} events, vol={chunk['high_volatility'].mean():.3f}"


def main():
    fmm = pd.read_parquet(FMM_FILE)
    print(f"Loading fixture_market_matches: {fmm['fixture_id'].nunique()} fixtures, {fmm['condition_id'].nunique()} markets")

    fixture_groups = {}
    for _, row in fmm.iterrows():
        fid  = int(row['fixture_id'])
        date = str(row['event_date'])
        home = str(row['home'])
        away = str(row['away'])
        cid  = str(row['condition_id'])
        if fid not in fixture_groups:
            fixture_groups[fid] = {'date': date, 'home': home, 'away': away, 'cids': []}
        fixture_groups[fid]['cids'].append(cid)

    print(f"Processing {len(fixture_groups)} unique fixtures with {MAX_WORKERS} workers...")

    all_chunks = []
    done = failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_fixture,
                      fid, g['date'], g['cids'], g['home'], g['away']): fid
            for fid, g in fixture_groups.items()
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
                print(f"  [{done}/{len(fixture_groups)}] {msg}", flush=True)
            else:
                failed += 1
                if failed <= 15 or done % 20 == 0:
                    print(f"  [{done}/{len(fixture_groups)}] SKIP — {msg}", flush=True)

    if not all_chunks:
        print("No data collected!")
        return

    dataset = pd.concat(all_chunks, ignore_index=True)
    dataset.to_parquet(OUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"Saved: {len(dataset)} rows, {dataset['fixture_id'].nunique()} fixtures")
    print(f"Volatility rate: {dataset['high_volatility'].mean():.3f}")
    print(f"Failed: {failed}/{len(fixture_groups)}")
    print(f"Columns: {list(dataset.columns)}")


if __name__ == '__main__':
    main()
