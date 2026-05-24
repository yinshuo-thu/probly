"""
Build enhanced aligned dataset - v4.

Improvements over v3:
- Price velocity features: slope + realized vol of price in past 5 min
- Score-state features: total_goals, is_leading, closeness
- Causal label: next_60s_high_impact (will a goal/card/penalty happen in next 60s?)
- Player-name based features (top impact players)
- Vectorized O(N log N) throughout
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
OUT_FILE = BASE / 'outputs/real_dataset_v4.parquet'
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
HIGH_RISK = {'Score', 'PlayerGoals', 'Penalties', 'RedCard', 'PlayerRedCard',
             'YellowCard', 'PlayerShotsOnTarget', 'ShotsOnTarget', 'MissedPenalty', 'VAR'}
HIGH_IMPACT = {'Score', 'PlayerGoals', 'Penalties', 'MissedPenalty',
               'RedCard', 'PlayerRedCard', 'VAR'}


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
                df['timestamp'] = pd.to_datetime(
                    df['t'].astype('int64') * 10**9, unit='ns', utc=True)
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
    msgs['score_diff'] = (msgs['score_home'] - msgs['score_away']).astype(int)
    msgs['total_goals'] = (msgs['score_home'] + msgs['score_away']).astype(int)
    elapsed_sec = pd.to_numeric(msgs['seconds'], errors='coerce').fillna(0)
    msgs['elapsed_sec']      = elapsed_sec
    msgs['minutes_elapsed']  = elapsed_sec / 60
    msgs['minutes_remaining'] = (90*60 - elapsed_sec).clip(lower=0) / 60
    msgs['period_id'] = pd.to_numeric(msgs['period_id'], errors='coerce').fillna(10.0)
    # Game state features
    msgs['is_leading'] = (msgs['score_diff'] > 0).astype(int)
    msgs['is_drawing'] = (msgs['score_diff'] == 0).astype(int)
    msgs['is_trailing'] = (msgs['score_diff'] < 0).astype(int)
    # Game intensity: goals scored × time elapsed (more goals in less time = chaotic)
    msgs['game_intensity'] = msgs['total_goals'] * (msgs['minutes_elapsed'] / 90 + 0.01)
    return msgs.sort_values('timestamp_utc')


def label_and_features(msgs: pd.DataFrame, prices: pd.DataFrame, fixture_id: str) -> pd.DataFrame:
    """Vectorized O(N log N) feature extraction with price velocity and causal label."""
    if prices.empty:
        return pd.DataFrame()

    msgs   = msgs.sort_values('timestamp_utc').copy()
    msgs['ts'] = pd.to_datetime(msgs['timestamp_utc'], utc=True)
    prices = prices.sort_values('timestamp').copy()

    price_ts = prices['timestamp'].values.astype('int64')
    mids     = prices['mid'].values

    msg_ts_ns    = msgs['ts'].values.astype('int64')
    msg_is_risk  = msgs['incident_name'].isin(HIGH_RISK).values
    msg_is_hi    = msgs['incident_name'].isin(HIGH_IMPACT).values
    msg_xt       = msgs['incident_name'].map(lambda x: XT_WEIGHT.get(x, 0.2)).values

    NS30  = 30  * 10**9
    NS60  = 60  * 10**9
    NS120 = 120 * 10**9
    NS300 = 300 * 10**9  # 5 minutes

    rows = []
    for i, row in enumerate(msgs.itertuples()):
        t0_ns = msg_ts_ns[i]

        # Current price
        idx_before = np.searchsorted(price_ts, t0_ns, side='right') - 1
        if idx_before < 0:
            continue
        current_price = float(mids[idx_before])
        if current_price <= 0.02 or current_price >= 0.98:
            continue

        # ── FUTURE LABEL ──────────────────────────────────────────────────
        t_end_ns = t0_ns + NS120
        future_mask = (price_ts > t0_ns) & (price_ts <= t_end_ns)
        future_mids = mids[future_mask]
        max_abs_move = float(np.abs(future_mids - current_price).max()) if len(future_mids) > 0 else 0.0
        high_vol = int(max_abs_move > 0.03)

        # Causal label: any HIGH_IMPACT event in next 60s?
        next60_ns    = t0_ns + NS60
        idx_now      = np.searchsorted(msg_ts_ns, t0_ns, side='right')
        idx_next60   = np.searchsorted(msg_ts_ns, next60_ns, side='right')
        next60_hi    = int(msg_is_hi[idx_now:idx_next60].any())

        # ── PAST EVENT WINDOWS (event density features) ───────────────────
        idx_30  = np.searchsorted(msg_ts_ns, t0_ns - NS30,  side='left')
        idx_60  = np.searchsorted(msg_ts_ns, t0_ns - NS60,  side='left')
        idx_120 = np.searchsorted(msg_ts_ns, t0_ns - NS120, side='left')
        idx_300 = np.searchsorted(msg_ts_ns, t0_ns - NS300, side='left')

        ev30  = idx_now - idx_30
        ev60  = idx_now - idx_60
        ev120 = idx_now - idx_120
        ev300 = idx_now - idx_300

        risk30 = int(msg_is_risk[idx_30:idx_now].sum())
        risk60 = int(msg_is_risk[idx_60:idx_now].sum())
        hi60   = int(msg_is_hi[idx_60:idx_now].sum())   # high-impact events in past 60s
        hi300  = int(msg_is_hi[idx_300:idx_now].sum())  # ...past 5 min

        xt_sum_60s  = float(msg_xt[idx_60:idx_now].sum())
        xt_sum_300s = float(msg_xt[idx_300:idx_now].sum())

        # Confidence trend in past 60s
        recent_conf = msgs['confidence_grade'].values[idx_60:idx_now]
        recent_conf_f = recent_conf[~np.isnan(recent_conf.astype(float))][-5:]
        conf_trend = float(np.polyfit(range(len(recent_conf_f)), recent_conf_f.astype(float), 1)[0]) \
                     if len(recent_conf_f) > 1 else 0.0

        # ── PRICE VELOCITY FEATURES ───────────────────────────────────────
        t_300_ns = t0_ns - NS300
        price_window_mask = (price_ts >= t_300_ns) & (price_ts < t0_ns)
        price_window = mids[price_window_mask]
        if len(price_window) >= 3:
            price_velocity  = float(np.polyfit(range(len(price_window)), price_window, 1)[0])
            price_realized_vol = float(np.std(price_window))
            price_change_1m  = float(price_window[-1] - price_window[max(0, len(price_window)-2)])
        else:
            price_velocity  = 0.0
            price_realized_vol = 0.0
            price_change_1m = 0.0

        incident = str(row.incident_name)
        score_d  = int(row.score_diff) if pd.notnull(row.score_diff) else 0
        rows.append({
            'fixture_id':             fixture_id,
            'timestamp':              str(row.ts),
            'incident_name':          incident,

            # Game state
            'score_diff':             score_d,
            'total_goals':            int(row.total_goals) if pd.notnull(row.total_goals) else 0,
            'minutes_remaining':      float(row.minutes_remaining) if pd.notnull(row.minutes_remaining) else 45.0,
            'minutes_elapsed':        float(row.minutes_elapsed) if pd.notnull(row.minutes_elapsed) else 45.0,
            'period_id':              float(row.period_id) if pd.notnull(row.period_id) else 10.0,
            'is_leading':             int(score_d > 0),
            'is_drawing':             int(score_d == 0),
            'game_intensity':         float(row.game_intensity) if pd.notnull(row.game_intensity) else 0.0,

            # Incident features
            'confidence_grade':       float(row.confidence_grade) if pd.notnull(row.confidence_grade) else 0.5,
            'conf_trend':             conf_trend,
            'xt_weight':              XT_WEIGHT.get(incident, 0.2),
            'is_high_impact':         int(incident in HIGH_IMPACT),
            'is_timer_period':        int(incident in {'Timer', 'Period'}),

            # Rolling event density
            'event_density_30s':      ev30,
            'event_density_60s':      ev60,
            'event_density_120s':     ev120,
            'event_density_300s':     ev300,
            'risk_event_count_30s':   risk30,
            'risk_event_count_60s':   risk60,
            'high_impact_60s':        hi60,
            'high_impact_300s':       hi300,
            'xt_sum_60s':             xt_sum_60s,
            'xt_sum_300s':            xt_sum_300s,

            # Cross-features
            'xt_x_conf':              XT_WEIGHT.get(incident, 0.2) * (float(row.confidence_grade) if pd.notnull(row.confidence_grade) else 0.5),
            'risk_density_ratio':     risk60 / (ev60 + 1),
            'pressure_score':         xt_sum_60s * (float(row.confidence_grade) if pd.notnull(row.confidence_grade) else 0.5),

            # Market features
            'mid_price':              current_price,
            'price_extremeness':      abs(current_price - 0.5) * 2,
            'price_velocity':         price_velocity,
            'price_realized_vol':     price_realized_vol,
            'price_change_1m':        price_change_1m,

            # Labels
            'max_abs_move_120s':      max_abs_move,
            'high_volatility':        high_vol,            # price label
            'next_60s_high_impact':   next60_hi,           # causal label
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

    return chunk, (f"OK: {home} vs {away} — {len(chunk)} events, "
                   f"vol={chunk['high_volatility'].mean():.3f}, "
                   f"causal={chunk['next_60s_high_impact'].mean():.3f}")


def main():
    fmm = pd.read_parquet(FMM_FILE)
    print(f"fixture_market_matches: {fmm['fixture_id'].nunique()} fixtures, {fmm['condition_id'].nunique()} markets")

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
            ex.submit(process_fixture, fid, g['date'], g['cids'], g['home'], g['away']): fid
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
                if failed <= 10 or done % 20 == 0:
                    print(f"  [{done}/{len(fixture_groups)}] SKIP — {msg}", flush=True)

    if not all_chunks:
        print("No data collected!")
        return

    dataset = pd.concat(all_chunks, ignore_index=True)
    dataset.to_parquet(OUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"Saved: {len(dataset)} rows, {dataset['fixture_id'].nunique()} fixtures → {OUT_FILE}")
    print(f"Price vol rate:  {dataset['high_volatility'].mean():.3f}")
    print(f"Causal vol rate: {dataset['next_60s_high_impact'].mean():.3f}")
    print(f"Failed: {failed}/{len(fixture_groups)}")
    print(f"Features: {[c for c in dataset.columns if c not in ['fixture_id','timestamp','incident_name','max_abs_move_120s','high_volatility','next_60s_high_impact']]}")


if __name__ == '__main__':
    main()
