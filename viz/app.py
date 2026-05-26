"""
Probly Sports Pricing — Rich Interactive Dashboard

Run: python viz/app.py
Then open: http://localhost:5001
"""

import sys, json, math, os
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

BASE       = Path(__file__).parent.parent
OUTPUTS    = BASE / 'outputs'
CLOB_API   = 'https://clob.polymarket.com'
FMM_FILE   = BASE / 'data/polymarket/fixture_market_matches.parquet'
PMXT_DIR   = BASE / 'data/polymarket/prices'

app = Flask(__name__)

# ── cached data ────────────────────────────────────────────────────────────────
_pred_df   = None
_fmm_df    = None
_pmxt_cache = {}


def _json_safe(value, default=None):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else default
    return value


def _iso(ts):
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _condition_price_path(condition_id: str) -> Path:
    safe = str(condition_id).replace('0x', '')[:32]
    return PMXT_DIR / f'{safe}.parquet'

def pred_df():
    global _pred_df
    if _pred_df is None:
        paths = [OUTPUTS / 'test_predictions.parquet']
        for p in paths:
            if p.exists():
                _pred_df = pd.read_parquet(p)
                _pred_df['timestamp'] = pd.to_datetime(_pred_df['timestamp'], utc=True, errors='coerce')
                _pred_df['fixture_id'] = _pred_df['fixture_id'].astype(str)
                break
    return _pred_df

def fmm_df():
    global _fmm_df
    if _fmm_df is None and FMM_FILE.exists():
        _fmm_df = pd.read_parquet(FMM_FILE)
        _fmm_df['fixture_id'] = _fmm_df['fixture_id'].astype(str)
    return _fmm_df


def fixture_markets(fixture_id: str) -> list:
    fmm = fmm_df()
    if fmm is None:
        return []
    rows = fmm[fmm['fixture_id'].astype(str) == str(fixture_id)].copy()
    markets = []
    for _, r in rows.iterrows():
        cid = str(r['condition_id'])
        p = _condition_price_path(cid)
        pmxt_rows = 0
        if p.exists():
            try:
                pmxt_rows = int(pq.ParquetFile(p).metadata.num_rows)
            except Exception:
                pmxt_rows = 0
        markets.append({
            'condition_id': cid,
            'question': str(r.get('question', '')),
            'market_type': str(r.get('market_type', 'unknown')),
            'has_pmxt': p.exists(),
            'pmxt_rows': pmxt_rows,
        })
    return sorted(markets, key=lambda x: (not x['has_pmxt'], -x['pmxt_rows'], x['market_type']))


def load_pmxt_prices(condition_id: str,
                     start_ts: pd.Timestamp,
                     end_ts: pd.Timestamp,
                     target_mean: float | None = None,
                     max_points: int = 2600) -> tuple[pd.DataFrame, dict]:
    """Load local PMXT orderbook updates and downsample to a UI-friendly series.

    PMXT files contain both binary outcomes. When asset_id is present, choose the
    outcome whose mean mid is closest to the model-aligned CLOB series so the
    direction is consistent with the existing labels.
    """
    path = _condition_price_path(condition_id)
    if not path.exists():
        return pd.DataFrame(), {}

    key = (str(condition_id), str(start_ts), str(end_ts), round(float(target_mean or 0.0), 4), max_points)
    if key in _pmxt_cache:
        return _pmxt_cache[key]

    df = pd.read_parquet(path)
    if df.empty or 'timestamp' not in df.columns:
        return pd.DataFrame(), {}

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
    if df.empty:
        return pd.DataFrame(), {}

    for col in ['best_bid', 'best_ask', 'price', 'size']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['mid'] = (df['best_bid'] + df['best_ask']) / 2
    df = df[(df['mid'] > 0.005) & (df['mid'] < 0.995)]
    if df.empty:
        return pd.DataFrame(), {}

    chosen = 'all'
    if 'asset_id' in df.columns and df['asset_id'].nunique() > 1:
        candidates = []
        for asset, g in df.groupby('asset_id'):
            mid_mean = float(g['mid'].mean())
            target = target_mean if target_mean is not None and math.isfinite(target_mean) else 0.5
            candidates.append((abs(mid_mean - target), -len(g), str(asset), g))
        _, _, chosen, df = sorted(candidates, key=lambda x: (x[0], x[1]))[0]
    elif 'side' in df.columns and (df['side'] == 'BUY').any():
        buy = df[df['side'] == 'BUY']
        if len(buy) >= 10:
            chosen = 'BUY'
            df = buy

    df = (df.sort_values('timestamp')
            .drop_duplicates(subset=['timestamp'], keep='last')
            .set_index('timestamp'))

    # Preserve sub-minute movement while avoiding tens of thousands of browser points.
    if len(df) > max_points:
        df = df.resample('2s').last().dropna(subset=['mid'])
    if len(df) > max_points:
        step = int(math.ceil(len(df) / max_points))
        df = df.iloc[::step]

    out = df.reset_index()[['timestamp', 'mid', 'best_bid', 'best_ask', 'price', 'size']].copy()
    out['t'] = out['timestamp'].map(_iso)
    meta = {
        'condition_id': str(condition_id),
        'source': 'PMXT local orderbook',
        'points': int(len(out)),
        'chosen_asset': chosen,
        'has_bid_ask': bool({'best_bid', 'best_ask'}.issubset(out.columns)),
    }
    result = (out, meta)
    _pmxt_cache[key] = result
    return result

def load_metrics():
    p = OUTPUTS / 'metrics_history.json'
    return json.load(open(p)) if p.exists() else {}


def load_json_output(name, default):
    p = OUTPUTS / name
    if not p.exists():
        return default
    try:
        return json.load(open(p))
    except Exception:
        return default


def best_threshold(default=0.5) -> float:
    m = load_metrics()
    models = m.get('models', [])
    if not models:
        return default
    best_name = m.get('best_model')
    best = next((x for x in models if x.get('model') == best_name), None)
    if best is None:
        best = max(models, key=lambda x: x.get('recall' if m.get('optimize_for') == 'coverage_recall' else 'f1', 0))
    return float(best.get('threshold', default))


def pricing_threshold(default=0.64) -> float:
    """High-confidence threshold for pricing alerts, not recall-max coverage."""
    m = load_json_output('lead_pricing_metrics.json', {})
    try:
        return float(m['test']['high_confidence']['threshold'])
    except Exception:
        return default


def add_pricing_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create stricter, trade-facing pricing signals from lead path forecasts.

    The recall classifier intentionally fires broadly. For the UI default we
    require a lead score, expected move magnitude, and expected fair-price edge.
    """
    out = df.copy()
    price = pd.to_numeric(out.get('price', out.get('mid_price', 0.5)), errors='coerce').fillna(0.5)
    score = pd.to_numeric(out.get('lead30_intercept_score', 0), errors='coerce').fillna(0).clip(0, 2)
    pred_abs = pd.to_numeric(out.get('lead30_pred_abs_peak', out.get('lead30_pred_abs_move', out.get('lead30_pred_abs_move_120s', 0))), errors='coerce').fillna(0)
    pred_after = pd.to_numeric(out.get('lead30_pred_price_after', out.get('lead30_pred_price_after_150s', np.nan)), errors='coerce')
    edge = (pred_after - price).abs().fillna(0)
    th = pricing_threshold()
    out['pricing_signal'] = score
    out['pricing_edge'] = edge
    out['pricing_alert'] = ((score >= th) & (pred_abs >= 0.035) & (edge >= 0.015)).astype(int)
    return out

def load_summary():
    ds_v4 = OUTPUTS / 'real_dataset_v4.parquet'
    ds_v3 = OUTPUTS / 'real_dataset.parquet'
    metrics = load_metrics()
    for ds_path in [ds_v4, ds_v3]:
        if ds_path.exists():
            df = pd.read_parquet(ds_path, columns=['fixture_id','high_volatility'])
            return {
                'dataset': ds_path.name,
                'total_rows': int(len(df)),
                'fixtures':   int(df['fixture_id'].nunique()),
                'vol_rate':   round(float(df['high_volatility'].mean()), 4),
                'best_model': metrics.get('best_model','N/A'),
                'best_f1':    metrics.get('best_f1', 0.0),
            }
    return {}


@app.route('/api/data_coverage')
def api_data_coverage():
    """How much football Hyper data is currently used by the Polymarket study."""
    msg_files = [p for p in (BASE / 'data/hyper/football').glob('*/*/messages.parquet')
                 if not p.name.startswith('._')]
    fixture_dirs = {p.parent.name for p in msg_files}
    snapshot_path = OUTPUTS / 'data_coverage_snapshot.json'
    snapshot = {}
    if snapshot_path.exists():
        try:
            snapshot = json.load(open(snapshot_path))
        except Exception:
            snapshot = {}

    fmm = fmm_df()
    ds_path = OUTPUTS / 'real_dataset_v4.parquet'
    pred = pred_df()

    dataset_fixtures = set()
    dataset_rows = 0
    if ds_path.exists():
        ds = pd.read_parquet(ds_path, columns=['fixture_id'])
        dataset_rows = int(len(ds))
        dataset_fixtures = set(ds['fixture_id'].astype(str))

    matched_fixtures = set(fmm['fixture_id'].astype(str)) if fmm is not None else set()
    pmxt_markets = 0
    matched_markets = 0
    if fmm is not None:
        matched_markets = int(fmm['condition_id'].nunique())
        pmxt_markets = sum(1 for cid in fmm['condition_id'].astype(str) if _condition_price_path(cid).exists())

    by_date = []
    if fmm is not None and dataset_fixtures:
        tmp = fmm.copy()
        tmp['fixture_id'] = tmp['fixture_id'].astype(str)
        tmp['used_in_dataset'] = tmp['fixture_id'].isin(dataset_fixtures)
        by_date = (tmp.groupby('event_date').agg(
            matched_fixtures=('fixture_id', 'nunique'),
            used_fixtures=('used_in_dataset', lambda s: int(tmp.loc[s.index, 'fixture_id'][s].nunique())),
            markets=('condition_id', 'nunique'),
        ).reset_index().sort_values('event_date').to_dict(orient='records'))

    return jsonify({
        'local_hyper_message_files': len(msg_files) or int(snapshot.get('local_hyper_message_files', 0)),
        'local_hyper_fixture_dirs_with_messages': len(fixture_dirs) or int(snapshot.get('local_hyper_fixture_dirs_with_messages', 0)),
        'matched_polymarket_fixtures': len(matched_fixtures),
        'matched_polymarket_markets': matched_markets,
        'pmxt_market_files_available': int(pmxt_markets),
        'dataset_rows': dataset_rows,
        'dataset_fixtures': len(dataset_fixtures),
        'test_prediction_rows': int(len(pred)) if pred is not None else 0,
        'test_prediction_fixtures': int(pred['fixture_id'].nunique()) if pred is not None else 0,
        'matched_not_used_fixtures': len(matched_fixtures - dataset_fixtures),
        'hyper_to_dataset_fixture_use_rate': round(len(dataset_fixtures) / max(len(fixture_dirs), 1), 4),
        'matched_to_dataset_fixture_use_rate': round(len(dataset_fixtures) / max(len(matched_fixtures), 1), 4),
        'by_date': by_date,
    })


# ── main page ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── basic APIs ─────────────────────────────────────────────────────────────────
@app.route('/api/metrics')
def api_metrics():
    return jsonify(load_metrics())

@app.route('/api/continuous_metrics')
def api_continuous_metrics():
    return jsonify(load_json_output('continuous_pricing_metrics.json', {}))

@app.route('/api/lead_pricing_metrics')
def api_lead_pricing_metrics():
    return jsonify(load_json_output('lead_pricing_metrics.json', {}))

@app.route('/api/summary')
def api_summary():
    return jsonify(load_summary())

@app.route('/api/model_comparison')
def api_model_comparison():
    m = load_metrics()
    return jsonify([{
        'model':     x['model'],
        'f1':        x.get('f1', 0),
        'precision': x.get('precision', 0),
        'recall':    x.get('recall', 0),
        'roc_auc':   x.get('roc_auc', 0),
        'threshold': x.get('threshold', 0.5),
    } for x in m.get('models', [])])


@app.route('/api/markov_overview')
def api_markov_overview():
    metrics = load_json_output('markov_xt_metrics.json', {})
    heatmap = load_json_output('markov_event_heatmap.json', [])
    paths = load_json_output('markov_top_paths.json', [])
    return jsonify({
        'metrics': metrics,
        'heatmap': heatmap,
        'top_paths': paths[:30],
    })

@app.route('/api/feature_importance')
def api_feature_importance():
    m = load_metrics()
    for mdl in sorted(m.get('models', []), key=lambda x: x.get('f1', 0), reverse=True):
        if 'feature_importance' in mdl:
            raw = mdl['feature_importance']
            maxv = max(abs(v) for v in raw.values()) or 1
            return jsonify([{'feature': k, 'importance': abs(v)/maxv}
                            for k, v in sorted(raw.items(), key=lambda x: abs(x[1]), reverse=True)][:15])
    return jsonify([])


# ── fixture list ───────────────────────────────────────────────────────────────
@app.route('/api/fixture_list')
def api_fixture_list():
    df = pred_df()
    if df is None:
        return jsonify([])
    fmm = fmm_df()

    result = []
    for fid, grp in df.groupby('fixture_id'):
        grp = grp.sort_values('timestamp')
        meta = {}
        if fmm is not None:
            row = fmm[fmm['fixture_id'] == fid]
            if not row.empty:
                meta = row.iloc[0][['home', 'away', 'event_date']].to_dict()
        markets = fixture_markets(fid)
        result.append({
            'fixture_id':  fid,
            'home':        meta.get('home', '?'),
            'away':        meta.get('away', '?'),
            'event_date':  str(meta.get('event_date', '')),
            'markets':     len(markets),
            'pmxt_markets': sum(1 for m in markets if m['has_pmxt']),
            'n_events':    int(len(grp)),
            'vol_rate':    round(float(grp['high_volatility'].mean()), 3),
            'price_min':   round(float(grp['mid_price'].min()), 3),
            'price_max':   round(float(grp['mid_price'].max()), 3),
            'price_range': round(float(grp['mid_price'].max() - grp['mid_price'].min()), 3),
            'best_pred':   round(float(grp['volatility_prob'].max()), 3),
        })
    return jsonify(sorted(result, key=lambda x: x['price_range'], reverse=True))


# ── per-fixture timeline (rich) ────────────────────────────────────────────────
@app.route('/api/fixture_timeline/<fixture_id>')
def api_fixture_timeline(fixture_id):
    df = pred_df()
    if df is None:
        return jsonify({'error': 'no data'})

    sub = df[df['fixture_id'] == str(fixture_id)].copy()
    if sub.empty:
        return jsonify({'error': f'fixture {fixture_id} not found'})

    sub = sub.sort_values('timestamp')
    markets = fixture_markets(fixture_id)
    requested_cid = request.args.get('condition_id')
    if requested_cid:
        active_market = next((m for m in markets if m['condition_id'] == requested_cid), None)
    else:
        active_market = next((m for m in markets if m['has_pmxt']), markets[0] if markets else None)

    # 1-minute price buckets (smooth price line)
    sub['minute_bucket'] = sub['timestamp'].dt.floor('1min')
    price_agg = {
        'price': ('mid_price', 'mean'),
        'vol_pred': ('volatility_prob', 'mean'),
        'max_pred': ('volatility_prob', 'max'),   # peak prediction within the minute
        'actual_vol': ('high_volatility', 'mean'),
        'n': ('mid_price', 'count'),
    }
    continuous_cols = {
        'pred_abs_move': ('predicted_abs_move_120s', 'mean'),
        'pred_abs_peak': ('predicted_abs_move_120s', 'max'),
        'actual_abs_move': ('max_abs_move_120s', 'mean'),
        'actual_abs_peak': ('max_abs_move_120s', 'max'),
        'pred_price_change': ('predicted_price_change_1m', 'mean'),
        'pred_price_after': ('predicted_price_after_1m', 'last'),
        'continuous_risk': ('continuous_risk_score', 'max'),
        'markov_v2_abs': ('markov_v2_value_abs_move', 'max'),
        'markov_v2_path': ('markov_v2_path_abs_score', 'max'),
        'lead30_actual_abs_move': ('lead30_max_abs_move_120s', 'mean'),
        'lead30_actual_abs_peak': ('lead30_max_abs_move_120s', 'max'),
        'lead30_pred_abs_move': ('lead30_pred_abs_move_120s', 'mean'),
        'lead30_pred_abs_peak': ('lead30_pred_abs_move_120s', 'max'),
        'lead30_signed_move': ('lead30_pred_signed_move_120s', 'mean'),
        'lead30_pred_price_after': ('lead30_pred_price_after_150s', 'last'),
        'lead30_intercept_score': ('lead30_intercept_score', 'max'),
    }
    for out_col, spec in continuous_cols.items():
        if spec[0] in sub.columns:
            price_agg[out_col] = spec
    price_line = sub.groupby('minute_bucket').agg(**price_agg).reset_index()
    # Use RFC-3339 'T' separator so JS Date() parses correctly
    price_line['t'] = price_line['minute_bucket'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Higher-resolution prediction path: 10-second buckets keep model spikes visible.
    sub['ten_sec_bucket'] = sub['timestamp'].dt.floor('10s')
    pred_agg = {
        'price': ('mid_price', 'last'),
        'pred_mean': ('volatility_prob', 'mean'),
        'pred_max': ('volatility_prob', 'max'),
        'actual_vol': ('high_volatility', 'mean'),
        'event_count': ('incident_name', 'count'),
    }
    for out_col, spec in continuous_cols.items():
        if spec[0] in sub.columns:
            pred_agg[out_col] = spec
    pred_line = sub.groupby('ten_sec_bucket').agg(**pred_agg).reset_index()
    pred_line = add_pricing_signal_columns(pred_line)
    pred_line['t'] = pred_line['ten_sec_bucket'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    price_line = add_pricing_signal_columns(price_line)

    # Event markers (non-Timer/Period events only, sample to max 300)
    events = sub[~sub['incident_name'].isin({'Timer', 'Period',
        'Player Completed Passes (Raw)', 'Player Passes',
        'Player Minutes Played', 'Player Successes Tackles Percentage'})].copy()
    # Keep high-xt events always, sample rest
    hi_events = events[events['xt_weight'] >= 2.0]
    lo_events = events[events['xt_weight'] < 2.0].sample(
        min(100, len(events[events['xt_weight'] < 2.0])), random_state=42
    ) if len(events[events['xt_weight'] < 2.0]) > 0 else pd.DataFrame()
    markers_df = pd.concat([hi_events, lo_events]).sort_values('timestamp')

    markers = [{
        't':          r.timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'incident':   r.incident_name,
        'xt_weight':  float(r.xt_weight) if pd.notnull(r.xt_weight) else 0.0,
        'price':      float(r.mid_price),
        'vol_pred':   float(r.volatility_prob),
        'actual_vol': int(r.high_volatility),
        'max_move':   float(r.max_abs_move_120s) if pd.notnull(r.max_abs_move_120s) else 0.0,
        'pred_abs_move': float(getattr(r, 'predicted_abs_move_120s', 0.0)) if pd.notnull(getattr(r, 'predicted_abs_move_120s', 0.0)) else 0.0,
        'pred_price_change': float(getattr(r, 'predicted_price_change_1m', 0.0)) if pd.notnull(getattr(r, 'predicted_price_change_1m', 0.0)) else 0.0,
        'continuous_risk': float(getattr(r, 'continuous_risk_score', 0.0)) if pd.notnull(getattr(r, 'continuous_risk_score', 0.0)) else 0.0,
        'lead30_pred_abs_move': float(getattr(r, 'lead30_pred_abs_move_120s', 0.0)) if pd.notnull(getattr(r, 'lead30_pred_abs_move_120s', 0.0)) else 0.0,
        'lead30_intercept_score': float(getattr(r, 'lead30_intercept_score', 0.0)) if pd.notnull(getattr(r, 'lead30_intercept_score', 0.0)) else 0.0,
        'score_diff': int(r.score_diff) if pd.notnull(r.score_diff) else 0,
        'minutes_remaining': float(r.minutes_remaining) if pd.notnull(r.minutes_remaining) else 0.0,
    } for r in markers_df.itertuples()]

    # Raw local PMXT orderbook path, if available for the selected market.
    pmxt_line = []
    pmxt_meta = {}
    if active_market and active_market.get('has_pmxt'):
        start = sub['timestamp'].min() - pd.Timedelta(minutes=10)
        end = sub['timestamp'].max() + pd.Timedelta(minutes=10)
        pmxt_df, pmxt_meta = load_pmxt_prices(
            active_market['condition_id'],
            start, end,
            target_mean=float(sub['mid_price'].mean()),
        )
        pmxt_line = [{
            't': row.t,
            'mid': _json_safe(row.mid),
            'bid': _json_safe(row.best_bid),
            'ask': _json_safe(row.best_ask),
            'price': _json_safe(row.price),
            'size': _json_safe(row.size),
        } for row in pmxt_df.itertuples()]

    # Price jumps from the displayed orderbook path. These are visual diagnostics,
    # not labels, because labels are still built from the CLOB candle series.
    jump_events = []
    if pmxt_line:
        pm = pd.DataFrame(pmxt_line)
        pm['mid'] = pd.to_numeric(pm['mid'], errors='coerce')
        pm['prev_mid'] = pm['mid'].shift()
        pm['move'] = (pm['mid'] - pm['prev_mid']).abs()
        jumps = pm[pm['move'] >= 0.03].sort_values('move', ascending=False).head(20)
        jump_events = [{
            't': str(r.t),
            'mid': float(r.mid),
            'move': float(r.move),
        } for r in jumps.itertuples()]

    # Advance warning analysis: for each actual big move, did model fire early?
    big_moves = sub[sub['max_abs_move_120s'] > 0.05].copy()
    advance_data = []
    if len(big_moves) > 0:
        big_moves_sorted = big_moves.sort_values('timestamp')
        # Group consecutive big-move events (within 2 min) into single "events"
        move_events = []
        last_t = None
        for _, r in big_moves_sorted.iterrows():
            if last_t is None or (r['timestamp'] - last_t).total_seconds() > 120:
                move_events.append({'t': r['timestamp'], 'move': r['max_abs_move_120s']})
            last_t = r['timestamp']

        for mv in move_events[:20]:
            t_move = mv['t']
            # Check model prediction 30s, 60s, 120s before the move
            for lead_s in [15, 30, 60, 90, 120]:
                t_before = t_move - pd.Timedelta(seconds=lead_s)
                window = sub[(sub['timestamp'] >= t_before) &
                             (sub['timestamp'] < t_move - pd.Timedelta(seconds=lead_s//2))]
                if not window.empty:
                    advance_data.append({
                        'lead_seconds': lead_s,
                        'pred_prob':    float(window['volatility_prob'].mean()),
                        'move_size':    float(mv['move']),
                        't':            str(t_move),
                    })

    # Metadata
    fmm = fmm_df()
    meta = {}
    if fmm is not None:
        row = fmm[fmm['fixture_id'] == str(fixture_id)]
        if not row.empty:
            meta = row.iloc[0][['home', 'away', 'event_date', 'condition_id']].to_dict()

    return jsonify({
        'fixture_id':   fixture_id,
        'home':         str(meta.get('home', 'Home')),
        'away':         str(meta.get('away', 'Away')),
        'event_date':   str(meta.get('event_date', '')),
        'condition_id': str(meta.get('condition_id', '')),
        'n_events':     int(len(sub)),
        'vol_rate':     round(float(sub['high_volatility'].mean()), 3),
        'pricing_threshold': pricing_threshold(),
        'markets':      markets,
        'active_market': active_market or {},
        'price_line':   price_line.drop(columns=['minute_bucket']).replace({np.nan: None}).to_dict(orient='records'),
        'pred_line':    pred_line.drop(columns=['ten_sec_bucket']).replace({np.nan: None}).to_dict(orient='records'),
        'pmxt_line':    pmxt_line,
        'pmxt_meta':    pmxt_meta,
        'jump_events':  jump_events,
        'markers':      markers,
        'advance_data': advance_data,
    })


@app.route('/api/fixture_deepdive/<fixture_id>')
def api_fixture_deepdive(fixture_id):
    df = pred_df()
    if df is None:
        return jsonify({'error': 'no data'})

    sub = df[df['fixture_id'] == str(fixture_id)].copy()
    if sub.empty:
        return jsonify({'error': f'fixture {fixture_id} not found'})

    sub = sub.sort_values('timestamp').reset_index(drop=True)
    threshold = pricing_threshold()

    # Multi-signal state line: model risk, price pressure, event pressure, score context.
    sub['minute_bucket'] = sub['timestamp'].dt.floor('1min')
    agg_spec = {
        'price': ('mid_price', 'last'),
        'pred_peak': ('volatility_prob', 'max'),
        'pred_mean': ('volatility_prob', 'mean'),
        'actual_vol': ('high_volatility', 'mean'),
        'score_diff': ('score_diff', 'last'),
        'total_goals': ('total_goals', 'last'),
    }
    optional_cols = {
        'event_density_60s': ('event_density_60s', 'max'),
        'xt_pressure': ('xt_sum_60s', 'max'),
        'price_realized_vol': ('price_realized_vol', 'max'),
        'price_velocity': ('price_velocity', 'mean'),
        'high_impact_60s': ('high_impact_60s', 'max'),
        'markov_path_score': ('markov_path_score', 'max'),
        'markov_path_confidence': ('markov_path_confidence', 'max'),
        'markov_event_move': ('markov_event_move', 'max'),
        'pred_abs_move': ('predicted_abs_move_120s', 'mean'),
        'pred_abs_peak': ('predicted_abs_move_120s', 'max'),
        'actual_abs_move': ('max_abs_move_120s', 'mean'),
        'actual_abs_peak': ('max_abs_move_120s', 'max'),
        'pred_price_change': ('predicted_price_change_1m', 'mean'),
        'pred_price_after': ('predicted_price_after_1m', 'last'),
        'continuous_risk': ('continuous_risk_score', 'max'),
        'markov_v2_abs': ('markov_v2_value_abs_move', 'max'),
        'markov_v2_event_abs': ('markov_v2_event_abs_move', 'max'),
        'markov_v2_path': ('markov_v2_path_abs_score', 'max'),
        'lead30_actual_abs_move': ('lead30_max_abs_move_120s', 'mean'),
        'lead30_actual_abs_peak': ('lead30_max_abs_move_120s', 'max'),
        'lead30_pred_abs_move': ('lead30_pred_abs_move_120s', 'mean'),
        'lead30_pred_abs_peak': ('lead30_pred_abs_move_120s', 'max'),
        'lead30_signed_move': ('lead30_pred_signed_move_120s', 'mean'),
        'lead30_pred_price_after': ('lead30_pred_price_after_150s', 'last'),
        'lead30_intercept_score': ('lead30_intercept_score', 'max'),
    }
    for out_col, spec in optional_cols.items():
        if spec[0] in sub.columns:
            agg_spec[out_col] = spec
    state = sub.groupby('minute_bucket').agg(**agg_spec).reset_index()
    state = add_pricing_signal_columns(state)
    state['t'] = state['minute_bucket'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    state_line = state.drop(columns=['minute_bucket']).replace({np.nan: None}).to_dict(orient='records')

    # Consecutive model-fire windows. These are the most actionable units for a trader.
    sub = add_pricing_signal_columns(sub.rename(columns={'mid_price': 'price'}))
    fired = sub[sub['pricing_alert'] == 1].copy()
    windows = []
    if not fired.empty:
        gaps = fired['timestamp'].diff().dt.total_seconds().fillna(0)
        fired['window_id'] = (gaps > 90).cumsum()
        for _, g in fired.groupby('window_id'):
            start, end = g['timestamp'].min(), g['timestamp'].max()
            context = sub[(sub['timestamp'] >= start - pd.Timedelta(seconds=30)) &
                          (sub['timestamp'] <= end + pd.Timedelta(seconds=120))]
            incidents = (context[~context['incident_name'].isin({'Timer', 'Period'})]
                         ['incident_name'].value_counts().head(4))
            markov_context = {}
            if 'markov_path_score' in context.columns and not context.empty:
                peak_idx = context['markov_path_score'].idxmax()
                peak_markov = context.loc[peak_idx]
                markov_context = {
                    'markov_peak': round(float(context['markov_path_score'].max()), 4),
                    'markov_confidence': round(float(context.get('markov_path_confidence', pd.Series([0])).max()), 4),
                    'markov_event_group': str(peak_markov.get('event_group', '')),
                    'markov_jump_level': str(peak_markov.get('markov_jump_level', '')),
                }
            continuous_context = {}
            if 'predicted_abs_move_120s' in context.columns and not context.empty:
                continuous_context = {
                    'pred_abs_peak': round(float(context['predicted_abs_move_120s'].max()), 4),
                    'pred_price_change': round(float(context.get('predicted_price_change_1m', pd.Series([0])).mean()), 4),
                    'continuous_risk': round(float(context.get('continuous_risk_score', pd.Series([0])).max()), 4),
                }
            if 'lead30_pred_abs_move_120s' in context.columns and not context.empty:
                continuous_context.update({
                    'lead30_pred_abs_peak': round(float(context['lead30_pred_abs_move_120s'].max()), 4),
                    'lead30_intercept_score': round(float(context.get('lead30_intercept_score', pd.Series([0])).max()), 4),
                })
            actual = int(context['high_volatility'].max()) if not context.empty else 0
            windows.append({
                'start': _iso(start),
                'end': _iso(end),
                'duration_s': round(float((end - start).total_seconds()), 1),
                'peak_prob': round(float(g['pricing_signal'].max()), 4),
                'avg_prob': round(float(g['pricing_signal'].mean()), 4),
                'price_start': round(float(g['price'].iloc[0]), 4),
                'price_end': round(float(g['price'].iloc[-1]), 4),
                'price_change': round(float(g['price'].iloc[-1] - g['price'].iloc[0]), 4),
                'max_future_move': round(float(context['max_abs_move_120s'].max()), 4) if not context.empty else 0.0,
                'actual_high_vol': actual,
                'label': 'hit' if actual else 'false_alarm',
                'top_incidents': [{'incident': k, 'n': int(v)} for k, v in incidents.items()],
                **markov_context,
                **continuous_context,
            })
    windows = sorted(windows, key=lambda x: x['peak_prob'], reverse=True)[:25]

    # Price jump alignment: for large PMXT moves, was the model already elevated?
    markets = fixture_markets(fixture_id)
    requested_cid = request.args.get('condition_id')
    active_market = (next((m for m in markets if m['condition_id'] == requested_cid), None)
                     if requested_cid else next((m for m in markets if m['has_pmxt']), None))
    jump_alignment = []
    if active_market and active_market.get('has_pmxt'):
        pmxt, _ = load_pmxt_prices(
            active_market['condition_id'],
            sub['timestamp'].min() - pd.Timedelta(minutes=10),
            sub['timestamp'].max() + pd.Timedelta(minutes=10),
            target_mean=float(sub['price'].mean()),
            max_points=5000,
        )
        if not pmxt.empty:
            pmxt = pmxt.sort_values('timestamp').copy()
            pmxt['prev_mid'] = pmxt['mid'].shift()
            pmxt['move'] = (pmxt['mid'] - pmxt['prev_mid']).abs()
            jumps = pmxt[pmxt['move'] >= 0.03].copy()
            jumps = jumps.sort_values('timestamp').head(80)
            for r in jumps.itertuples():
                t = r.timestamp
                prev = sub[(sub['timestamp'] >= t - pd.Timedelta(seconds=180)) &
                           (sub['timestamp'] <= t)]
                if prev.empty:
                    continue
                prev = add_pricing_signal_columns(prev.rename(columns={'mid_price': 'price'}))
                peak_idx = prev['pricing_signal'].idxmax()
                peak = prev.loc[peak_idx]
                key_events = prev[prev['xt_weight'] >= 2.0]
                event_name = key_events['incident_name'].iloc[-1] if not key_events.empty else ''
                event_lead = None
                if not key_events.empty:
                    event_lead = round(float((t - key_events['timestamp'].iloc[-1]).total_seconds()), 1)
                jump_alignment.append({
                    't': _iso(t),
                    'mid': round(float(r.mid), 4),
                    'move': round(float(r.move), 4),
                    'prior_peak_prob': round(float(peak['pricing_signal']), 4),
                    'model_lead_s': round(float((t - peak['timestamp']).total_seconds()), 1),
                    'was_alerted': bool(int(peak.get('pricing_alert', 0)) == 1),
                    'nearest_event': str(event_name),
                    'event_lead_s': event_lead,
                })
    jump_alignment = sorted(jump_alignment, key=lambda x: x['move'], reverse=True)[:30]

    # Incident impact table within the selected test fixture.
    incident_stats = (sub.groupby('incident_name').agg(
        n=('high_volatility', 'count'),
        avg_prob=('volatility_prob', 'mean'),
        peak_prob=('volatility_prob', 'max'),
        vol_rate=('high_volatility', 'mean'),
        avg_move=('max_abs_move_120s', 'mean'),
        avg_xt=('xt_weight', 'mean'),
        markov_score=('markov_path_score', 'mean') if 'markov_path_score' in sub.columns else ('xt_weight', 'mean'),
        markov_move=('markov_event_move', 'mean') if 'markov_event_move' in sub.columns else ('max_abs_move_120s', 'mean'),
        pred_abs_move=('predicted_abs_move_120s', 'mean') if 'predicted_abs_move_120s' in sub.columns else ('max_abs_move_120s', 'mean'),
        continuous_risk=('continuous_risk_score', 'mean') if 'continuous_risk_score' in sub.columns else ('volatility_prob', 'mean'),
        lead30_pred_abs=('lead30_pred_abs_move_120s', 'mean') if 'lead30_pred_abs_move_120s' in sub.columns else ('max_abs_move_120s', 'mean'),
        lead30_score=('lead30_intercept_score', 'mean') if 'lead30_intercept_score' in sub.columns else ('volatility_prob', 'mean'),
    ).reset_index())
    incident_stats = (incident_stats[incident_stats['n'] >= 20]
                      .sort_values(['avg_prob', 'vol_rate'], ascending=False)
                      .head(24)
                      .round(4)
                      .to_dict(orient='records'))

    return jsonify({
        'fixture_id': str(fixture_id),
        'threshold': threshold,
        'state_line': state_line,
        'signal_windows': windows,
        'jump_alignment': jump_alignment,
        'incident_impact': incident_stats,
        'active_market': active_market or {},
    })


# ── advance warning aggregate analysis ────────────────────────────────────────
@app.route('/api/advance_warning')
def api_advance_warning():
    """Aggregate: for big price moves, what's the avg model prob N seconds before?"""
    df = pred_df()
    if df is None:
        return jsonify([])

    df = df.sort_values(['fixture_id', 'timestamp'])
    result = []
    for fid, grp in df.groupby('fixture_id'):
        grp = grp.sort_values('timestamp').reset_index(drop=True)
        ts_arr   = grp['timestamp'].values
        prob_arr = grp['volatility_prob'].values
        move_arr = grp['max_abs_move_120s'].values

        # Find events with large actual moves
        big = np.where(move_arr > 0.08)[0]
        for idx in big:
            t_move = ts_arr[idx]
            move   = float(move_arr[idx])
            for lead_s in [15, 30, 60, 90, 120]:
                t_before = t_move - np.timedelta64(lead_s, 's')
                mask = (ts_arr >= t_before) & (ts_arr < t_move)
                if mask.any():
                    result.append({
                        'lead_seconds': lead_s,
                        'avg_pred':     float(prob_arr[mask].mean()),
                        'move_size':    move,
                    })

    if not result:
        return jsonify([])

    df_res = pd.DataFrame(result)
    agg = df_res.groupby('lead_seconds').agg(
        avg_pred=('avg_pred', 'mean'),
        median_pred=('avg_pred', 'median'),
        n=('avg_pred', 'count'),
    ).reset_index()
    return jsonify(agg.to_dict(orient='records'))


# ── per-incident type stats ────────────────────────────────────────────────────
@app.route('/api/incident_stats')
def api_incident_stats():
    df = pred_df()
    if df is None:
        return jsonify([])
    stats = (df.groupby('incident_name').agg(
        vol_rate=('high_volatility', 'mean'),
        pred_rate=('predicted_volatile', 'mean'),
        avg_pred_prob=('volatility_prob', 'mean'),
        n=('high_volatility', 'count'),
        avg_xt=('xt_weight', 'mean'),
    ).reset_index()
    .query('n >= 50')
    .sort_values('vol_rate', ascending=False)
    .head(20))
    return jsonify(stats.round(4).to_dict(orient='records'))


# ── volatility timeline (aggregate) ───────────────────────────────────────────
@app.route('/api/volatility_timeline')
def api_volatility_timeline():
    df = pred_df()
    if df is None:
        return jsonify([])
    fid = request.args.get('fixture_id')
    if fid:
        df = df[df['fixture_id'] == fid]
    df = df.sort_values('timestamp')
    df['minute_bucket'] = df['timestamp'].dt.floor('1min')
    agg = df.groupby('minute_bucket').agg(
        avg_pred=('volatility_prob', 'mean'),
        actual_rate=('high_volatility', 'mean'),
        avg_price=('mid_price', 'mean'),
        n=('volatility_prob', 'count'),
    ).reset_index()
    agg['minute_bucket'] = agg['minute_bucket'].astype(str)
    return jsonify(agg.head(400).to_dict(orient='records'))


# ── confusion matrix data ──────────────────────────────────────────────────────
@app.route('/api/confusion')
def api_confusion():
    df = pred_df()
    if df is None:
        return jsonify({})
    tp = int(((df['predicted_volatile']==1) & (df['high_volatility']==1)).sum())
    fp = int(((df['predicted_volatile']==1) & (df['high_volatility']==0)).sum())
    fn = int(((df['predicted_volatile']==0) & (df['high_volatility']==1)).sum())
    tn = int(((df['predicted_volatile']==0) & (df['high_volatility']==0)).sum())
    return jsonify({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})


# ── price distribution (calibration check) ─────────────────────────────────────
@app.route('/api/calibration')
def api_calibration():
    df = pred_df()
    if df is None:
        return jsonify([])
    df = df.copy()
    df['prob_bin'] = pd.cut(df['volatility_prob'], bins=10, labels=False)
    cal = df.groupby('prob_bin').agg(
        avg_prob=('volatility_prob', 'mean'),
        actual_rate=('high_volatility', 'mean'),
        n=('high_volatility', 'count'),
    ).reset_index().dropna()
    return jsonify(cal.to_dict(orient='records'))


if __name__ == '__main__':
    print('Probly Sports Pricing Dashboard')
    port = int(os.environ.get('PORT', '5001'))
    print(f'Open: http://localhost:{port}')
    app.run(debug=True, port=port, host='0.0.0.0')
