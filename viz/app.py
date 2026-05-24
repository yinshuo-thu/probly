"""
Probly Sports Pricing — Rich Interactive Dashboard

Run: python viz/app.py
Then open: http://localhost:5001
"""

import sys, json
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from flask import Flask, render_template, jsonify, request

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

BASE       = Path(__file__).parent.parent
OUTPUTS    = BASE / 'outputs'
CLOB_API   = 'https://clob.polymarket.com'
FMM_FILE   = BASE / 'data/polymarket/fixture_market_matches.parquet'

app = Flask(__name__)

# ── cached data ────────────────────────────────────────────────────────────────
_pred_df   = None
_fmm_df    = None

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

def load_metrics():
    p = OUTPUTS / 'metrics_history.json'
    return json.load(open(p)) if p.exists() else {}

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


# ── main page ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── basic APIs ─────────────────────────────────────────────────────────────────
@app.route('/api/metrics')
def api_metrics():
    return jsonify(load_metrics())

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
        result.append({
            'fixture_id':  fid,
            'home':        meta.get('home', '?'),
            'away':        meta.get('away', '?'),
            'event_date':  str(meta.get('event_date', '')),
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

    # 1-minute price buckets (smooth price line)
    sub['minute_bucket'] = sub['timestamp'].dt.floor('1min')
    price_line = sub.groupby('minute_bucket').agg(
        price=('mid_price', 'mean'),
        vol_pred=('volatility_prob', 'mean'),
        actual_vol=('high_volatility', 'mean'),
        n=('mid_price', 'count'),
    ).reset_index()
    price_line['t'] = price_line['minute_bucket'].astype(str)

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
        't':          str(r.timestamp),
        'incident':   r.incident_name,
        'xt_weight':  float(r.xt_weight) if pd.notnull(r.xt_weight) else 0.0,
        'price':      float(r.mid_price),
        'vol_pred':   float(r.volatility_prob),
        'actual_vol': int(r.high_volatility),
        'max_move':   float(r.max_abs_move_120s) if pd.notnull(r.max_abs_move_120s) else 0.0,
        'score_diff': int(r.score_diff) if pd.notnull(r.score_diff) else 0,
        'minutes_remaining': float(r.minutes_remaining) if pd.notnull(r.minutes_remaining) else 0.0,
    } for r in markers_df.itertuples()]

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
        'price_line':   price_line[['t','price','vol_pred','actual_vol','n']].to_dict(orient='records'),
        'markers':      markers,
        'advance_data': advance_data,
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
    print('Open: http://localhost:5001')
    app.run(debug=True, port=5001, host='0.0.0.0')
