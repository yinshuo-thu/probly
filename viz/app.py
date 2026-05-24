"""
Probly Sports Pricing — Interactive Web Dashboard

Run: python viz/app.py
Then open: http://localhost:5001
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, jsonify

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

app = Flask(__name__)


def load_predictions():
    path = OUTPUTS_DIR / "test_predictions.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Normalize column names for dashboard compatibility
    renames = {
        'high_volatility':  'high_volatility_label',
        'mid_price':        'polymarket_price',
        'volatility_prob':  'predicted_volatility_prob',
        'incident_name':    'event_type',
        'fixture_id':       'match_id',
    }
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].astype(str)
    return df.head(5000)


def load_metrics():
    path = OUTPUTS_DIR / "metrics_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_summary():
    path = OUTPUTS_DIR / "pipeline_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Build summary from real dataset stats
    metrics = load_metrics()
    ds_path = OUTPUTS_DIR / "real_dataset.parquet"
    if ds_path.exists():
        df = pd.read_parquet(ds_path)
        return {
            'dataset': 'real_dataset.parquet',
            'total_rows': len(df),
            'fixtures': int(df['fixture_id'].nunique()),
            'vol_rate': round(float(df['high_volatility'].mean()), 4),
            'best_model': metrics.get('best_model', 'N/A'),
            'best_f1': metrics.get('best_f1', 0.0),
        }
    return {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics")
def api_metrics():
    return jsonify(load_metrics())


@app.route("/api/summary")
def api_summary():
    return jsonify(load_summary())


@app.route("/api/predictions")
def api_predictions():
    df = load_predictions()
    if df is None:
        return jsonify({"error": "No prediction data found."})

    cols = [c for c in [
        "timestamp", "match_id", "minutes_remaining", "score_diff",
        "event_type", "polymarket_price",
        "predicted_volatility_prob", "high_volatility_label",
        "xt_weight", "confidence_grade",
    ] if c in df.columns]
    return jsonify(df[cols].replace({np.nan: None}).to_dict(orient="records"))


@app.route("/api/feature_importance")
def api_feature_importance():
    metrics = load_metrics()
    models = metrics.get('models', [])
    # Find RF or best model feature importance
    for m in sorted(models, key=lambda x: x.get('f1', 0), reverse=True):
        if 'feature_importance' in m:
            items = [{'feature': k, 'importance': v}
                     for k, v in m['feature_importance'].items()]
            return jsonify(items[:15])
    return jsonify([])


@app.route("/api/volatility_timeline")
def api_volatility_timeline():
    df = load_predictions()
    if df is None:
        return jsonify([])

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["minute_bucket"] = df["timestamp"].dt.floor("1min")
        agg = df.groupby("minute_bucket").agg(
            avg_pred=("predicted_volatility_prob", "mean"),
            actual_rate=("high_volatility_label", "mean"),
            avg_price=("polymarket_price", "mean"),
            n=("predicted_volatility_prob", "count"),
        ).reset_index()
        agg["minute_bucket"] = agg["minute_bucket"].astype(str)
        return jsonify(agg.head(300).to_dict(orient="records"))
    return jsonify([])


@app.route("/api/model_comparison")
def api_model_comparison():
    metrics = load_metrics()
    models = metrics.get('models', [])
    return jsonify([{
        'model': m['model'],
        'f1':        m.get('f1', 0),
        'precision': m.get('precision', 0),
        'recall':    m.get('recall', 0),
        'roc_auc':   m.get('roc_auc', 0),
    } for m in models])


@app.route("/api/incident_stats")
def api_incident_stats():
    """Return vol rate per incident type for analysis chart."""
    ds_path = OUTPUTS_DIR / "real_dataset.parquet"
    if not ds_path.exists():
        return jsonify([])
    df = pd.read_parquet(ds_path, columns=['incident_name', 'high_volatility'])
    stats = (df.groupby('incident_name')['high_volatility']
               .agg(['mean', 'count'])
               .rename(columns={'mean': 'vol_rate', 'count': 'n'})
               .reset_index()
               .query('n >= 100')
               .sort_values('vol_rate', ascending=False)
               .head(25))
    return jsonify(stats.to_dict(orient='records'))


if __name__ == "__main__":
    print("Probly Sports Pricing Dashboard")
    print("Open: http://localhost:5001")
    app.run(debug=True, port=5001, host="0.0.0.0")
