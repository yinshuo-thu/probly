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
    if path.exists():
        df = pd.read_parquet(path)
        # Convert timestamps to string for JSON
        for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
            df[col] = df[col].astype(str)
        # Limit rows for frontend
        return df.head(5000)
    return None


def load_metrics():
    path = OUTPUTS_DIR / "metrics_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def load_summary():
    path = OUTPUTS_DIR / "pipeline_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
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
        return jsonify({"error": "No prediction data found. Run the pipeline first."})

    # Return key columns for visualization
    cols = [c for c in [
        "timestamp", "match_id", "elapsed_time", "score_home", "score_away",
        "event_type", "markov_win_prob", "polymarket_price",
        "predicted_volatility_prob", "high_volatility_label",
        "xt_value", "xt_sum_60s", "event_density_60s",
        "confidence_val", "markov_volatility_score", "markov_price_delta",
        "score_diff", "minutes_remaining",
    ] if c in df.columns]

    return jsonify(df[cols].replace({np.nan: None}).to_dict(orient="records"))


@app.route("/api/feature_importance")
def api_feature_importance():
    path = OUTPUTS_DIR / "models" / "feature_importance_xgb.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        df.columns = ["importance"]
        return jsonify(df.head(15).reset_index().to_dict(orient="records"))
    return jsonify([])


@app.route("/api/volatility_timeline")
def api_volatility_timeline():
    df = load_predictions()
    if df is None:
        return jsonify([])

    # Aggregate by time bucket for chart
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["minute_bucket"] = df["timestamp"].dt.floor("1min")
        agg = df.groupby("minute_bucket").agg(
            avg_pred=("predicted_volatility_prob", "mean"),
            actual_rate=("high_volatility_label", "mean"),
            avg_price=("polymarket_price", "mean") if "polymarket_price" in df.columns else ("predicted_volatility_prob", "mean"),
            n=("predicted_volatility_prob", "count"),
        ).reset_index()
        agg["minute_bucket"] = agg["minute_bucket"].astype(str)
        return jsonify(agg.head(200).to_dict(orient="records"))
    return jsonify([])


if __name__ == "__main__":
    print("Probly Sports Pricing Dashboard")
    print("Open: http://localhost:5001")
    app.run(debug=True, port=5001, host="0.0.0.0")
