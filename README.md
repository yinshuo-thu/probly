# Probly Sports Pricing

> **体育事件实时定价研究 — LSports × Polymarket 跨市场价格发现**

Predict Polymarket high-volatility windows **before** they happen using LSports real-time event stream data.  
Core: xT (Expected Threat) framework + rolling event density + ML classifiers.

---

## Project Goal

Polymarket prediction markets show sharp price swings triggered by in-game events (goals, red cards, VAR reviews). We detect **precursors** from the LSports event stream to:
- Predict whether the market price will move >3 cents in the next 2 minutes
- Rank events by expected threat weight (xT framework)
- Provide real-time signal for trading decisions

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yinshuo-thu/probly.git
cd probly

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build aligned dataset (LSports events + Polymarket CLOB prices)
python src/build_dataset_v3.py

# 4. Train models
python src/train_models.py

# 5. Launch dashboard
python viz/app.py
# → http://localhost:5001
```

## Pipeline Architecture

```
LSports Hyper event stream   Polymarket CLOB API
(messages.parquet per game)  /prices-history (1-min candles)
          ↓                           ↓
   build_dataset_v3.py  ←── fixture_market_matches.parquet
          ↓                  (163 verified condition_ids)
   real_dataset.parquet (1.33M events, 54 fixtures)
          ↓
   train_models.py
   LR → RF → XGBoost → LightGBM (threshold-optimized)
          ↓
   outputs/metrics_history.json
          ↓
   viz/app.py (Flask + Plotly dashboard)
```

## Code Structure

```
probly/
├── src/
│   ├── build_dataset_v3.py      # LSports + CLOB → aligned dataset
│   ├── train_models.py          # LR/RF/XGBoost/LightGBM training
│   ├── match_fixtures.py        # Match LSports fixtures to Polymarket markets
│   ├── filter_football_markets.py  # Filter relevant Polymarket football markets
│   └── download_football.py     # Download LSports football data
├── viz/
│   ├── app.py                   # Flask REST API + web server
│   └── templates/index.html     # Plotly dashboard (dark theme)
├── data/
│   └── polymarket/
│       └── fixture_market_matches.parquet  # 163 verified fixture-market pairs
└── outputs/
    ├── real_dataset.parquet     # 1.33M aligned events (gitignored)
    ├── test_predictions.parquet # Model predictions (gitignored)
    └── metrics_history.json     # F1/AUC per model
```

## Model Results (Real Data — 1.33M events, 54 fixtures)

| Model | F1 | Precision | Recall | ROC AUC | Threshold |
|-------|-----|-----------|--------|---------|-----------|
| Logistic Regression | 48.2% | 32.0% | 97.2% | 0.595 | 0.38 |
| **Random Forest** | **48.3%** | **33.6%** | **85.3%** | **0.583** | **0.38** |
| XGBoost | 47.1% | 31.2% | 95.6% | 0.562 | 0.12 |
| LightGBM | 46.9% | 31.7% | 89.6% | 0.557 | 0.16 |

**Best model: Random Forest (F1=48.3%)**

> **Note on label ceiling**: The 120-second future price window creates label contamination — routine events (Timer, Player Passes) inherit ~28% volatility rate from coincidental overlap with unrelated market moves. Theoretical F1 ceiling with this label is ~0.55. Higher F1 requires either: (a) shorter window (30s), or (b) causal label ("did a goal occur in next 30s?").

## Features (17 total)

| Feature | Type | Description |
|---------|------|-------------|
| `xt_weight` | float | xT value for current incident type (0=Timer, 5=Goal) |
| `is_high_impact` | binary | 1 if current event is Goal/RedCard/Penalty/VAR |
| `risk_event_count_30/60s` | int | High-risk events in past 30/60 seconds |
| `event_density_30/60/120s` | int | Total events in rolling windows |
| `mid_price` | float | Current Polymarket market price |
| `price_extremeness` | float | `|price - 0.5| × 2` (distance from maximum uncertainty) |
| `minutes_remaining` | float | Time left in match |
| `score_diff` | int | Home minus away goals |
| `confidence_grade` | float | LSports event confidence (0-1) |
| `xt_x_conf` | float | `xt_weight × confidence_grade` interaction |
| `risk_density_ratio` | float | `risk_events / (total_events + 1)` |

## Volatility Label

```
high_volatility = 1 if max(|price_t - price_t0|) > 0.03
                       for t in [t0, t0+120s]
```
Price data: Polymarket CLOB API 1-minute candles via `/prices-history`.

---

*Research project. LSports Hyper data + Polymarket CLOB API.*
