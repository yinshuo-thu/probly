# Probly Sports Pricing

> **体育事件实时定价研究 — LSports × Polymarket 跨市场价格发现**

Predict Polymarket high-volatility windows **before** they happen using LSports real-time event stream data.  
Core: xT (Expected Threat) framework + rolling event density + price velocity + ML classifiers.

---

## Project Goal

Polymarket prediction markets show sharp price swings triggered by in-game events (goals, red cards, VAR reviews). We detect **precursors** from the LSports event stream to:
- Predict whether the market price will move >3 cents in the next 2 minutes
- Rank events by expected threat weight (xT framework)
- Quantify advance-warning capability: how well does the model predict N seconds *before* a big move?
- Provide real-time signal for trading decisions

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yinshuo-thu/probly.git
cd probly

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build aligned dataset (v4: price velocity + causal label + game state)
python src/build_dataset_v4.py

# 4. Train models (auto-detects v4 vs v3)
python src/train_models.py

# 5. Launch interactive dashboard
python viz/app.py
# → http://localhost:5001
```

## Pipeline Architecture

```
LSports Hyper event stream     Polymarket CLOB API
(messages.parquet per game)    /prices-history (1-min candles)
          ↓                            ↓
   build_dataset_v4.py  ←── fixture_market_matches.parquet
          ↓                   (163 verified condition_ids)
   real_dataset_v4.parquet (163 fixtures, 25+ features)
          │
          ├── price_velocity, price_realized_vol (5-min rolling)
          ├── high_impact events (goal/card/penalty counts)
          ├── game state (score, intensity, period)
          └── two labels: high_volatility + next_60s_high_impact
          ↓
   train_models.py
   LR → RF → XGBoost → LightGBM (threshold-optimized)
          ↓
   outputs/metrics_history.json + test_predictions.parquet
          ↓
   viz/app.py (Flask + Plotly interactive dashboard)
```

## Code Structure

```
probly/
├── src/
│   ├── build_dataset_v4.py      # v4: price velocity + causal label + game state
│   ├── build_dataset_v3.py      # v3: baseline O(N log N) vectorized build
│   ├── train_models.py          # LR/RF/XGBoost/LightGBM (auto-detects v3/v4)
│   ├── expand_matches.py        # Gamma API search for more fixture-market matches
│   ├── match_fixtures.py        # Match LSports fixtures to Polymarket markets
│   └── download_football.py     # Download LSports football data
├── viz/
│   ├── app.py                   # Flask REST API + web server
│   └── templates/index.html     # Plotly dashboard (dark theme, Chinese UI)
├── data/
│   └── polymarket/
│       └── fixture_market_matches.parquet  # 163 verified fixture-market pairs
└── outputs/
    ├── real_dataset_v4.parquet  # 163-fixture dataset (gitignored)
    ├── test_predictions.parquet # Model predictions (gitignored)
    └── metrics_history.json     # F1/AUC per model
```

## Model Results

### v4 dataset (1.33M events, 54 fixtures, 30 features)

| Model | F1 | Precision | Recall | ROC AUC | Threshold |
|-------|-----|-----------|--------|---------|-----------|
| Logistic Regression | 53.0% | 50.0% | 56.4% | 0.730 | 0.48 |
| **Random Forest** | **58.6%** | **60.4%** | **57.0%** | **0.789** | **0.52** |
| XGBoost | 54.6% | 53.5% | 55.9% | 0.747 | — |
| LightGBM | 55.0% | 48.8% | 62.9% | 0.746 | — |

### vs v3 baseline (17 features)

| Metric | v3 | v4 | Improvement |
|--------|----|----|-------------|
| F1 (RF) | 48.3% | **58.6%** | **+10.3pp** |
| AUC (RF) | 0.583 | **0.789** | **+0.206** |
| Precision | 33.6% | **60.4%** | **+26.8pp** |

Key improvements: `price_velocity`, `price_realized_vol`, `xt_sum_60s/300s`, `high_impact_60s`, `game_intensity`

> **v3 label ceiling**: The 120-second price window labels routine events as volatile from coincidental overlap (~28% base rate). v4 includes `next_60s_high_impact` (causal label) for cleaner training signal.

## Features (v4 — 25 total)

### Core event features
| Feature | Description |
|---------|-------------|
| `xt_weight` | xT value for current incident (0=Timer, 5=Goal) |
| `is_high_impact` | 1 if Goal/RedCard/Penalty/VAR |
| `risk_event_count_30/60s` | High-risk events in past 30/60s |
| `event_density_30/60/120/300s` | Total events in rolling windows |
| `xt_sum_60s`, `xt_sum_300s` | Cumulative xT in past 60s / 5min |
| `high_impact_60s`, `high_impact_300s` | Goal/card/penalty counts in past windows |

### Market features (new in v4)
| Feature | Description |
|---------|-------------|
| `mid_price` | Current Polymarket price |
| `price_velocity` | Price slope (5-min linear fit) |
| `price_realized_vol` | Price std over past 5 minutes |
| `price_change_1m` | Price delta in last 1-min candle |
| `price_extremeness` | `|price - 0.5| × 2` |

### Game state features (new in v4)
| Feature | Description |
|---------|-------------|
| `score_diff` | Home minus away goals |
| `total_goals` | Total goals scored |
| `is_leading` / `is_drawing` | Binary game state |
| `game_intensity` | `total_goals × (min_elapsed/90)` |
| `minutes_remaining` | Time left (clipped at 0) |
| `period_id` | Match period |

### Cross-features
| Feature | Description |
|---------|-------------|
| `xt_x_conf` | `xt_weight × confidence_grade` |
| `pressure_score` | `xt_sum_60s × confidence_grade` |
| `risk_density_ratio` | `risk_events / (total_events + 1)` |

## Labels

```
# Price-based (primary)
high_volatility = 1 if max(|price_t - price_t0|) > 0.03
                       for t in [t0, t0+120s]

# Causal (v4, less noisy)
next_60s_high_impact = 1 if any(Goal/RedCard/Penalty/VAR) in [t0, t0+60s]
```

## Dashboard

The interactive dashboard at `http://localhost:5001` shows:
- **Live fixture selector**: 17 test fixtures ranked by price range
- **Main timeline chart**: dual-axis — Polymarket odds (orange) vs model prediction probability (blue area), with emoji event markers (⚽ goals, 🟥 red cards, ⚡ penalties)
- **Advance warning analysis**: how far in advance the model fires before big price moves
- **Incident bubble chart**: each event type plotted by actual vol rate vs predicted probability
- **Model comparison table**: F1/AUC/threshold for all 4 models
- **Calibration curve**: model probability vs actual volatility rate

---

*Research project. LSports Hyper data + Polymarket CLOB API.*
