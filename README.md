# Probly Sports Pricing

LSports Hyper x Polymarket sports-pricing research dashboard.

The project studies whether LSports football event streams can price or
intercept Polymarket sports-market moves before the market has fully adjusted.
It now includes three layers:

- Recall-first high-volatility interception.
- Markov/xT explainability and state-value features.
- Continuous and lead-time price-path forecasting.

Production dashboard:

- https://autoalpha.cn/probly/

## Current Result

The strictest deployed test is the lead-time target:

- At event time `t`, only current and past data are available.
- The target starts after a 30 second gap: `[t+30s, t+150s]`.
- This avoids counting immediate Polymarket reaction as an "advance" signal.

Test set lead-pricing result:

| Mode | Future High-Vol Intervals | Predicted Intercepts | Accurate Intercepts | Missed | False Alarms | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coverage | 110,597 | 168,711 | 83,465 | 27,132 | 85,246 | 49.47% | 75.47% |
| High confidence | 110,597 | 66,002 | 39,724 | 70,873 | 26,278 | 60.19% | 35.92% |

Continuous lead metrics:

- Lead absolute-move Spearman: `0.3963`
- Lead absolute-move top-decile lift: `1.8021`
- Lead signed-move Spearman: `0.2371`

Interpretation:

- The signal is real enough for early-risk coverage and market-maker protection.
- The high-confidence mode is cleaner, but still leaves many missed intervals.
- Directional signed-price forecasting remains the weakest part and needs more
  Polymarket orderbook/trade-level features.

## Dashboard

The Flask + Plotly dashboard focuses on practical inspection rather than only
global metrics.

Main views:

- Fixture selector ranked by price range and PMXT availability.
- Unified time-window controls: full match, 90m, 45m, 15m, 5m.
- Main timeline:
  - Polymarket PMXT/CLOB price path.
  - Prediction probability and hit/miss/false-alarm bands.
  - Forecast price path on the same Price axis:
    - model `+60s` price forecast.
    - lead model `+150s` price forecast generated from `t`.
  - Key LSports events.
- Continuous pricing chart:
  - predicted and realized `|Delta P|`.
  - lead-window predicted and realized `|Delta P|`.
  - lead intercept score.
  - forecast price paths shifted to their forecast time.
- Markov/xT explainability:
  - event x remaining-time heatmap.
  - top paths and state-pressure timeline.

## Data Boundary

The local LSports football universe is much larger than the aligned modelling
set. Current model training only uses fixtures with a usable Polymarket market
mapping and price path.

Important files:

- LSports Hyper football events: `data/hyper/football/...`
- Polymarket fixture matches: `data/polymarket/fixture_market_matches.parquet`
- PMXT orderbook paths: `data/polymarket/prices/*.parquet`
- Main modelling table: `outputs/real_dataset_v4.parquet`
- Dashboard predictions: `outputs/test_predictions.parquet`

Large parquet and model artifacts are intentionally gitignored.

## Pipeline

```text
LSports Hyper messages
        +
Polymarket CLOB/PMXT prices
        |
        v
src/build_dataset_v4.py
        |
        v
outputs/real_dataset_v4.parquet
        |
        +--> src/train_models.py
        |       baseline classifiers
        |
        +--> src/markov_xt_model.py
        |       confidence-weighted Markov/xT state values
        |
        +--> src/optimize_markov_augmented_recall.py
        |       recall-first high-volatility interception
        |
        +--> src/train_continuous_pricing.py
        |       same-window continuous price movement forecast
        |
        +--> src/train_lead_pricing.py
                strict 30s lead-time pricing forecast
```

## Key Scripts

| Script | Purpose |
| --- | --- |
| `src/build_dataset_v4.py` | Build event-price aligned football dataset. |
| `src/train_models.py` | Chronological baseline classifier training. |
| `src/markov_xt_model.py` | Markov/xT state, transition, value, path artifacts. |
| `src/optimize_recall_models.py` | Recall-first model search. |
| `src/optimize_markov_augmented_recall.py` | Recall model with Markov features and early stopping. |
| `src/train_continuous_pricing.py` | Continuous `|Delta P|` and signed move forecast. |
| `src/train_lead_pricing.py` | 30s lead-gap price-move and intercept forecast. |
| `viz/app.py` | Flask API and dashboard server. |
| `viz/templates/index.html` | Plotly frontend. |

## Quick Start

```bash
pip install -r requirements.txt

python src/build_dataset_v4.py
python src/train_models.py
python src/markov_xt_model.py
python src/optimize_markov_augmented_recall.py
python src/train_continuous_pricing.py
python src/train_lead_pricing.py

python viz/app.py
```

Open:

- http://localhost:5001

Useful APIs:

- `/api/metrics`
- `/api/continuous_metrics`
- `/api/lead_pricing_metrics`
- `/api/fixture_list`
- `/api/fixture_timeline/<fixture_id>`
- `/api/fixture_deepdive/<fixture_id>`

## Deployment

Current production package lives on the server at:

- `/opt/probly_display`

Service:

- `probly-display`
- binds to `127.0.0.1:8085`
- public paths:
  - `/probly/`
  - `/probly-api/`

Deployment needs the Flask app, template, and refreshed output artifacts. Large
parquet files are transferred outside git.

## What Is Working

- Chronological fixture split avoids event-level train/test leakage.
- Validation-selected thresholds are used instead of tuning directly on test.
- Lead target excludes the first 30 seconds of Polymarket reaction.
- Dashboard can inspect individual fixtures at multiple time scales.
- PMXT orderbook paths are used when available; otherwise the dashboard falls
  back to the CLOB candle path used in training labels.
- Metrics now report counts in plain language:
  - total future high-volatility intervals.
  - predicted intercept intervals.
  - accurate intercepts.
  - missed intervals.
  - false alarms.

## TODO / Weak Points

- Signed price direction is still not strong enough. It needs orderbook depth,
  spread, trade flow, maker/taker attribution, and market-type-specific labels.
- Current PMXT coverage is partial. More Polymarket markets and fixtures should
  be matched before drawing broad conclusions.
- Market types are mixed in the aligned dataset. Win/Draw, O/U, BTTS, corners,
  and spread-like markets need separate target functions.
- The lead model predicts movement magnitude better than exact fair price.
  Position sizing should use the high-confidence mode until signed forecasts
  improve.
- The chart can show forecast-vs-real paths, but it is not a PnL backtest yet.
  A fill/slippage/orderbook simulator is still required.
- Confidence-grade dynamics should be modelled as a path, not just as a point
  feature.
- Calibration should be monitored by league, liquidity bucket, and market type.

## Research Notes

Implementation work and model attempts are recorded in:

- `docs/worklogs/2026-05-25_markov_xt_rewrite.md`

Reference requirement/spec copies are in:

- `docs/hf_docs/`

