# Probly Sports Pricing — Task Summary

> **Goal**: Predict Polymarket high-volatility windows BEFORE they happen using LSports real-time event stream data.  
> **Core methods**: Markov Chain state transitions + xT (Expected Threat) framework + ML classifiers.

---

## Research Problem

Polymarket prediction markets show sharp price swings during live sports events. These volatility windows are triggered by in-game events (goals, red cards, injuries, etc.) that change the true probability of an outcome. If we can **detect the precursors** of these events from the LSports event stream **before** the price moves, we can:

1. Predict the direction of the price jump
2. Estimate the magnitude (L1-L5 volatility grade)
3. Give a time-advance warning (seconds to ~2 minutes ahead)

---

## Data Sources

### LSports Hyper Dataset
- **Location**: `probly/lsports-hyper-dataset` (HuggingFace: `probly/lsports-hyper-dataset`)
- **Size**: ~25GB Parquet, 108K+ matches, 21 sports
- **Key fields**:
  - `match_id`, `sport_id`, `timestamp`
  - `event_type` — state transition label (e.g. GOAL, CORNER, SHOT_ON_TARGET)
  - `confidence` — model confidence [0,1] in event classification
  - `team_id`, `player_id`
  - `score_home`, `score_away`, `match_status`
  - `period`, `elapsed_time`
  - Zone/position data (for xT grid)

### LSports Trade Dataset
- **Location**: `probly/lsports-trade-dataset` (HuggingFace: `probly/lsports-trade-dataset`)
- **Size**: ~46MB Parquet
- **Key fields**:
  - `fixture_id`, `market_id`, `odds_provider`
  - `odds_home`, `odds_away`, `odds_draw`
  - `timestamp` — snapshot time
  - Multiple bookmaker feeds

---

## Method Architecture

### Step 1 — State Representation (Markov Chain)
- Define game states: `(score_diff, period, time_bucket, zone, momentum_score)`
- Build transition matrix `P(s' | s, event)` from historical Hyper data
- Each state maps to a **win probability** P(home_win | state)
- State transitions = probability delta = **implied Polymarket price move**

### Step 2 — xT Framework (Expected Threat)
- Map pitch zones to 12×8 grid
- Assign xT value to each zone based on historical scoring probability
- Event sequence → xT delta per possession chain
- High xT accumulation → leading indicator of scoring threat → price move precursor

### Step 3 — Volatility Prediction
- Target variable: `high_volatility_flag` (Polymarket price moves >3% within next 2 minutes)
- Features:
  - Current Markov state + transition probability
  - xT delta over last 60/30/10 seconds
  - Confidence score trend
  - Score difference, time remaining
  - Recent event density (events per 30s window)
  - Bookmaker odds movement (from Trade data)
- Models tried: Logistic Regression → Random Forest → XGBoost → LSTM

### Step 4 — Volatility Grade (L1-L5)
- L1: price move 1-2% (minor event)
- L2: 2-5% (significant event, e.g. corner/shot on target)
- L3: 5-10% (goal scored or major momentum shift)
- L4: 10-20% (red card, penalty)
- L5: >20% (match-changing event)

---

## Success Criteria

- **Precision** ≥ 0.85 on L3+ volatility prediction (5-min lead time)
- **Recall** ≥ 0.80 (don't miss major volatility windows)
- **F1** ≥ 0.82
- **Lead time**: prediction fires ≥ 30 seconds before Polymarket price moves

---

## Deliverables

1. `src/load_datasets.py` — data loading utilities
2. `src/markov_model.py` — Markov Chain state transition model
3. `src/xt_model.py` — xT framework implementation
4. `src/volatility_predictor.py` — ML volatility prediction pipeline
5. `src/feature_engineering.py` — feature extraction
6. `viz/dashboard/` — interactive web dashboard (Plotly + Flask)
7. `docs/MODEL_SPEC.md` — full model specification

---

## Iteration Log

| Version | Model | F1 | Notes |
|---------|-------|-----|-------|
| v0.1 | Logistic Regression baseline | TBD | First pass on sampled data |
| v0.2 | Random Forest + xT features | TBD | Add xT grid |
| v0.3 | XGBoost + Markov features | TBD | Full feature set |
| v0.4+ | LSTM sequence model | TBD | Time-series modeling |
