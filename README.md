# Probly Sports Pricing

> **体育事件实时定价研究 — LSports × Polymarket 跨市场价格发现**

Predict Polymarket high-volatility windows **before** they happen using LSports real-time event stream data.  
Core methods: Markov Chain state transitions + xT (Expected Threat) framework + ML/DL classifiers.

---

## Project Goal

Polymarket prediction markets show sharp price swings triggered by in-game events (goals, red cards, injuries). We detect **precursors** of these events from the LSports event stream **ahead of time** to:
- Predict the direction of the price jump
- Estimate magnitude (L1-L5 volatility grade)
- Give ≥30s advance warning before Polymarket price moves

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yinshuo-thu/probly.git
cd probly

# 2. Install dependencies
pip install -r requirements.txt

# 3. Login to HuggingFace
hf auth login

# 4. Download data
python src/load_datasets.py

# 5. Run model pipeline
python src/run_pipeline.py

# 6. Launch dashboard
python viz/app.py
```

## Code Structure

```
probly/
├── src/
│   ├── load_datasets.py      # HuggingFace data loading
│   ├── feature_engineering.py # Feature extraction
│   ├── markov_model.py       # Markov Chain state transitions
│   ├── xt_model.py           # xT (Expected Threat) framework
│   ├── volatility_predictor.py # ML volatility prediction
│   └── run_pipeline.py       # End-to-end pipeline
├── viz/
│   ├── app.py                # Flask web dashboard
│   └── templates/            # HTML templates
├── docs/
│   ├── TASK_SUMMARY.md       # Full task specification
│   └── MODEL_SPEC.md         # Model architecture details
├── notebooks/                # EDA and experiment notebooks
└── outputs/                  # Model artifacts, metrics
```

## Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| LSports Hyper | ~25GB | 108K+ matches, real-time event stream + confidence |
| LSports Trade | ~46MB | Bookmaker odds snapshots, multi-provider |

## Model Iterations

| Version | Model | F1 | Status |
|---------|-------|-----|--------|
| v0.1 | Logistic Regression baseline | - | In Progress |
| v0.2 | Random Forest + xT features | - | Pending |
| v0.3 | XGBoost + Markov features | - | Pending |
| v0.4 | LSTM sequence model | - | Pending |

## Volatility Grade

| Level | Price Move | Typical Trigger |
|-------|-----------|-----------------|
| L1 | 1-2% | Minor event |
| L2 | 2-5% | Corner, shot on target |
| L3 | 5-10% | Goal scored |
| L4 | 10-20% | Red card, penalty |
| L5 | >20% | Match-changing event |

---

*Powered by LSports Hyper + Polymarket data. Research project by [probly](https://huggingface.co/spaces/probly/probly-sports-pricing).*
