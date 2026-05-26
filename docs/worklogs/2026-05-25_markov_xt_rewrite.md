# 2026-05-25 Markov xT Rewrite Worklog

## Source Documents

- Requirement/model text was supplied directly in the conversation.
- The remote HuggingFace document URL previously returned HTTP 401, so implementation is based on the copied text plus the existing local file at `docs/hf_docs/MODEL_SPEC_markov_pricing.md`.

## Current Project Reading

- Current aligned modelling table: `outputs/real_dataset_v4.parquet`
- Rows: 1,332,812
- Matched fixtures in modelling table: 54
- Current label: `high_volatility`, derived from large forward Polymarket price movement over the event window.
- Important boundary: this table is not the full `probly/lsports-hyper-dataset` football universe. It is the subset with Polymarket/price alignment, so coverage and model recall must be reported separately.

## Gap Against Requirement

- Existing recall-first detector is useful for coverage, but mostly behaves like a black-box classifier.
- Requirement asks for a Markov Chain xT style layer:
  - explicit state vector
  - confidence-weighted transition matrix
  - event/state price impact
  - value iteration
  - path confidence and event attribution
  - heatmaps and top paths
- Frontend needs to show practical time-series comparison and explainability, not only global F1/recall metrics.

## Implementation Plan

1. Keep the existing recall-first detector as the best warning engine if it remains stronger.
2. Add a separate Markov xT model that trains on the same chronological fixture split.
3. Use states:
   - score bucket
   - remaining-time bucket
   - period bucket
   - momentum bucket
   - event group
   - confidence bucket
4. Learn confidence-weighted transitions and state/event price impact.
5. Run value iteration to estimate forward price-risk value.
6. Select a threshold on validation with a recall-first objective and precision floor.
7. Export practical artifacts for frontend:
   - `markov_xt_predictions.parquet`
   - `markov_xt_metrics.json`
   - `markov_event_heatmap.json`
   - `markov_top_paths.json`
   - `markov_state_values.parquet`

## Attempt Log

- Attempt 1 started: implement a standalone Markov xT training/evaluation script.
- Attempt 1 result:
  - Script: `src/markov_xt_model.py`
  - Outputs written:
    - `outputs/markov_xt_predictions.parquet`
    - `outputs/markov_xt_metrics.json`
    - `outputs/markov_event_heatmap.json`
    - `outputs/markov_top_paths.json`
    - `outputs/markov_state_values.parquet`
  - Test result: total high-volatility intervals 90,792; predicted/intercepted intervals 19,114; accurate intercepts 8,736; missed 82,056; false alarms 10,378.
  - Conclusion: pure discrete Markov xT is interpretable but not enough as the main warning model. Recall is 9.62%, so it should be used as an explanation/feature layer.
- Attempt 2 started: use Markov xT state/event/path values as recall-model features.
- Attempt 2 result:
  - Script: `src/optimize_markov_augmented_recall.py`
  - Added features:
    - `markov_state_risk`
    - `markov_event_risk`
    - `markov_event_move`
    - `markov_state_value_raw`
    - `markov_path_score`
    - `markov_path_confidence`
  - Early stopping triggered after 10 consecutive candidates without effective validation recall improvement.
  - Best validation-selected candidate: `ExtraTrees balanced + Markov`
  - Test result: total high-volatility intervals 90,792; predicted/intercepted intervals 223,657; accurate intercepts 81,196; missed 9,596; false alarms 142,461.
  - Metrics: precision 36.30%, recall 89.43%, F1 51.64%, F3 78.01%, AUC 69.26%.
  - Compared with previous promoted model:
    - Accurate intercepts: 80,715 -> 81,196
    - Missed intervals: 10,077 -> 9,596
    - False alarms: 143,305 -> 142,461
  - Conclusion: small but real improvement, plus better explainability. Promoted `outputs/markov_augmented_test_predictions.parquet` to `outputs/test_predictions.parquet` and updated `outputs/metrics_history.json`.
- Frontend/API update:
  - Added `/api/markov_overview`.
  - Added global Markov event x remaining-time heatmap and Top-K path table.
  - Added Markov path score to per-fixture state timeline and signal-window context.

## 2026-05-26 Continuous Pricing + Time-Selectable Frontend

- User feedback:
  - Whole-match time series hides details; every time-series view should support selectable time windows.
  - Binary high-volatility classification is useful for warning coverage, but pricing research should also forecast continuous price movement.
  - Markov state logic should be pushed further toward the original state-transition/xT idea.
- Attempt 3 started: add a continuous pricing layer on top of the Markov explanation layer.
- Implementation:
  - Script: `src/train_continuous_pricing.py`
  - Targets:
    - `max_abs_move_120s`: expected absolute Polymarket price movement over the next 120 seconds.
    - `price_change_1m`: expected signed 1-minute price change.
  - Markov v2 state:
    - score bucket
    - remaining-time bucket
    - period bucket
    - momentum bucket
    - pressure bucket
    - price-zone bucket
    - event group
    - confidence bucket
  - Additional context:
    - score context bucket
    - realized volatility bucket
    - confidence and event-density weighted transitions
  - Learned tables:
    - state expected absolute move
    - event/time/score-context expected absolute move
    - state expected signed move
    - event/time/score-context expected signed move
    - value-iteration state values for absolute and signed movement
  - Model family:
    - validation search over HistGradientBoostingRegressor losses/configs for absolute movement.
    - separate signed-change HistGradientBoostingRegressor.
- Data leakage check:
  - First run showed unrealistically perfect signed-change performance.
  - Root cause: `FEATURE_COLS` from the binary classifier included `price_change_1m`, which is the signed regression target.
  - Fix: remove `price_change_1m` from continuous-model features before training.
- Attempt 3 result after leakage fix:
  - Rows train/validation/test: 831,329 / 213,872 / 287,611.
  - Continuous feature count: 37.
  - Markov v2 transition states: 12,467.
  - Best absolute-move validation model: `HGB abs poisson-like`.
  - Validation absolute-move metrics: MAE 0.02750, RMSE 0.06887, R2 0.1491, Spearman 0.3855, top-decile lift 3.1425.
  - Test absolute-move metrics: MAE 0.03596, RMSE 0.07903, R2 0.0887, Spearman 0.4038, top-decile lift 2.6528.
  - Test signed-change metrics: MAE 0.01970, RMSE 0.05440, R2 0.1310, Spearman 0.6279.
  - Direction accuracy on samples with absolute 1-minute move >= 2c: 87.80% over 70,289 rows.
  - Outputs:
    - `outputs/continuous_pricing_predictions.parquet`
    - `outputs/continuous_pricing_metrics.json`
    - `outputs/models/continuous_pricing_models.pkl`
  - Promoted continuous prediction columns into `outputs/test_predictions.parquet`.
- Frontend/API update:
  - Added `/api/continuous_metrics`.
  - Added continuous fields to `/api/fixture_timeline/<fixture_id>` and `/api/fixture_deepdive/<fixture_id>`.
  - Added a unified time-window toolbar: full match, 90m, 45m, 15m, 5m.
  - Time-window selection now relayouts the main timeline, continuous price-path chart, and state-pressure chart together.
  - Added continuous price-path chart comparing:
    - predicted `|ΔP|` over 120 seconds
    - realized `|ΔP|` over 120 seconds
    - predicted signed 1-minute change
    - current price and predicted 1-minute-after price
  - Added continuous risk line to the state-pressure chart and continuous signal bars to the incident impact chart.
- Verification:
  - `python -m py_compile src/train_continuous_pricing.py viz/app.py` passed.
  - Local Flask app on port 5011 returned `/api/continuous_metrics`.
  - Local `/api/fixture_timeline/<fixture_id>` returned continuous fields in `price_line`, `pred_line`, and event markers.

## 2026-05-26 Lead-Time Pricing Requirement

- User requirement:
  - Stop only after the model shows reasonable advance pricing ability versus real Polymarket data.
  - The display must make the lead visible, not only show same-window volatility.
  - Push completed code to GitHub.
- Attempt 4 started: create a stricter lead-time target.
- Target definition:
  - At LSports event time `t`, use only current/past features.
  - Predict Polymarket movement beginning after a 30 second gap: `[t+30s, t+150s]`.
  - Continuous target: `lead30_max_abs_move_120s`.
  - Signed target: `lead30_signed_move_120s`.
  - Intercept target: `lead30_high_volatility = lead30_max_abs_move_120s > 3c`.
  - This is stricter than `max_abs_move_120s`, because the first 30 seconds of immediate market reaction are excluded.
- Implementation:
  - Script: `src/train_lead_pricing.py`.
  - Reuses the Markov v2 state/reward/value features.
  - Trains:
    - lead absolute move regressor.
    - lead signed move regressor.
    - lead intercept classifier used only to calibrate alert scores.
  - Uses validation-selected thresholds:
    - coverage threshold: aims to intercept many future move intervals.
    - high-confidence threshold: requires about 60% validation precision for cleaner pricing signals.
  - Outputs:
    - `outputs/lead_pricing_predictions.parquet`
    - `outputs/lead_pricing_metrics.json`
    - `outputs/models/lead_pricing_models.pkl`
  - Promotes lead fields into `outputs/test_predictions.parquet`.
- Attempt 4 result:
  - Rows train/validation/test: 831,329 / 213,872 / 287,611.
  - Test lead absolute-move Spearman: 0.3963.
  - Test lead absolute-move top-decile lift: 1.8021.
  - Coverage threshold test:
    - Total future high-volatility intervals: 110,597.
    - Predicted intercept intervals: 168,711.
    - Accurate intercepts: 83,465.
    - Missed intervals: 27,132.
    - False alarms: 85,246.
    - Precision: 49.47%.
    - Recall: 75.47%.
  - High-confidence threshold test:
    - Predicted intercept intervals: 66,002.
    - Accurate intercepts: 39,724.
    - Missed intervals: 70,873.
    - False alarms: 26,278.
    - Precision: 60.19%.
    - Recall: 35.92%.
  - Conclusion:
    - The model has measurable advance pricing power after removing the first 30 seconds of Polymarket reaction.
    - It is useful as a two-mode system:
      - coverage mode for market-maker protection and broad early hedging.
      - high-confidence mode for cleaner directional/pricing opportunities.
- Frontend/API update:
  - Added `/api/lead_pricing_metrics`.
  - Added lead fields to fixture timeline and deepdive APIs.
  - Added top dashboard strip for lead-window metrics.
  - Added lead lines to the main time series and continuous pricing chart:
    - `提前30s预测 |ΔP|`
    - `t+30s后实际 |ΔP|`
    - lead intercept score.
- Verification:
  - `python -m py_compile src/train_lead_pricing.py viz/app.py` passed.
  - Local `/api/lead_pricing_metrics` returned coverage precision 0.4947, recall 0.7547, high-confidence precision 0.6019.
  - Local `/api/fixture_timeline/<fixture_id>` returned `lead30_pred_abs_peak` and `lead30_intercept_score`.

## 2026-05-26 Frontend Forecast Path Fix

- User feedback:
  - The frontend appeared to show straight prediction lines that did not follow Polymarket.
- Diagnosis:
  - The main timeline mixed probability/risk values and price-move magnitudes on the same right-side prediction axis.
  - `|Delta P|` values are usually 0.02-0.08 while the axis is 0-1, so they visually look like nearly flat lines.
  - The main timeline did not show the model's forecast price path on the same axis as Polymarket price.
- Fix:
  - Removed `|Delta P|` traces from the main timeline's probability axis.
  - Kept `|Delta P|` comparison in the dedicated continuous-pricing chart.
  - Added price-axis forecast paths to the main timeline:
    - `模型预测价格 +60s`
    - `提前30s模型价格 +150s`
  - Shifted forecast-price traces to the timestamp they forecast, so the line can be compared visually against later Polymarket prices.
- Interpretation:
  - The model is still a lead-time risk/pricing model, not a pure price tracker.
  - A forecast path should follow the market state through current price plus expected future move; exact signed movement remains a TODO.
