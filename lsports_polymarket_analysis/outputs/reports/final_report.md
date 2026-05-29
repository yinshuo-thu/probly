# LSports x Polymarket Data Value Report

## Executive Summary

This repository now contains a reproducible pipeline for studying whether LSports
football event streams can improve Polymarket sports-market pricing, stale-price
detection, and event-driven risk controls.

The current evidence is split into three layers:

1. **Official Polymarket `/prices-history` for one key match**: coarse price
   history moved after LSports goal timestamps by about 9 / 44 / 51 seconds.
2. **Historical BBO from PMXT for the same key match**: BBO mid moved before
   the LSports goal timestamp by about 9.8 / 5.6 / 1.8 seconds. On this more
   precise BBO benchmark, Polymarket BBO was faster than the LSports goal event
   timestamp for the three observed goals.
3. **LSports pre-goal alert model on held-out fixtures**: a simple logistic
   hazard model trained on earlier fixtures has weak out-of-sample signal. On
   a 32-fixture held-out test set, the conservative validation-selected alert
   threshold produced 28 alert episodes, 3 true alerts, and 25 false alerts
   (precision 10.7%, goal recall 2.8%). This is not yet usable as a direct
   trading trigger.

The important practical conclusion is:

**Do not trade simply after an LSports goal event. In the key PMXT BBO case,
BBO had already moved. The useful research direction is earlier than the goal:
use LSports event-flow intensity as a risk filter or pre-goal warning signal,
then require BBO/liquidity/spread confirmation before acting.**

## Key Match BBO Result

- Fixture: `18746260`
- Match: Red Bull Bragantino vs Carabobo FC
- Event: Copa Sudamericana, kickoff `2026-05-28T00:30:00Z`
- Polymarket event slug: `sud-bra-car-2026-05-27`
- Market: Red Bull Bragantino win, Yes token
- Historical BBO source: PMXT Archive

| LSports goal UTC | BBO move UTC | BBO lag vs goal | Faster side |
|---|---|---:|---|
| 2026-05-28 00:46:54.967 | 2026-05-28 00:46:45.178 | -9.8s | BBO |
| 2026-05-28 02:07:21.622 | 2026-05-28 02:07:16.022 | -5.6s | BBO |
| 2026-05-28 02:24:12.788 | 2026-05-28 02:24:10.969 | -1.8s | BBO |

Negative lag means the BBO move happened before the LSports goal timestamp.

Figures:

- `outputs/figures/single_match_bbo_goal_windows.png`
- `outputs/figures/single_match_bbo_latency_bars.png`
- `outputs/figures/single_match_signal_vs_bbo.png`

## Held-Out Alert Evaluation

The `alert` shown in the signal plot is **not** an official LSports abnormal-match
flag. It is a model-derived warning score:

`y_t = 1{a goal occurs in (t, t + 120 seconds]}`

`hazard_t = sigmoid(beta_0 + sum_j beta_j * zscore(x_{t,j}))`

where `x_t` includes current score, match minute, period, cards, and rolling
60/180/300/600 second event counts or xT-weighted sums for attacks, dangerous
attacks, shots, corners, and live events.

Leakage control:

- Train: 2026-05-24 to 2026-05-26
- Validation threshold selection: 2026-05-27
- Final test: 2026-05-28
- Test fixtures were not used for model fitting or threshold choice.

Held-out result at the validation-selected conservative threshold `0.6874`:

| split | fixtures | goals | alerts | true | false | precision | goal recall | false alerts / fixture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 37 | 106 | 40 | 4 | 36 | 10.0% | 3.8% | 0.97 |
| test | 32 | 109 | 28 | 3 | 25 | 10.7% | 2.8% | 0.78 |

This means the simple alert formula has weak predictive power. It can be useful
as an exploratory risk feature, but not as a standalone "goal is coming" signal.

Figures:

- `outputs/figures/heldout_alert_threshold_tradeoff.png`
- `outputs/figures/heldout_alert_test_by_fixture.png`
- `outputs/figures/heldout_alert_score_distribution.png`

## Batch LSports Coverage

- Sampled matches: 200
- Goals identified: 596
- Average goals per match: 2.98
- Share of matches with at least one goal: 92.5%
- Median live-event gap by match: 3.06 seconds
- Median p90 live-event gap by match: 9.05 seconds
- Archive lag is batch ingestion lag, not real-time feed latency.

## Reproduction

```bash
cd lsports_polymarket_analysis

python scripts/download_hf_data.py --docs --max 40
python scripts/batch_event_analysis.py --max 40

export PMXT_API_KEY="<your_pmxt_key>"
python scripts/single_match_bbo_analysis.py --lookback-sec 10 --lookahead-sec 300
python scripts/signal_strategy_analysis.py
python scripts/heldout_alert_evaluation.py
```

## Recommended Next Modeling Step

Move from "predict goal in the next N seconds" to "predict BBO repricing before
the BBO reprices":

- Target: future BBO mid move, e.g. `abs(mid_t+h - mid_t) >= 3c`.
- Features: LSports event-flow intensity plus current BBO spread, depth, and
  recent microprice movement.
- Validation: grouped and chronological by fixture/date/league.
- Action rule: alert only when the model score is high, BBO has not already
  moved, spread is tradable, and liquidity is sufficient.
