# Held-Out LSports Alert Evaluation

## Alert Formula

The alert is not an LSports abnormal-match field. It is a model score computed from LSports event-flow features available at time `t`.

For each fixture, the event stream is sampled every 30 seconds in this batch feature table. At each time `t`, features include current match minute, score state, period, cumulative cards, and rolling 60/180/300/600 second counts or xT-weighted sums for live events, attacks, dangerous attacks, shots, and corners.

The training label is:

`y_t = 1{a goal occurs in (t, t + 120 seconds]}`

The fitted logistic hazard is:

`hazard_t = sigmoid(beta_0 + sum_j beta_j * zscore(x_{t,j}))`

The score is smoothed with a 3-step rolling mean. An alert episode starts when the smoothed score is above the chosen threshold. Repeated crossings within 90 seconds are collapsed into one episode.

A true alert means a goal occurs within the next 120 seconds. A false alert means no goal occurs in that horizon.

## Leakage Control

- Train dates: `2026-05-24,2026-05-25,2026-05-26`
- Validation dates for threshold selection: `2026-05-27`
- Final held-out test dates: `2026-05-28`
- Test fixtures are not used for model fitting or threshold selection.

## Summary

| split | threshold | n_fixtures | n_goals | n_alerts | true_alerts | false_alerts | precision | goal_recall | alerts_per_fixture | false_alerts_per_fixture | mean_goal_lead_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 0.687 | 37 | 106 | 40 | 4 | 36 | 0.100 | 0.038 | 1.081 | 0.973 | 33.664 |
| test | 0.687 | 32 | 109 | 28 | 3 | 25 | 0.107 | 0.028 | 0.875 | 0.781 | 69.359 |

## Selected Features

minute, home_score, away_score, score_diff, total_goals, n_events_60s, xt_sum_60s, n_danger_60s, n_shot_60s, n_attack_60s, n_corner_60s, n_events_180s, xt_sum_180s, n_danger_180s, n_shot_180s, n_events_300s, xt_sum_300s, n_danger_300s, n_shot_300s, n_events_600s, xt_sum_600s, cum_yellow, cum_red, period_id

## Figures

- `outputs/figures/heldout_alert_threshold_tradeoff.png`
- `outputs/figures/heldout_alert_test_by_fixture.png`
- `outputs/figures/heldout_alert_score_distribution.png`

## Interpretation

This is the first honest alert test. If precision is low or false alerts per fixture are high, the signal should be treated as a risk filter rather than a trading trigger. The next step is to combine this held-out LSports hazard with Polymarket BBO filters on mapped markets.
