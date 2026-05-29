# Multi-Match LSports Signal Evaluation

## What `alert` Means

`alert` is not an official abnormal-match flag from LSports. It is a prototype model threshold crossing computed from LSports event-flow features.

At every 10-second step `t`, the feature vector includes score state, match minute, cards, period, and rolling 60/180/300/600 second counts or xT-weighted sums of attacks, dangerous attacks, shots, corners, and all live events. The label is:

`y_t = 1{a goal occurs in (t, t + 180 seconds]}`

The fitted score is:

`hazard_t = sigmoid(beta_0 + sum_j beta_j * zscore(x_{t,j}))`

The plotted line is a 3-step rolling average of `hazard_t`. An alert episode starts when the smoothed score exceeds a threshold; repeated crossings inside 90 seconds are collapsed into one episode.

An alert is a true alert if a goal happens within the next 180 seconds. Otherwise it is counted as a false alert. A BBO move is preempted only when the alert timestamp is before the first BBO mid-price move linked to that goal.

## Fixture Summary

| fixture_id | event_date | home | away | market_id | n_goals | n_bbo_snapshots | n_bbo_moves | bbo_first_count | lsports_first_count | median_bbo_lag_sec | operating_threshold | alerts | true_alerts | false_alerts | precision | goal_recall | unique_bbo_moves_preempted | mean_goal_lead_sec | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18746260 | 2026-05-28 | Red Bull Bragantino | Carabobo FC | 2126503 | 3 | 2338 | 3 | 3 | 0 | -5.600 | 0.848 | 5 | 4 | 1 | 0.800 | 1.000 | 3 | 90.230 | ok |
| 18935052 | 2026-05-26 | Saint-Etienne | Nice | 2347821 | 6 | 228 | 0 | 0 | 0 |  | 0.089 | 6 | 0 | 6 | 0.000 | 0.000 | 0 |  | ok |
| 18834749 | 2026-05-27 | Crystal Palace FC | Rayo Vallecano de Madrid | 2252990 | 1 | 914 | 1 | 1 | 0 | -8.012 | 0.025 | 9 | 0 | 9 | 0.000 | 0.000 | 0 |  | ok |

## Threshold Metrics

| fixture_id | event_date | market_id | threshold | n_alerts | true_alerts | false_alerts | precision | goal_recall | bbo_preempt_alerts | unique_bbo_moves_preempted | mean_goal_lead_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18746260 | 2026-05-28 | 2126503 | 0.184 | 6 | 0 | 6 | 0.000 | 0.000 | 0 | 0 |  |
| 18746260 | 2026-05-28 | 2126503 | 0.283 | 5 | 0 | 5 | 0.000 | 0.000 | 0 | 0 |  |
| 18746260 | 2026-05-28 | 2126503 | 0.466 | 5 | 0 | 5 | 0.000 | 0.000 | 0 | 0 |  |
| 18746260 | 2026-05-28 | 2126503 | 0.624 | 5 | 0 | 5 | 0.000 | 0.000 | 0 | 0 |  |
| 18746260 | 2026-05-28 | 2126503 | 0.739 | 4 | 2 | 2 | 0.500 | 0.667 | 2 | 2 | 174.393 |
| 18746260 | 2026-05-28 | 2126503 | 0.848 | 5 | 4 | 1 | 0.800 | 1.000 | 4 | 3 | 90.230 |
| 18935052 | 2026-05-26 | 2347821 | 0.010 | 9 | 0 | 9 | 0.000 | 0.000 | 0 | 0 |  |
| 18935052 | 2026-05-26 | 2347821 | 0.016 | 6 | 0 | 6 | 0.000 | 0.000 | 0 | 0 |  |
| 18935052 | 2026-05-26 | 2347821 | 0.027 | 4 | 0 | 4 | 0.000 | 0.000 | 0 | 0 |  |
| 18935052 | 2026-05-26 | 2347821 | 0.089 | 6 | 0 | 6 | 0.000 | 0.000 | 0 | 0 |  |
| 18935052 | 2026-05-26 | 2347821 | 0.259 | 7 | 0 | 7 | 0.000 | 0.000 | 0 | 0 |  |
| 18935052 | 2026-05-26 | 2347821 | 0.600 | 4 | 1 | 3 | 0.250 | 0.167 | 0 | 0 | 169.762 |
| 18834749 | 2026-05-27 | 2252990 | 0.005 | 6 | 0 | 6 | 0.000 | 0.000 | 0 | 0 |  |
| 18834749 | 2026-05-27 | 2252990 | 0.007 | 6 | 0 | 6 | 0.000 | 0.000 | 0 | 0 |  |
| 18834749 | 2026-05-27 | 2252990 | 0.014 | 8 | 0 | 8 | 0.000 | 0.000 | 0 | 0 |  |
| 18834749 | 2026-05-27 | 2252990 | 0.025 | 9 | 0 | 9 | 0.000 | 0.000 | 0 | 0 |  |
| 18834749 | 2026-05-27 | 2252990 | 0.052 | 7 | 0 | 7 | 0.000 | 0.000 | 0 | 0 |  |
| 18834749 | 2026-05-27 | 2252990 | 0.099 | 4 | 0 | 4 | 0.000 | 0.000 | 0 | 0 |  |

## Interpretation

This is still a prototype, because each fixture is fitted on its own event stream. The results are useful for checking whether the formula has intuitive behavior and whether thresholding produces too many false alerts. Production use should train on many prior matches and validate by held-out fixture/date/league.

Figures:

- `outputs/figures/multi_match_bbo_latency_distribution.png`
- `outputs/figures/multi_match_signal_precision.png`
