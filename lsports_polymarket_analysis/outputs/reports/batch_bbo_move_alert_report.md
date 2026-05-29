# Batch BBO Move And Microstructure Alert Analysis

## Scope

This report uses mapped fixtures with retrievable historical BBO snapshots around LSports goal timestamps. It does not save raw orderbook archives.

## Fixture Summary

| fixture_id | home | away | n_goals | n_bbo_snapshots | n_bbo_moves | bbo_first_count | lsports_first_count | median_bbo_lag_sec | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18746260 | Red Bull Bragantino | Carabobo FC | 3 | 2338 | 3 | 3 | 0 | -5.600 | ok |
| 18935052 | Saint-Etienne | Nice | 6 | 228 | 0 | 0 | 0 |  | ok |
| 18834749 | Crystal Palace FC | Rayo Vallecano de Madrid | 1 | 914 | 1 | 1 | 0 | -8.012 | ok |

## BBO Move Timing

| fixture_id | goal_idx | goal_ts | reaction_ts | reaction_latency_sec | price_jump_magnitude |
| --- | --- | --- | --- | --- | --- |
| 18746260 | 1 | 2026-05-28 00:46:54.967075500+00:00 | 2026-05-28 00:46:45.178000+00:00 | -9.789 | 0.050 |
| 18746260 | 2 | 2026-05-28 02:07:21.622370400+00:00 | 2026-05-28 02:07:16.022000+00:00 | -5.600 | -0.095 |
| 18746260 | 3 | 2026-05-28 02:24:12.788107900+00:00 | 2026-05-28 02:24:10.969000+00:00 | -1.819 | -0.455 |
| 18834749 | 1 | 2026-05-27 20:12:04.013586500+00:00 | 2026-05-27 20:11:56.002000+00:00 | -8.012 | 0.140 |

Negative latency means BBO moved before the LSports goal timestamp.

## Exploratory Pre-Move Signals

| signal | n | median_lead_to_bbo_sec | median_lead_to_goal_sec |
| --- | --- | --- | --- |
| depth_drop_50pct | 3 | 2.002 | 9.788 |
| early_mid_drift_1c | 2 | 3.995 | 7.705 |
| spread_shock | 3 | 4.022 | 9.788 |
| update_burst_5s | 4 | 0.100 | 6.818 |

Interpretation: BBO-only early alerts are often just early repricing, quote churn, or liquidity withdrawal. They should be combined with LSports event intensity to distinguish true goal risk from noisy market microstructure.

Figures:

- `outputs/figures/batch_bbo_move_latency.png`
- `outputs/figures/batch_bbo_micro_alert_leads.png`
- `outputs/figures/batch_bbo_microstructure_windows.png`
