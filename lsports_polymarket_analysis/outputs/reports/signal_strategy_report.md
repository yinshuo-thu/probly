# LSports Signal Strategy Prototype

## Objective

Use LSports event-flow features to forecast goal-driven BBO repricing before the BBO itself jumps, while keeping false alerts low.

## Prototype

- Fixture: `18746260`
- Feature step: 10s
- Prediction horizon: 180s
- Score: logistic goal-hazard fitted on rolling LSports event features, smoothed over 3 steps.
- Alert scoring: collapse repeated threshold crossings into episodes with 90s cooldown.

## Threshold Sweep

| threshold | n_alerts | true_alerts | false_alerts | precision | goal_recall | bbo_preempt_alerts | unique_bbo_moves_preempted | mean_goal_lead_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.184 | 27.000 | 6.000 | 21.000 | 0.222 | 1.000 | 5.000 | 3.000 | 68.647 |
| 0.283 | 23.000 | 6.000 | 17.000 | 0.261 | 1.000 | 6.000 | 3.000 | 71.981 |
| 0.466 | 18.000 | 6.000 | 12.000 | 0.333 | 1.000 | 6.000 | 3.000 | 95.314 |
| 0.624 | 16.000 | 6.000 | 10.000 | 0.375 | 1.000 | 6.000 | 3.000 | 65.314 |
| 0.739 | 9.000 | 6.000 | 3.000 | 0.667 | 1.000 | 6.000 | 3.000 | 125.314 |
| 0.848 | 6.000 | 4.000 | 2.000 | 0.667 | 1.000 | 4.000 | 3.000 | 90.230 |

## Recommended Modeling Path

1. Use historical BBO mid moves as the target: future `abs(mid - baseline) >= 3c` within 10-180s, not just future goals.
2. Train cross-match models with grouped validation by fixture/date/league.
3. Prefer calibrated gradient boosting or discrete-time survival models over single-match logistic regression.
4. Add false-positive controls: minimum liquidity/spread filters, cooldown, probability calibration, and precision-at-k alert thresholds.
5. Deploy as two-stage system: LSports goal-hazard alert first; trade/avoid only when BBO has not already moved or spread/liquidity allow execution.

## Interpretation

For this match, PMXT BBO moved a few seconds before the LSports goal timestamp. Therefore the direct edge is not simply 'trade after LSports goal'. The research opportunity is earlier: use LSports event intensity before the goal to anticipate the BBO jump, and treat BBO movement as the benchmark to beat.