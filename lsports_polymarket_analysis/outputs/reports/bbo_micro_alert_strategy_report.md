# BBO Microstructure Alert Strategy Test

## Objective

Test whether BBO microstructure can trigger a higher-precision alert before a larger 3c BBO repricing move.

## Target

`true alert = a 3c BBO move occurs within the next 10 seconds`

## Important Limitation

The current complete cached raw BBO sample is tiny: this strategy test uses 3 cached fixture(s) and 4 detected 3c BBO move(s). Treat these results as strategy diagnostics, not out-of-sample proof.

## Rule Strategy Summary

| n_alerts | true_alerts | false_alerts | precision | covered_moves | move_recall | median_lead_sec | strategy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 2 | 4 | 0.333 | 2 | 0.500 | 7.495 | mid1c |
| 5 | 2 | 3 | 0.400 | 2 | 0.500 | 6.995 | mid1c_and_spread2c |
| 2 | 0 | 2 | 0.000 | 0 | 0.000 |  | mid1c_and_depth50 |
| 4 | 2 | 2 | 0.500 | 2 | 0.500 | 4.495 | mid1c_and_update10 |
| 6 | 2 | 4 | 0.333 | 2 | 0.500 | 6.995 | mid1c_and_any_liquidity_stress |
| 6 | 3 | 3 | 0.500 | 3 | 0.750 | 5.022 | spread2c_or_depth50_update10 |

## Best Rule Candidates

| n_alerts | true_alerts | false_alerts | precision | covered_moves | move_recall | median_lead_sec | strategy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 3 | 3 | 0.500 | 3 | 0.750 | 5.022 | spread2c_or_depth50_update10 |
| 4 | 2 | 2 | 0.500 | 2 | 0.500 | 4.495 | mid1c_and_update10 |
| 5 | 2 | 3 | 0.400 | 2 | 0.500 | 6.995 | mid1c_and_spread2c |

## Leave-Fixture-Out Logistic Sanity Check

| n_alerts | true_alerts | false_alerts | precision | covered_moves | move_recall | median_lead_sec | test_fixture | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 2 | 1 | 0.667 | 2 | 0.667 | 2.100 | 18746260 | 0.500 |
| 3 | 2 | 1 | 0.667 | 2 | 0.667 | 2.100 | 18746260 | 0.700 |
| 3 | 2 | 1 | 0.667 | 2 | 0.667 | 2.100 | 18746260 | 0.900 |
| 2 | 1 | 1 | 0.500 | 1 | 1.000 | 3.002 | 18834749 | 0.500 |
| 3 | 1 | 2 | 0.333 | 1 | 1.000 | 3.002 | 18834749 | 0.700 |
| 2 | 1 | 1 | 0.500 | 1 | 1.000 | 3.002 | 18834749 | 0.900 |
| 1 | 0 | 1 | 0.000 | 0 |  |  | 18935052 | 0.500 |
| 1 | 0 | 1 | 0.000 | 0 |  |  | 18935052 | 0.700 |
| 1 | 0 | 1 | 0.000 | 0 |  |  | 18935052 | 0.900 |

## Interpretation

The most promising logic is not a single BBO signal. It is a confirmation rule: require early mid drift plus at least one liquidity-stress symptom (spread shock, depth drop, or update burst). This reduces noisy quote churn compared with raw update bursts, but the current sample is too small to claim stable high precision. The next robust test needs longer BBO windows and non-goal control periods.

Figure:

- `outputs/figures/bbo_micro_rule_strategy_precision.png`
- `outputs/figures/bbo_micro_mid_vs_microstructure.png`
