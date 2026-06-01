# Overfit Diagnostic Report

## Full-Horizon Overfit (H=4)
- Best MSE: 0.004436 (threshold: 0.0001)
- Best epoch: 288
- Passed: False

## Error Decomposition
- Continuous dims mean MSE: 0.000935
- Gripper dim MSE: 0.027909
- Gripper fraction of total: 629.1%

## H=1 Gate
- Best MSE: 0.004664
- Passed: False

## Lookup-Table Baseline
- Best MSE: 0.002013
- Passed: False

## Diagnosis
CRITICAL: lookup-table baseline fails. Basic pipeline/metric/loader problem. Check data loading, loss computation, and optimizer.

## Multi-Horizon Ladder
| Horizon | Best MSE | Passed |
|---------|----------|--------|
| 1 | 0.004664 | False |
| 2 | 0.002721 | False |
| 5 | 0.005030 | False |
| 4 | 0.004436 | False |

## Timestep Alignment Sweep
| Shift | Best MSE |
|-------|----------|
| -1 | 0.009302 |
| +0 | 0.008538 |
| +1 | 0.008446 |
| +2 | 0.007105 |

## Gripper Diagnostics
- gripper_mse: 0.027909457683563232
- gripper_mae: 0.08849474042654037
- sign_accuracy_active: 0.9947916865348816
- open_accuracy: 0.99609375
- close_accuracy: 0.9921875
- transition_precision: 1.0
- transition_recall: 0.6
- transition_f1: 0.7499999999999999
- close_timing_error_steps: 1.6
- sign_correlation: 0.9850628972053528
- n_open: 256
- n_close: 128
- n_neutral: 0
- n_transitions_true: 10
- n_transitions_pred: 6

