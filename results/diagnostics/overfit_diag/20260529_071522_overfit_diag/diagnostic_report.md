# Overfit Diagnostic Report

## Full-Horizon Overfit (H=4)
- Best MSE: 0.004940 (threshold: 0.0001)
- Best epoch: 296
- Passed: False

## Error Decomposition
- Continuous dims mean MSE: 0.000811
- Gripper dim MSE: 0.031760
- Gripper fraction of total: 643.0%

## H=1 Gate
- Best MSE: 0.004045
- Passed: False

## Lookup-Table Baseline
- Best MSE: inf
- Passed: False

## Diagnosis
CRITICAL: lookup-table baseline fails. Basic pipeline/metric/loader problem. Check data loading, loss computation, and optimizer.

## Multi-Horizon Ladder
| Horizon | Best MSE | Passed |
|---------|----------|--------|
| 1 | 0.004045 | False |
| 2 | 0.003215 | False |
| 5 | 0.005005 | False |
| 4 | 0.004940 | False |

## Timestep Alignment Sweep
| Shift | Best MSE |
|-------|----------|
| -1 | 0.009287 |
| +0 | 0.007782 |
| +1 | 0.007237 |
| +2 | 0.007251 |

## Gripper Diagnostics
- gripper_mse: 0.03175951540470123
- gripper_mae: 0.10147017985582352
- sign_accuracy_active: 0.9973958134651184
- open_accuracy: 1.0
- close_accuracy: 0.9921875
- transition_precision: 1.0
- transition_recall: 0.8
- transition_f1: 0.888888888888889
- close_timing_error_steps: 1.8
- sign_correlation: 0.9832441210746765
- n_open: 256
- n_close: 128
- n_neutral: 0
- n_transitions_true: 10
- n_transitions_pred: 8

