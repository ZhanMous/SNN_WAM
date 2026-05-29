# Overfit Diagnostic Report

## Full-Horizon Overfit (H=4)
- Best MSE: 0.004684 (threshold: 0.0001)
- Best epoch: 298
- Passed: False

## Error Decomposition
- Continuous dims mean MSE: 0.000817
- Gripper dim MSE: 0.029185
- Gripper fraction of total: 623.1%

## H=1 Gate
- Best MSE: 0.003452
- Passed: False

## Lookup-Table Baseline
- Best MSE: 0.000092
- Passed: True

## Diagnosis
H1_FAIL: single-step prediction cannot reach threshold. Basic action-target alignment or optimization problem. Check timestep alignment, action convention, and learning rate.

## Multi-Horizon Ladder
| Horizon | Best MSE | Passed |
|---------|----------|--------|
| 1 | 0.003452 | False |
| 2 | 0.002907 | False |
| 5 | 0.004373 | False |
| 4 | 0.004684 | False |

## Timestep Alignment Sweep
| Shift | Best MSE |
|-------|----------|
| -1 | 0.008356 |
| +0 | 0.008982 |
| +1 | 0.008426 |
| +2 | 0.007521 |

## Gripper Diagnostics
- gripper_mse: 0.029185185208916664
- gripper_mae: 0.09200135618448257
- sign_accuracy_active: 0.9973958134651184
- open_accuracy: 1.0
- close_accuracy: 0.9921875
- transition_precision: 1.0
- transition_recall: 0.8
- transition_f1: 0.888888888888889
- close_timing_error_steps: 1.7
- sign_correlation: 0.984246015548706
- n_open: 256
- n_close: 128
- n_neutral: 0
- n_transitions_true: 10
- n_transitions_pred: 8

