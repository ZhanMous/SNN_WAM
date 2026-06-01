# Overfit Diagnostic Report

## Full-Horizon Overfit (H=4)
- Best MSE: 0.003947 (threshold: 0.0001)
- Best epoch: 299
- Passed: False

## Error Decomposition
- Continuous dims mean MSE: 0.000725
- Gripper dim MSE: 0.023284
- Gripper fraction of total: 589.9%

## H=1 Gate
- Best MSE: inf
- Passed: False

## Lookup-Table Baseline
- Best MSE: inf
- Passed: False

## Diagnosis
CRITICAL: lookup-table baseline fails. Basic pipeline/metric/loader problem. Check data loading, loss computation, and optimizer.

## Gripper Diagnostics
- gripper_mse: 0.02328442968428135
- gripper_mae: 0.07361820340156555
- sign_accuracy_active: 0.9973958134651184
- open_accuracy: 1.0
- close_accuracy: 0.9921875
- transition_precision: 1.0
- transition_recall: 0.8
- transition_f1: 0.888888888888889
- close_timing_error_steps: 1.6
- sign_correlation: 0.9871664047241211
- n_open: 256
- n_close: 128
- n_neutral: 0
- n_transitions_true: 10
- n_transitions_pred: 8

