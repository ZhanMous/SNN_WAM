# G8 Metric Contract v2

## Primary Metrics (scientific)

### Continuous Regression
- **continuous_normalized_mse**: MSE of (pred - target) / std_safe, averaged over continuous dims and timesteps
- **continuous_raw_mse**: Raw MSE of 6 continuous action dims (diagnostic)
- **continuous_raw_mae**: MAE of 6 continuous action dims (diagnostic)
- Per-dimension continuous MSE for decomposition analysis

### Gripper Classification
- **gripper_sign_accuracy**: Fraction of timesteps where predicted sign matches target
- **gripper_transition_f1**: F1 score for detecting open/close transitions (tolerance=2 steps)
- **gripper_open_accuracy**: Accuracy on open timesteps only
- **gripper_close_accuracy**: Accuracy on close timesteps only

## Diagnostic Metrics (NOT primary)

- **global_raw_mse**: MSE over all 7 dims — retained for backward compatibility only
- **gripper_raw_mse**: MSE of gripper dim — diagnostic only (gripper is binary ±1, MSE is not meaningful)
- **old_1e4_gate**: Whether global_raw_mse < 1e-4 — engineering overfit gate only, NOT a scientific pass/fail

## Baseline Comparison Metrics

- **beat_last_action_continuous**: Whether variant's continuous_normalized_mse < last-action baseline
- **beat_last_action_gripper_f1**: Whether variant's gripper_transition_f1 > last-action baseline

## Rules

1. Primary scientific metric is continuous_normalized_mse + gripper_sign_accuracy + gripper_transition_f1.
2. global_raw_mse is diagnostic only; do not use as primary metric.
3. old_1e4 gate is engineering_overfit_gate_only; do not use as scientific pass/fail.
4. Mixed continuous+gripper raw MSE is not a suitable primary scientific metric.
5. All metrics are computed on same-demo teacher-forced H=1 evaluation.
