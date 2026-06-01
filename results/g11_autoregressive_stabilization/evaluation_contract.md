# G11 Evaluation Contract

## Purpose

Define three offline evaluation modes for assessing autoregressive stability of residual-action prediction models. G10 established that residual-action prediction is superior under teacher-forced H=1 conditions (0.0133 vs 0.0321 for action_history_gru, 2.4x improvement), but autoregressive error growth is rapid (0.007 to 0.30 over 60+ steps). G11 stabilizes offline autoregressive behavior and defines an explicit closed-loop readiness gate.

## Evaluation Modes

### 1. teacher_forced_h1

**Description:** Standard teacher-forced evaluation. At each timestep t, the model receives ground-truth action_history[t-k:t-1] and predicts action[t].

**Inputs:**
- observation/state at t
- action_history[t-k:t-1] (ground truth)
- task/instruction

**Target:** action[t] (direct) or action[t] - action[t-1] (residual)

**Reconstruction:** action[t-1] + predicted_residual (for residual models)

**No environment interaction.**

### 2. autoregressive_open_loop

**Description:** Predict actions in a closed loop over the recorded observation/state sequence. At t=0, history is initialized from ground truth (or documented bootstrap). At each subsequent step, the predicted action from the previous step is fed back into the action history. The observation/state sequence is fixed from the recording (not updated by the model).

**Inputs:**
- observation/state at t (from recording)
- action_history[t-k:t-1] (predicted for t > start, ground truth for t = start)
- task/instruction

**Target:** action[t] from recording

**No environment interaction. No simulator stepping.**

**Bootstrap rule:** At t=history_len, initialize history from ground-truth actions[0:history_len].

### 3. corrupted_history_robustness

**Description:** Test robustness to noisy/corrupted action history. Uses ground-truth history with controlled perturbations.

**Inputs:**
- observation/state at t (from recording)
- action_history[t-k:t-1] with added noise/corruption
- task/instruction

**Corruption types:**
- Gaussian noise: epsilon ~ N(0, sigma * std_of_residuals)
- Action dropout: randomly replace entries with last_action or zero
- Temporal jitter: shift history window by +/- 1 step

**No environment interaction.**

## Causal Contract (preserved from G10)

- Inputs may include observation/state at t, task/instruction, and previous action history.
- Residual target uses action[t] - action[t-1].
- Reconstruction is action[t-1] + predicted_residual.
- Inputs must NOT include action[t], future actions, future observations, or future states.

## Metrics

### Primary Metrics

| Metric | Description | Bucketed? |
|--------|-------------|-----------|
| continuous_normalized_mse | MSE of (pred - target) / std_safe over continuous dims | Yes (by horizon) |
| continuous_raw_mae | MAE of continuous action dims | Yes (by horizon) |
| gripper_sign_accuracy | Fraction of correct sign predictions | Yes (by horizon) |
| error_growth_slope | Linear slope of continuous_normalized_mse over horizon | No |
| max_error | Maximum continuous_normalized_mse over sequence | No |
| action_drift_norm | L2 norm of predicted action sequence mean minus expert mean | No |

### Per-Dimension Metrics

| Metric | Description |
|--------|-------------|
| per_dim_continuous_mse | MSE per continuous dimension (0-5) |
| per_dim_continuous_mae | MAE per continuous dimension |
| delta_rot_x_mse | MSE specifically for dimension 3 (delta_rot_x) |

### Horizon Buckets

| Bucket | Timesteps |
|--------|-----------|
| h=1 | First step only |
| h=5 | Steps 1-5 |
| h=10 | Steps 1-10 |
| h=20 | Steps 1-20 |
| h=40 | Steps 1-40 |
| h=60 | Steps 1-60 |
| full | Full sequence |

### Diagnostic Metrics

| Metric | Description |
|--------|-------------|
| gripper_transition_f1 | F1 for open/close transitions (diagnostic only) |
| worst_window_mse | MSE in worst 10-step window |
| worst_window_start | Start timestep of worst window |
| phase_error | Per-phase error if phase segmentation available |
| predicted_history_drift | Whether predicted histories converge to constant/zero/last-action |

## Models Under Evaluation

| Model | Type | Description |
|-------|------|-------------|
| last_action | baseline | Repeat last action from history |
| direct_action_action_history_gru | direct | GRU over action history, direct prediction |
| residual_action_action_history_gru | residual | GRU over action history, residual prediction |
| residual_action_full_state_plus_history | residual | GRU + full state, residual prediction |
| residual_action_full_state_plus_history_separate_heads | residual | GRU + full state, separate pos/rot heads, residual |

## Stabilization Variants

| Variant | Description | Label |
|---------|-------------|-------|
| No augmentation | Standard residual training (baseline) | baseline |
| History noise augmentation | Perturb action_history with noise from empirical residual distribution | history_noise_aug |
| History dropout / replacement | Randomly replace history entries with noisy or last_action entries | history_dropout_aug |
| Offline scheduled sampling | Mix ground-truth and predicted histories during training | offline_scheduled_sampling |
| Multi-step unrolled loss | Backprop through short autoregressive rollouts | offline_multistep_loss |
| Temporal smoothness regularization | Penalize excessive action changes | diagnostic_regularization |

## Output Files

| File | Description |
|------|-------------|
| evaluation_contract.md | This file |
| autoregressive_rollout_metrics.csv | Per-demo, per-mode, per-horizon metrics |
| error_growth_by_horizon.csv | Mean error by horizon bucket across demos |
| per_dim_autoregressive_errors.csv | Per-dimension error breakdown |
| autoregressive_baseline_ladder.csv | Comparison of all models under all modes |
| stabilization_variant_ladder.csv | Comparison of stabilization variants |
| multidemo_autoregressive_metrics.csv | Mean/std over held-out demos |
| heldout_demo_error_growth.csv | Error growth on held-out demos |
| closed_loop_readiness_gate.md | Readiness criteria assessment |
| failure_mode_analysis.md | Worst-case analysis |
| worst_rollout_windows.csv | Worst error windows |

## Closed-Loop Readiness Gate Criteria

A model may proceed to a limited closed-loop smoke test ONLY if ALL of:

1. Residual model beats last_action on teacher_forced continuous_normalized_mse.
2. Residual model beats last_action on autoregressive full-sequence continuous_normalized_mse.
3. Error growth slope is lower than the non-stabilized residual baseline.
4. No severe phase-specific blowup above 0.5 continuous_normalized_mse.
5. Gripper sign accuracy does not collapse (>80%) under autoregressive rollout.
6. Results hold on held-out demos from the same task.
7. Artifact registry and claims ledger pass checkers.

## Non-Claims

The following are NOT claimed by this evaluation:
- Closed-loop success
- Future-latent benefit/harm/no-effect
- WAM-GRU architecture validity
- DINOv2 suitability
- Offline improvement guarantees closed-loop success
- full_state_92d as true oracle (unless decomposition proven)
