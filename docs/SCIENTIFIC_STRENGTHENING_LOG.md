# Scientific Strengthening Log

This log tracks the progressive strengthening of diagnostic evidence and methodological rigor.

## 2026-05-29: G6 Representation Bottleneck Diagnostic

**Stage:** G6 representation bottleneck diagnostics
**Status:** Code implemented, tests pass (14/14), experiments pending

### What was implemented

1. **Oracle state baseline** (`OracleStateSplitMLP`): H=1 single-demo overfit using only proprioceptive/state[t] as input. Tests whether low-dimensional state contains sufficient information for next-action prediction.

2. **Raw image CNN baseline** (`RawImageCNN`): H=1 single-demo overfit using raw RGB observation[t] through a deliberately small CNN (Conv2d layers, 128 hidden dim). Tests whether raw pixels contain recoverable control-relevant information that DINO CLS loses.

3. **DINO feature variant ladder**: Systematic comparison of:
   - DINO CLS only (standard, 384-dim)
   - DINO patch mean (384-dim)
   - DINO CLS + patch mean (768-dim)
   - DINO patch tokens + attention pooling (when feasible)

4. **Representation-action retrieval diagnostics**: For each representation variant:
   - Latent variance per dimension
   - Adjacent timestep cosine similarity
   - Nearest-neighbor timestep retrieval (detects if representation encodes demo phase)
   - Nearest-neighbor action retrieval MSE
   - Correlation between latent distance and action distance
   - PCA top-5 concentration

5. **Latent dynamics prediction**: Tests whether z_{t+1} = f(z_t, a_t) is learnable. Measures:
   - Latent MSE on held-out pairs
   - Cosine error
   - Nearest-neighbor next-frame retrieval accuracy

6. **Goal-feature planning diagnostic** (optional): Given z_t and z_{t+h}, tests whether learned latent dynamics can predict that true actions move closer to goal than zero/random actions.

### Design decisions

- **Single-script orchestration**: All baselines run under one script (`src/eval/g6_representation_bottleneck.py`) to ensure consistent data loading, train/val split, and causal contract enforcement.

- **Reuses existing infrastructure**: Causal contract checker (`causal_next_action_v1_check`), training/evaluation loops (`_repair_train_epoch`, `_repair_evaluate`), and dataset classes (`ShiftedTargetWindowDataset`) from the existing codebase.

- **Strict causal contract**: All baselines use `target_shift=0` and `action_horizon=1`. Inputs are observation[t], proprio[t], action_history[t-k:t-1], task_id. Target is action[t]. No leakage.

- **Conservative documentation**: No scientific claims are made from code alone. All claims require experimental results.

### What is NOT claimed

- Future-latent improves performance
- Future-latent harms performance
- Future-latent has no effect
- WAM-GRU architecture is valid
- WAM-GRU architecture is fundamentally invalid
- DINOv2 is unsuitable
- DINO-WM is invalid
- Closed-loop failure is purely covariate shift
- LIBERO evaluator is broken

### Documentation policy

No entries were added to `docs/RESULT_ARTIFACTS.md` or `docs/CLAIMS_LEDGER.md` because no experimental results exist yet. These will be updated after experiments are run and results are generated.

### Pending experiments
- `results/g6_representation_bottleneck/oracle_state_baseline.csv`
- `results/g6_representation_bottleneck/raw_image_cnn_overfit.csv`
- `results/g6_representation_bottleneck/dino_feature_variant_ladder.csv`
- `results/g6_representation_bottleneck/representation_action_retrieval_report.md`
- `results/g6_representation_bottleneck/latent_dynamics_prediction.csv`
- `results/g6_representation_bottleneck/goal_feature_planning_diagnostic.csv`

### Test coverage

14 tests in `tests/test_g6_representation_bottleneck.py`:
- Causal contract rejection tests (action[t] in input, target_shift != 0, future latent in input)
- Model shape tests (OracleStateSplitMLP, RawImageCNN, DinoVariantMLP, LatentDynamicsMLP)
- Retrieval metrics tests
- Latent dynamics diagnostic tests
- Git info test
- Artifact CSV format test (all required fields present)

## 2026-05-29: G8 Mixed-Action Objective and Metric Repair

**Stage:** G8 split continuous+gripper metric repair
**Status:** Implemented, tests pass (15/15), diagnostics run

### What was implemented

1. **Action contract v2**: Documented split of 7-dim action vector into 6 continuous (delta position + orientation) + 1 binary gripper (±1). Includes per-dim statistics, gripper encoding rules, and metric definitions.

2. **Split metrics**: Replaced global raw-action MSE with:
   - Primary: continuous_normalized_mse, gripper_sign_accuracy, gripper_transition_f1
   - Diagnostic: global_raw_mse, old_1e4_gate (engineering only)

3. **Split-head models**: All baselines use `SplitActionGripperHead` with separate continuous regression + gripper classification heads. Loss: SmoothL1 (normalized) + BCEWithLogitsLoss.

4. **Baseline ladder (10 baselines)**: last_action, mean_action, linear_ar, action_history_gru, proprio_only_state, proprio_plus_history, full_state_92d, full_state_plus_history, dino_cls, dino_cls_plus_history.

5. **Raw image loader**: `LazyRawImageDataset` resolves HDF5 frame references to 128x128x3 uint8 RGB. Deferred to keep scope focused.

6. **Full state decomposition audit**: 92-dim state is MuJoCo qpos+qvel; exact decomposition uncertain without model XML. Conservative label 'full_state_92d' retained.

### Key findings

- 5 of 8 trainable baselines beat last_action on continuous normalized MSE
- Best: full_state_plus_history (0.0170), proprio_plus_history (0.0196), dino_cls_plus_history (0.0230)
- Gripper transition F1 capped at 0.667 (only 2 transitions in 103 steps)
- Old 1e-4 gate is engineering_overfit_gate_only, not scientific pass/fail

### Test coverage

15 tests in `tests/test_g8_mixed_action_metrics.py`:
- Action contract identification tests
- Continuous metrics exclude gripper
- Gripper metrics do not affect continuous MSE
- Global raw MSE marked diagnostic
- Old 1e-4 gate engineering only
- Target shift=-1 leakage test
- Baseline comparison metrics
- Normalized MSE uses action stats
- Model shape tests (SplitLinearAR, SplitMLP, SplitGRU, SplitGRUPlusState)
- Frame reference resolution test
- Artifact CSV format test

## 2026-05-29: G9 Residual Error Attribution and Action Target Repair

**Stage:** G9 residual error attribution
**Status:** Implemented, tests pass (7/7), diagnostics run

### Key findings

1. **Residual errors are systematic, not random noise.** Autocorrelation ranges from 0.36 (dim 5) to 0.79 (dim 3), indicating phase-dependent errors.

2. **Capacity plateaus at medium (256 hidden).** full_state_medium achieves 0.0233, full_state_large achieves 0.0244 (slightly worse). The residual error is NOT capacity-limited.

3. **Residual action target is easier.** Predicting action[t] - action[t-1] achieves 0.0121 (1.4x better than direct prediction 0.0170). This suggests modeling action dynamics (change) is more tractable than modeling absolute actions.

4. **Orientation dim 3 (delta_rot_x) has highest normalized error** (0.0627) and strong time correlation (0.68), suggesting orientation prediction is the hardest subproblem.

5. **Worst timesteps cluster around large-motion segments** (t=60,65,83-95), consistent with scale-dependent errors.

6. **Normalization is verified correct** (train-only, gripper excluded, no leakage).

7. **Alignment is correct** (shift=0 is valid causal alignment).

8. **Gripper transition F1 is uninformative** (avg 2.2 transitions/demo across 50 demos).

9. **Upper bounds confirm pipeline validity:** lookup table achieves 9e-14, timestep embedding achieves 5.6e-5.

### Test coverage

7 tests in `tests/test_g9_residual_action_repair.py`:
- Residual attribution basic test
- Per-dim MSE excludes gripper
- Autocorrelation range test
- Split metrics consistency
- Shift sanity labels
- Normalization excludes gripper
- CSV format test
