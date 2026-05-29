# Closed-Loop Rollout Findings

## R-G5-MLP-LIBERO-ROLLOUT-SMOKE-001

Status: closed-loop smoke, not reportable.

Checkpoint/config compatibility: passed. See `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/compatibility_report.json`.

Evaluation plan:
- Suite: `libero_spatial`
- Task ID: `0`
- Task name: `pick up the black bowl between the plate and the ramekin and place it on the plate`
- Initial state: fixed LIBERO benchmark `init_state_id=0`
- Episodes: `1`
- Seed: `20260529`
- Max policy steps: `3`
- Settle steps after fixed init: `5`
- Action chunking: receding horizon, first action only (`action_chunk_exec=1`)
- Video recording: enabled

Outputs:
- Episode CSV: `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/eval_rollout.csv`
- Summary: `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/summary.json`
- Failure media: `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/failure_videos/episode_0000.npy`
- Notes: `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/notes.md`

Result:
- Success rate: `0/1 = 0.0`
- Completion steps for successes: none
- Failure counts: `max_steps_reached: 1`
- The failed episode is present in `eval_rollout.csv`; it was not filtered.

Claim audit:
- Supported: the evaluator can run this MLP checkpoint in a real LIBERO closed-loop environment with fixed initial state bookkeeping and episode-level logging.
- Unsupported: any claim that the MLP is effective at LIBERO control.
- Unsupported: any comparison against GRU, WAM-GRU, SNN, or future-latent variants.
- Unsupported: any claim that future-latent prediction improves rollout success.

Limitations:
- The checkpoint was trained with `dry_run: true` and mock data, so this is only an evaluator plumbing smoke.
- The worktree was dirty (`git_dirty=True`) and the run used only one episode with a three-step horizon.

## R-G5-WAM-GRU-FUTURE-LIBERO-ROLLOUT-SMOKE-001

Status: closed-loop smoke, not reportable.

Checkpoint/config compatibility: passed. See `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/compatibility_report.json`.

Evaluation plan:
- Suite: `libero_spatial`
- Task ID: `0`
- Task name: `pick up the black bowl between the plate and the ramekin and place it on the plate`
- Initial states: fixed LIBERO benchmark `init_state_id=0,1,2,3,4`
- Episodes: `5`
- Seed: `0`
- Max policy steps: `300`
- Settle steps after fixed init: `5`
- Action chunking: receding horizon, first action only (`action_chunk_exec=1`)
- Video recording: enabled
- Device: `cuda`

Outputs:
- Episode CSV: `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/eval_rollout.csv`
- Summary: `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/summary.json`
- Failure media:
  - `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0000.npy`
  - `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0001.npy`
  - `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0002.npy`
  - `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0003.npy`
  - `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0004.npy`
- Notes: `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/notes.md`

Result:
- Success rate: `0/5 = 0.0`
- Completion steps for successes: none
- Failure counts: `max_steps_reached: 5`
- All failed episodes are present in `eval_rollout.csv`; none were filtered.

Claim audit:
- Supported: this WAM-GRU future-latent checkpoint can be loaded and executed in a real LIBERO closed-loop environment with fixed initial states and episode-level logging.
- Supported: in this five-episode smoke on task `0`, the checkpoint did not solve any episode within 300 policy steps.
- Unsupported: any claim that future-latent prediction improves rollout success.
- Unsupported: any comparison with MLP, GRU, or SNN.

Limitations:
- The run covers one task and five initial states only (smoke, not reportable).
- No failure taxonomy beyond `max_steps_reached` was assigned from the videos.

## R-G5-WAM-GRU-NO-FUTURE-LIBERO-ROLLOUT-SMOKE-001

Status: closed-loop smoke, not reportable.

Checkpoint/config compatibility: passed. See `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/compatibility_report.json`.

Evaluation plan (matched to future variant):
- Suite: `libero_spatial`
- Task ID: `0`
- Task name: `pick up the black bowl between the plate and the ramekin and place it on the plate`
- Initial states: fixed LIBERO benchmark `init_state_id=0,1,2,3,4`
- Episodes: `5`
- Seed: `0`
- Max policy steps: `300`
- Settle steps after fixed init: `5`
- Action chunking: receding horizon, first action only (`action_chunk_exec=1`)
- Video recording: enabled
- Device: `cuda`

Outputs:
- Episode CSV: `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/eval_rollout.csv`
- Summary: `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/summary.json`
- Failure media:
  - `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/failure_videos/episode_0000.npy`
  - `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/failure_videos/episode_0001.npy`
  - `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/failure_videos/episode_0002.npy`
  - `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/failure_videos/episode_0003.npy`
  - `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/failure_videos/episode_0004.npy`
- Notes: `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_rollout/notes.md`

Result:
- Success rate: `0/5 = 0.0`
- Completion steps for successes: none
- Failure counts: `max_steps_reached: 5`
- All failed episodes are present in `eval_rollout.csv`; none were filtered.

Claim audit:
- Supported: this WAM-GRU no-future checkpoint can be loaded and executed in a real LIBERO closed-loop environment with fixed initial states and episode-level logging.
- Supported: in this five-episode smoke on task `0`, the checkpoint did not solve any episode within 300 policy steps.
- Unsupported: any claim that removing future-latent prediction harms or helps rollout success.

Limitations:
- The run covers one task and five initial states only (smoke, not reportable).
- No failure taxonomy beyond `max_steps_reached` was assigned from the videos.

## Matched Comparison: WAM-GRU Future vs. No-Future

Matched parameters: same task (`0`), same init states (`0–4`), same seed (`0`), same max steps (`300`), same device (`cuda`).

| Variant | Success rate | Failure reason |
|---------|-------------|----------------|
| WAM-GRU future | 0/5 = 0.0 | max_steps_reached: 5 |
| WAM-GRU no-future | 0/5 = 0.0 | max_steps_reached: 5 |

Claim audit:
- Supported: both variants were evaluated under identical conditions on the same task and init states.
- Supported: neither variant solved any episode in this smoke.
- Unsupported: any claim that future-latent prediction improves or degrades rollout success, because the sample size (5 episodes, 1 task) is too small and both scored zero.
- Unsupported: any comparison with MLP, GRU, or SNN variants.

Limitations:
- Five episodes on one task is a smoke, not a reportable evaluation.
- Both variants hit the 300-step ceiling; no failure mode distinction is possible from step counts alone.
- Video-level failure taxonomy was not performed.

## Diagnostic Evaluation: Expert Replay and Baselines

Status: diagnostic evaluation. Expert replay validates the evaluator; model results are conclusive.

### Evaluator Validity Check: Expert Action Replay

Purpose: verify the closed-loop evaluator can produce task success when given correct actions.

- Suite: `libero_spatial`
- Tasks: `1, 2, 3` (task `0` has no demo HDF5)
- Episodes per task: `10`
- Max steps: `300`
- Seed: `0`
- Device: `cuda`
- Action source: demo_0 actions from HDF5, replayed open-loop

| Task | Task name | Success | Notes |
|------|-----------|---------|-------|
| 1 | pick up the black bowl next to the ramekin | 8/10 | init states 4, 5 failed |
| 2 | pick up the black bowl from table center | 10/10 | all succeeded |
| 3 | pick up the black bowl on the cookie box | 9/10 | 1 failure |
| **Total** | | **27/30 = 90%** | |

Conclusion: **the evaluator is valid**. Expert actions achieve 90% success. The 3 failures are from init states that differ enough from demo_0's initial state that the recorded actions don't generalize.

### Sanity Baselines

| Baseline | Success | Notes |
|----------|---------|-------|
| Zero action (hold pose) | 0/30 | expected: no movement |
| Random action (uniform) | 0/30 | expected: no coordinated behavior |

### Model Evaluation (Matched to Expert)

Same tasks (`1, 2, 3`), same init states (`0–9`), same seed (`0`), same max steps (`300`):

| Variant | Success rate | Failure reason |
|---------|-------------|----------------|
| Expert replay | 27/30 = 90% | 3 timeouts |
| WAM-GRU future | 0/30 = 0% | max_steps_reached: 30 |
| WAM-GRU no-future | 0/30 = 0% | max_steps_reached: 30 |
| Zero action | 0/30 = 0% | max_steps_reached: 30 |
| Random action | 0/30 = 0% | max_steps_reached: 30 |

### Claim Audit

- Supported: the evaluator is a valid closed-loop test — expert replay achieves 90% success.
- Supported: both WAM-GRU variants (future and no-future) fail to solve any episode across 30 episodes on 3 tasks.
- Supported: WAM-GRU performs no better than zero-action or random-action baselines.
- Supported: future-latent prediction provides no observable benefit or harm vs. no-future.
- Unsupported: any claim that WAM-GRU is effective at LIBERO closed-loop control.
- Unsupported: any claim that future-latent prediction improves rollout success.
- Unsupported: generalization to tasks 0, 4–9 (not evaluated).

### Implications

1. Both WAM-GRU checkpoints produce actions that fail to solve any LIBERO spatial task in closed-loop evaluation.
2. The failure is not an evaluator artifact: expert replay succeeds on the same evaluator, tasks, and init states.
3. The failure is not task-specific: all 3 tasks show 0% model success.
4. The failure is not a future-latent issue: both future and no-future variants perform identically.
5. The failure mode is uniform: all episodes hit the 300-step ceiling (timeout_no_progress).
6. The models may be producing actions that don't move the robot meaningfully, or that move it in unproductive directions.

### Recommendations

1. **Investigate model action output**: log model predictions during rollout to check if actions are near-zero, oscillating, or divergent.
2. **Check training data alignment**: verify that the training action convention matches the evaluator's action convention (7D, [-1,1], gripper sign).
3. **Check DINOv2 latent alignment**: verify that the encoder produces similar latents at eval time as at training time.
4. **Try longer training or different hyperparameters**: the models may be undertrained.
5. **Add an action clipping guard**: clip model actions to [-1, 1] before env.step() to prevent extreme commands.
