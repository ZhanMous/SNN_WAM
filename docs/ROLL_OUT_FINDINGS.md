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
- The WAM-GRU rollout attempt was not completed because DINOv2 encoder loading reached a Hugging Face metadata request timeout; no WAM-GRU closed-loop result is registered here.
