# Rollout Evaluation Notes

Status: closed_loop_smoke
Suite: libero_spatial
Task IDs: [0]
Episodes: 1
Max policy steps: 3
Success rate: 0.0

Failure counts:
- max_steps_reached: 1

Evaluator limitations:
- This rollout is a small smoke run unless summary.json marks it otherwise.
- No model comparison is implied by this single-checkpoint evaluation.
- The policy consumes only action history and the current observation latent; no demonstration actions or future observations are used during rollout.
- Failure episodes remain in eval_rollout.csv and are not filtered.

Recorded media:
- episode 0: results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/failure_videos/episode_0000.npy
