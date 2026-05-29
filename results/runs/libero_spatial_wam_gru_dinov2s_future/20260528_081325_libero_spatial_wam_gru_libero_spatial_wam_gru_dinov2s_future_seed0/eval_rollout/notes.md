# Rollout Evaluation Notes

Status: closed_loop_smoke
Suite: libero_spatial
Task IDs: [0]
Episodes: 5
Max policy steps: 300
Success rate: 0.0

Failure counts:
- max_steps_reached: 5

Evaluator limitations:
- This rollout is a small smoke run unless summary.json marks it otherwise.
- No model comparison is implied by this single-checkpoint evaluation.
- The policy consumes only action history and the current observation latent; no demonstration actions or future observations are used during rollout.
- Failure episodes remain in eval_rollout.csv and are not filtered.

Recorded media:
- episode 0: results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0000.npy
- episode 1: results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0001.npy
- episode 2: results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0002.npy
- episode 3: results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0003.npy
- episode 4: results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_rollout/failure_videos/episode_0004.npy
