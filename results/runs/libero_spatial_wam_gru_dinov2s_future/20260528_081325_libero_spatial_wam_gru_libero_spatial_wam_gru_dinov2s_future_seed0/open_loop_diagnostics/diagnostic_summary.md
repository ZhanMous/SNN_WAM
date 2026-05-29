# Teacher-Forced Open-Loop Diagnostic Summary

Run directory: `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0`
Checkpoint: `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/best.pt`
Split: `val`
Dry run: `False`

## Action MSE

| Baseline | Action MSE | Beats baseline? |
|---|---:|---|
| model | 0.019102212 | reference |
| zero_action | 0.22222447 | yes |
| random_action_train_gaussian | 0.42075995 | yes |
| mean_action_train | 0.20015374 | yes |
| last_action | 0.029960487 | yes |

## Interpretation Boundaries

- This is teacher-forced open-loop action prediction on demonstration windows.
- It does not measure closed-loop success, policy robustness, or future-latent rollout benefit.
- The gripper diagnostic uses the last action dimension.

Metrics CSV: `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/open_loop_diagnostics/open_loop_metrics.csv`
Trace CSV: `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/open_loop_diagnostics/action_trace_diagnostics.csv`
