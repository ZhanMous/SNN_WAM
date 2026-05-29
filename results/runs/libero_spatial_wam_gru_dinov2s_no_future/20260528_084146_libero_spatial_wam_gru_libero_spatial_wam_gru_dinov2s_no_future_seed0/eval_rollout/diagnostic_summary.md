# Diagnostic Summary

Model: wam_gru_dinov2s_no_future
Suite: libero_spatial
Tasks: [1, 2, 3]
Total episodes: 30
Successes: 0/30 = 0.0%

Failure categories:
  timeout_no_progress: 30

Notes:
- timeout_no_progress: episode hit max_steps without success
- env_done_early: environment terminated before max_steps without success
- environment_error: evaluator/environment exception
- success: episode succeeded

Limitations:
- Failure videos (.npy) can be loaded for visual inspection.

Open-loop action diagnostic:
- Path: `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/open_loop_diagnostics/open_loop_metrics.csv`
- Teacher-forced val action MSE: `0.018413026`
- Simple baselines on the same windows: zero `0.222224474`, random `0.420759946`, train mean `0.200153738`, last action `0.029960487`
- Interpretation: the checkpoint beats all listed open-loop action-MSE baselines, but this is not closed-loop success evidence.

Single-demo overfit diagnostic:
- Path: `results/diagnostics/single_demo_overfit/20260529_wam_gru_no_future_single_demo_overfit_lr0p001_seed0/summary.json`
- Best same-demo teacher-forced action MSE: `0.000527482`
- Predeclared near-zero threshold: `0.000100`
- Interpretation: the run did not validate near-zero single-demo overfit; policy-training architecture claims remain invalid.

Failure classification:
- Closed-loop failure remains `timeout_no_progress`.
- Open-loop MSE alone does not explain the rollout failure.
- Future-latent benefit or harm is not observable because both learned closed-loop policies have zero success.
