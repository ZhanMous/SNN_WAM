# Diagnostic Summary

Model: wam_gru_dinov2s_future
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
- No action-level analysis is performed in this diagnostic.
- Failure videos (.npy) can be loaded for visual inspection.
