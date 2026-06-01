# G1 DINO-WM Reproduction Smoke — Summary

## Outcome: PASS WITH RISKS

The smoke command `train_offline.py --dry_run --max_steps 1` exited with code 0 and produced a complete artifact set: metrics.csv, checkpoint.pt, best.pt, summary.json, split.json, seeds.txt, train.log, normalization_stats.json.

## Metrics Produced

metrics.csv was produced with the following values:

| Metric | Train | Val |
|---|---|---|
| action_mse | 132.897 | 86.127 |
| future_latent_cosine_error | 1.274 | 1.169 |
| future_latent_mse | 136.681 | 89.142 |
| total_loss | 134.171 | 87.296 |

**These metrics are from mock synthetic data, not real LIBERO data or real DINOv2 patch features.** They validate that the training pipeline (data loading → model forward → loss computation → metrics logging → checkpoint saving) works end-to-end.

## What This Smoke Validates

1. Config loading from `g0_patch_latent_smoke.yaml`
2. Mock dry-run data generation (deterministic, no LIBERO needed)
3. WAM-GRU model instantiation with `smoke_time_index` encoder (latent_dim=8)
4. Patch latent tensor shapes: [B, T, P, D] where P=6 patches
5. Forward pass with action conditioning and future latent prediction
6. Loss computation: action_loss + future_latent_loss
7. Metrics logging to CSV and JSON
8. Checkpoint save/load
9. Normalization stats recording

## What This Smoke Does NOT Validate

1. Real DINOv2 patch feature extraction (uses smoke_time_index encoder)
2. Real LIBERO data loading (uses mock data)
3. DINO-WM-style spatial patch dynamics prediction
4. Multi-step latent drift
5. Planning / action optimization
6. Any SNN, ES, or EGGROLL code
7. Closed-loop robot performance

## Model Info

- Architecture: WAM-GRU (temporal_adapter=wam_gru)
- Visual encoder: smoke_time_index (synthetic, latent_dim=8)
- Text encoder: stub
- Hidden dim: 256
- Parameters: 286,780 (all trainable)
- Action dim: 7
- Patch structure: P=6, D=8

## Data Info

- Suite: libero_spatial (mock)
- Split: mock_dry_run_separate_synthetic_trajectories
- Train windows: 13, Val windows: 13
- History len: 4, Action horizon: 4, Future horizon: 4
- No real LIBERO data loaded
