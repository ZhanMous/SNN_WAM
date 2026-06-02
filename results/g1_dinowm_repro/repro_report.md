# G1 DINO-WM Minimal Reproduction Smoke Report

## Scope

- Original DINO-WM only (via closest existing infrastructure)
- No SNN
- No LIBERO port
- No ES / EGGROLL
- Smallest official environment/task/config found: `configs/smoke/g0_patch_latent_smoke.yaml` with `--dry_run --max_steps 1`

## Exact Commands Executed

```bash
# 1. Environment checks
conda run -n snnwam-libero python --version
conda run -n snnwam-libero python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi

# 2. Quality gate
conda run -n snnwam-libero bash scripts/quality_gate.sh

# 3. Smoke reproduction
conda run -n snnwam-libero python src/train/train_offline.py \
  --config configs/smoke/g0_patch_latent_smoke.yaml \
  --dry_run --max_steps 1 \
  --output_dir results/g1_dinowm_repro/

# 4. Tests
conda run -n snnwam-libero python -m pytest tests/ -v
```

## DINOv2 Features

- **Downloaded or cached**: No. The smoke uses `smoke_time_index` encoder, not real DINOv2.
- **Reason**: Real DINOv2 requires downloading `facebook/dinov2-small` (~340MB) and processing real images. The smoke validates plumbing only.
- **Real DINOv2 path**: Available via `DINOv2PatchEncoder` class in `src/models/encoders.py`, but not exercised in this smoke.

## Dataset/Task Used

- **Suite**: `libero_spatial` (configured but not loaded)
- **Data source**: Mock synthetic data (`--dry_run` mode)
- **Split**: `mock_train_0` (train), `mock_val_0` (val)
- **Real LIBERO**: Not installed (`LIBERO_DATASET_ROOT` not set)

## Training Duration

- **Steps**: 1 (dry_run mode)
- **Epochs**: 0 (config) → 1 (actual, due to max_steps=1)
- **Wall time**: < 5 seconds
- **Compute**: CPU-only mock data processing

## GPU/Memory Observations

- **GPU**: NVIDIA GeForce RTX 5060 Ti (8GB VRAM)
- **CUDA available**: Yes
- **GPU utilization**: 3% (idle during mock data processing)
- **Memory usage**: 4327 MiB / 8151 MiB (not from this run)

## Failure Messages

- **None**: The smoke completed successfully with exit code 0.

## Artifacts Produced

| File | Meaning |
|---|---|
| `config.yaml` | Copy of `g0_patch_latent_smoke.yaml` used for this run |
| `command.txt` | Exact command executed |
| `git_commit.txt` | Commit hash + dirty state |
| `environment.txt` | Python, platform, env vars |
| `metrics.csv` | Train/val metrics for 1 epoch (mock data) |
| `checkpoint.pt` | Model checkpoint after 1 step |
| `best.pt` | Best model checkpoint |
| `summary.json` | Full run metadata |
| `split.json` | Mock train/val split |
| `seeds.txt` | Seed used (0) |
| `train.log` | Training log from the run |
| `normalization_stats.json` | Normalization statistics |
| `notes.md` | Limitations and blocked items |

## Key Metrics

| Metric | Train | Val |
|---|---|---|
| action_mse | 132.897 | 86.127 |
| future_latent_cosine_error | 1.274 | 1.169 |
| future_latent_mse | 136.681 | 89.142 |
| total_loss | 134.171 | 87.296 |
| parameter_count | 286,780 | - |

## Outcome

**PASS WITH RISKS**

The smoke command exited with code 0 and produced a complete artifact set. However:

1. All data is mock/synthetic — no real LIBERO trajectories loaded
2. The encoder is `smoke_time_index` (synthetic) — not real DINOv2 patch features
3. No DINO-WM-style action-conditioned future latent predictor exists in the codebase
4. The run validates plumbing (config → data → model → loss → metrics → checkpoint), not DINO-WM science

## DINOv2 Smoke Results (Real Encoder)

A second smoke was run with real DINOv2 ViT-S/14 features:

```bash
conda run -n snnwam-libero python src/train/train_offline.py \
  --config configs/smoke/g1_dinowm_repro_dinov2_smoke.yaml \
  --dry_run --max_steps 1 \
  --output_dir results/g1_dinowm_repro_dinov2/
```

| Metric | Train | Val |
|---|---|---|
| action_mse | 156.199 | 47.928 |
| future_latent_cosine_error | 1.025 | 0.959 |
| parameter_count | 769,564 | - |

Key differences from smoke_time_index:
- **Real DINOv2 encoder**: `facebook/dinov2-small` (revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`)
- **Latent dim**: 384 (real DINOv2-S CLS token dimension)
- **Parameter count**: 769,564 (vs 286,780 with smoke encoder)
- **Metrics**: Different values showing real features produce different behavior

This validates that real DINOv2 features flow through the training pipeline end-to-end.

## Limitations

- **Smoke-scale only.** This is not a benchmark reproduction.
- **Not scientific evidence.** Mock data, synthetic encoders, single step.
- **Does not validate DINO-WM.** No DINO-WM training loop exists. No real patch features. No spatial dynamics modeling.
- **Does not support claims about:** SNN, LIBERO, robustness, closed-loop robot performance, energy efficiency, generalization, planning.
- **Does not validate the DINO-WM mechanism.** The run tests that the existing WAM-GRU pipeline can handle patch-latent-shaped tensors, not that DINO-WM-style dynamics are learned.

## Next Steps for Real DINO-WM

1. **Implement DINO-WM-style action-conditioned future latent predictor** (Phase A task 3 from `docs/DINOWM_SNN_WORLDMODEL_PLAN.md`)
2. **Download DINOv2 ViT-S/14** (`facebook/dinov2-small`) and cache patch features on real LIBERO data
3. **Build transition dataset** with explicit shapes: `z_context [B, T, P, D]`, `actions [B, T, A]`, `z_target [B, H, P, D]`
4. **Train ANN/Transformer baseline** with gradient descent on real patch features
5. **Evaluate** one-step patch latent error, multi-step drift, nearest-neighbor retrieval
