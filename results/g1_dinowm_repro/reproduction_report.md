# G1 DINO-WM Minimal Reproduction Smoke Report

## Scope

- Original DINO-WM only (via closest existing infrastructure)
- No SNN
- No LIBERO port
- No ES / EGGROLL
- Smallest official environment/task/config found: `configs/smoke/g0_patch_latent_smoke.yaml` with `--dry_run --max_steps 1`

## Files Inspected

| File | Purpose |
|---|---|
| `README.md` | Official smoke commands and repository map |
| `docs/DINOWM_SNN_WORLDMODEL_PLAN.md` | Phase A tasks and DWM-G1 gate definition |
| `docs/RESULT_ARTIFACTS.md` | Artifact registry format and rules |
| `docs/CLAIMS_LEDGER.md` | Claim format and forbidden claims |
| `configs/smoke/g0_patch_latent_smoke.yaml` | G0 patch latent smoke config (closest to DINO-WM) |
| `configs/smoke/libero_spatial_action_only_smoke.yaml` | Action-only smoke config (alternative) |
| `src/train/train_offline.py` | Training entrypoint with --dry_run support |

## Selected Command

```bash
conda run -n snnwam-libero python src/train/train_offline.py \
  --config configs/smoke/g0_patch_latent_smoke.yaml \
  --dry_run --max_steps 1 \
  --output_dir results/g1_dinowm_repro/
```

**Why selected:** The `g0_patch_latent_smoke.yaml` is the only config that exercises the patch latent pipeline (DINOv2PatchEncoder → [B, T, P, D] tensors → WAM-GRU with future latent loss). It is the closest existing infrastructure to the DINO-WM data path. `--dry_run` uses deterministic mock data requiring no LIBERO installation. `--max_steps 1` limits to a single training step.

## Smoke-Scale Overrides

| Override | Rationale |
|---|---|
| `--dry_run` | Uses mock data, no LIBERO needed |
| `--max_steps 1` | Single training step, minimal compute |
| `epochs: 0` (in config) | Shape validation only, no convergence expected |
| `batch_size: 8` (in config) | Small batch for smoke |
| `--output_dir results/g1_dinowm_repro/` | Centralized artifact location |

## Intentionally Not Run

- Full DINO-WM training (no such code exists yet — Phase A task 3 not implemented)
- Real LIBERO data loading (requires LIBERO installation + dataset)
- DINOv2 patch feature extraction (requires `facebook/dinov2-small` download)
- Planning / action optimization sweeps
- Any SNN, ES, or EGGROLL code
- Multi-seed runs
- Full epoch training

## Artifacts

| File | Meaning |
|---|---|
| `config.yaml` | Copy of `g0_patch_latent_smoke.yaml` used for this run |
| `command.txt` | Exact command executed |
| `git_commit.txt` | Commit hash `da7afb1` + dirty state |
| `environment.txt` | Python 3.12.3, torch 2.12.0+cu130, numpy 1.26.4 |
| `run.log` | Full stdout/stderr of the smoke command |
| `metrics.csv` | Train/val metrics for 1 epoch (mock data) |
| `checkpoint.pt` | Model checkpoint after 1 step |
| `best.pt` | Best model checkpoint |
| `summary.json` | Full run metadata (model, data, normalization, metrics) |
| `split.json` | Mock train/val split |
| `seeds.txt` | Seed used (0) |
| `train.log` | Training log from the run |
| `normalization_stats.json` | Normalization statistics |
| `notes.md` | Limitations and blocked items |
| `summary.md` | This summary |
| `reproduction_report.md` | This report |

## Outcome

**PASS WITH RISKS**

The smoke command exited with code 0 and produced a complete artifact set including metrics.csv, checkpoint.pt, and summary.json. However:

1. All data is mock/synthetic — no real LIBERO trajectories loaded
2. The encoder is `smoke_time_index` (synthetic) — not real DINOv2 patch features
3. No DINO-WM-style action-conditioned future latent predictor exists in the codebase
4. The run validates plumbing (config → data → model → loss → metrics → checkpoint), not DINO-WM science

## Evidence

- `run.log`: Command exited with code 0, produced `run_dir=results/g1_dinowm_repro/20260601_071641_libero_spatial_wam_gru_g0_patch_latent_smoke_seed0`
- `metrics.csv`: action_mse=132.897 (train), 86.127 (val); future_latent_cosine_error=1.274 (train), 1.169 (val)
- `summary.json`: 286,780 parameters, WAM-GRU architecture, smoke_time_index encoder, mock split

## Limitations

- **Smoke-scale only.** This is not a benchmark reproduction.
- **Not scientific evidence.** Mock data, synthetic encoders, single step.
- **Does not validate DINO-WM.** No DINO-WM training loop exists. No real patch features. No spatial dynamics modeling.
- **Does not support claims about:** SNN, LIBERO, robustness, closed-loop robot performance, energy efficiency, generalization, planning.
- **Does not validate the DINO-WM mechanism.** The run tests that the existing WAM-GRU pipeline can handle patch-latent-shaped tensors, not that DINO-WM-style dynamics are learned.

## Next Steps

1. **Implement DINO-WM-style action-conditioned future latent predictor** (Phase A task 3 from `docs/DINOWM_SNN_WORLDMODEL_PLAN.md`)
2. **Download DINOv2 ViT-S/14** (`facebook/dinov2-small`) and cache patch features on real LIBERO data
3. **Build transition dataset** with explicit shapes: `z_context [B, T, P, D]`, `actions [B, T, A]`, `z_target [B, H, P, D]`
4. **Train ANN/Transformer baseline** with gradient descent on real patch features
5. **Evaluate** one-step patch latent error, multi-step drift, nearest-neighbor retrieval
