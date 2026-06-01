# Repository Reconnaissance Report

**Date:** 2026-06-01
**Branch:** `dinowm_snn_worldmodel`
**Commit:** `da7afb1` (Add top-level DINO-WM SNN scientific plan)

---

## 1. Repository Structure

```
SNN_WAM/
├── .agents/skills/          # 10 Codex agent skill definitions
├── .claude/                 # Claude settings (permissions, local overrides)
├── configs/                 # YAML experiment configs
│   ├── smoke/               # 4 smoke-test configs (action-only, BC-GRU, WAM-GRU)
│   ├── diagnostics/         # 2 diagnostic configs (BC-GRU proprio, WAM-GRU overfit)
│   └── reportable/          # 2 reportable configs (WAM-GRU w/ and w/o future latent)
├── docs/                    # 27 markdown documents (plans, contracts, audits, reports)
├── latents/                 # Pre-extracted DINOv2 latent HDF5 files
├── results/                 # Experiment artifacts
│   ├── runs/                # Training run results (config, summary, metrics, notes)
│   ├── smoke/               # Smoke test results
│   ├── diagnostics/         # Diagnostic results
│   ├── g6_*/g7_*/...g11_*/  # Gate-specific experiment results
│   ├── inspections/         # LIBERO data inspection JSON reports
│   ├── figures/             # Figure outputs (gitignored)
│   └── tables/              # Table outputs
├── scripts/                 # 21 utility/inspection/smoke scripts
├── splits/                  # Episode-level train/val/test split JSON
├── src/
│   ├── data/                # Dataset, split selection, normalization
│   ├── models/              # Model definitions (MLP, GRU, WAM-GRU, encoders, heads)
│   ├── train/               # Training and offline eval entry points
│   ├── eval/                # Rollout eval, diagnostics, gate-specific eval
│   └── utils/               # Utilities
├── tests/                   # 27 test files
├── AGENTS.md                # Agent workflow instructions
├── README.md                # Project overview (Chinese)
├── environment.yml          # Conda environment spec
└── pytest.ini               # Pytest markers
```

---

## 2. How to Run Current Training/Evaluation

### Environment Setup
```bash
conda env create -f environment.yml   # python=3.10, torch, numpy, h5py, pyyaml, pytest, tqdm
```

### Mock Dry-Run (no real data required)
```bash
python -m src.train.train_offline --config configs/smoke/libero_spatial_action_only_smoke.yaml
```

### Real Training (requires LIBERO HDF5 + pre-extracted DINOv2 latents)
```bash
python -m src.train.train_offline --config configs/reportable/libero_spatial_wam_gru_dinov2s_future.yaml
```

### Offline Evaluation
```bash
python -m src.train.eval_offline --config <run_config.yaml> --checkpoint <path_to_checkpoint.pt>
```

### Closed-Loop LIBERO Rollout
```bash
python -m src.eval.eval_rollout_libero --config <config.yaml> --checkpoint <checkpoint.pt>
```

### Smoke Scripts (in `scripts/`)
```bash
python scripts/extract_dinov2_latents.py   # Extract DINOv2 latents from LIBERO images
# Plus ~20 other inspection/smoke scripts
```

---

## 3. Where Key Components Live

### DINO/DINO-WM Encoder
- **Source:** `src/models/encoders.py` — `DINOv2VisualEncoder` class
  - Uses `facebook/dinov2-small` via HuggingFace transformers
  - Lazy-loads model on first forward pass
  - Produces CLS token embeddings `[B, 384]`
  - Pinned revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`
  - **Note:** Current implementation uses CLS tokens only; DINO-WM-style spatial patch features are planned but not yet implemented.
- **Latent extraction:** `scripts/extract_dinov2_latents.py`
- **Pre-extracted latents:** `latents/libero_spatial/dinov2_vits14/`

### Dynamics Predictor (World Model)
- **Source:** `src/models/temporal_gru.py` — `TemporalGRUWAMModel`
  - Action-conditioned world model: `action_history + z_t -> pred_actions + pred_future_latents`
  - Fusion layer for action-latent interaction
  - Separate action head and future latent head
  - Optional split gripper head support
- **Heads:** `src/models/heads.py`
  - `ActionChunkHead` — projects to `[B, H, A]` action chunks
  - `FutureLatentChunkHead` — projects to `[B, H, Z]` future latents
  - `SplitActionGripperHead` — splits continuous actions + categorical gripper logits

### Planner / Action Sequence Optimization
- **Status:** Not yet implemented. The plan (DINOWM_SNN_WORLDMODEL_PLAN.md) describes using the world model for action sequence optimization via ES/EGGROLL-style low-rank optimization or gradient training, but no planner code exists.

### SNN World Model
- **Status:** Not yet implemented. The `snn_lif` adapter is declared in config validation (`ALLOWED_TEMPORAL_ADAPTERS` in `src/utils/config.py`) and a placeholder config exists (`configs/libero_spatial_snn_lif.yaml`), but no actual SNN model code exists in `src/models/`.

### Configs
- **Main configs:** `configs/*.yaml` (MLP, GRU, SNN-LIF placeholders)
- **Smoke configs:** `configs/smoke/` (4 files)
- **Diagnostic configs:** `configs/diagnostics/` (2 files)
- **Reportable configs:** `configs/reportable/` (2 files)
- **Config validation:** `src/utils/config.py` — validates YAML against schema, allows temporal adapters: `mlp`, `gru`, `bc_gru`, `wam_gru`, `snn_lif`

---

## 4. Existing Tests

**27 test files** in `tests/`:

| Category | Files |
|---|---|
| **Core model tests** | `test_temporal_gru.py`, `test_temporal_mlp.py`, `test_encoders.py`, `test_dinov2_encoder.py` |
| **Data pipeline** | `test_trajectory_window.py`, `test_split_normalization_policy.py` |
| **Training/eval** | `test_train_offline.py`, `test_eval_offline.py`, `test_metrics.py` |
| **Infrastructure** | `test_config.py`, `test_experiment_io.py`, `test_repository_contract.py`, `test_dependency_policy.py`, `test_environment_workflow.py` |
| **LIBERO-specific** | `test_eval_rollout_libero.py`, `test_libero_bootstrap_gate.py`, `test_libero_smoke_scripts.py`, `test_inspect_libero_data.py`, `test_inspect_libero_demo.py` |
| **Gate-specific** | `test_g6_representation_bottleneck.py` through `test_g11_autoregressive_stabilization.py` |
| **Diagnostics** | `test_open_loop_diagnostics.py`, `test_overfit_diagnostics.py` |

Run all tests:
```bash
pytest -v
```

Run smoke-only tests:
```bash
pytest -v -m "not optional"
```

---

## 5. Missing Pieces for This Project

Based on the plan (`docs/DINOWM_SNN_WORLDMODEL_PLAN.md`) and current codebase:

### Critical Missing Components
1. **Spatial patch DINOv2 encoder** — Current `DINOv2VisualEncoder` only returns CLS tokens `[B, 384]`. The DINO-WM paradigm requires spatial patch features `[B, N_patches, D]` (e.g., 256 patches × 384 for ViT-S/14).
2. **SNN-LIF temporal adapter** — Declared in config but no model implementation. Needs `spikingjelly` or custom LIF neuron dynamics.
3. **Action-conditioned latent dynamics model** — `TemporalGRUWAMModel` exists but is GRU-based, not SNN-based. The core SNN world model needs to predict `z_{t+k}` from `z_t, a_t` in a spike-based latent space.
4. **Latent-to-image decoder** — For visualization and qualitative evaluation of predicted futures.
5. **Planner / action sequence optimizer** — ES/EGGROLL or gradient-based planning over the world model.
6. **SNN-specific training utilities** — Surrogate gradient support, membrane potential tracking, spike rate regularization.

### Infrastructure Gaps
7. **Spatial patch data pipeline** — Current `TrajectoryWindowDataset` handles CLS-level latents; needs to support patch-level `[B, N, D]` latents.
8. **DINO-WM-specific evaluation** — Metrics for spatial prediction quality (per-patch MSE, SSIM, etc.).
9. **ES/EGGROLL optimization framework** — For comparison against gradient-based planning.

### Data Gaps
10. **LIBERO spatial patch latents** — Pre-extracted CLS latents exist in `latents/`, but spatial patch latents (much larger) are not extracted yet.

---

## 6. Suggested Minimal First Runnable Command

### Quick Sanity Check (no data needed)
```bash
# Run all tests (validates current infrastructure is intact)
pytest -v --tb=short
```

### Smoke Training with Mock Data
```bash
# Mock dry-run of the WAM-GRU config (uses synthetic data)
python -m src.train.train_offline \
  --config configs/smoke/libero_spatial_wam_gru.yaml
```

### Real Training (if LIBERO data is available)
```bash
# Step 1: Extract DINOv2 latents (if not already done)
python scripts/extract_dinov2_latents.py

# Step 2: Train WAM-GRU with DINOv2-S and future latent loss
python -m src.train.train_offline \
  --config configs/reportable/libero_spatial_wam_gru_dinov2s_future.yaml
```

### First New-Route Work (DINO-WM SNN world model)
The recommended starting point is **Phase A** from `docs/DINOWM_SNN_WORLDMODEL_PLAN.md`:
1. Implement spatial patch extraction in `DINOv2VisualEncoder` (return `patch_tokens` instead of `cls_token`)
2. Update `TrajectoryWindowDataset` to handle `[B, N, D]` latent shapes
3. Register a new `snn_wm` model in `src/models/registry.py`
4. Train on spatial patch latents with MSE future prediction loss
5. Evaluate with patch-level metrics
