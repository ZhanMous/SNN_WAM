# Result Artifacts

This registry is the gate between raw experiment outputs and scientific claims. A result may not be cited in docs, slides, reports, or paper drafts unless it appears here.

## Smoke vs Reportable Artifacts

**Smoke artifacts** (under `results/smoke/`):
- Used for engineering validation, plumbing checks, and smoke tests
- May use dirty git state, stub encoders, mock data, or smoke latents
- Must NOT be cited as scientific evidence or reportable results
- Configs are in `configs/smoke/`

**Reportable artifacts** (under `results/runs/`):
- Used for scientific claims and paper results
- Require clean git state (`dirty=False` in `git_commit.txt`)
- Require `reproducibility.require_clean_git=true` in config
- Require real LIBERO data (not mock/smoke)
- Require frozen visual encoder (not stub)
- Must pass `scripts/preflight_reportable.py` before execution
- Configs are in `configs/`

## Artifact Registry

| Artifact ID | Run ID | Result Files | Config | Commit | Environment | Seeds | Evaluation Split | Command | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R-000 | template | `results/runs/<run_id>/metrics.csv` | `results/runs/<run_id>/config.yaml` | `results/runs/<run_id>/git_commit.txt` | `results/runs/<run_id>/environment.txt` | `0,1,2` | `test` | `results/runs/<run_id>/command.sh` | Template row only. |
| R-G3A-001 | `g3a_real_action_only_smoke_seed0` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/metrics.csv`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/summary.json`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/checkpoint.pt`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/best.pt` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/config.yaml` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/git_commit.txt` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/environment.json`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/environment.txt` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/seeds.txt` | train/val smoke split: 2 train trajectories, 1 val trajectory, 0 test trajectories | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/command.sh` | Status: engineering smoke test, not scientific result. Stub image/text encoders; action-history MLP only; no WAM, VLA, SNN, GRU, or closed-loop claim. See `docs/ACTION_ONLY_SMOKE_REPORT.md`. |
| R-G3B-MLP-MOCK-001 | `g3b_mlp_mock_smoke_seed0` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/metrics.csv`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/summary.json`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/checkpoint.pt`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/best.pt`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/split.json`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/notes.md` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/config.yaml` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/git_commit.txt` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/environment.json`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/environment.txt` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/seeds.txt` | mock dry-run train/val split; not real LIBERO evidence | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/command.sh` | Status: engineering smoke test, not scientific result. MLP action-only mock dry run, hidden_dim=256, parameter_count=80,412, trainable_parameter_count=80,412. |
| R-G3B-GRU-MOCK-001 | `g3b_gru_mock_smoke_seed0` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/metrics.csv`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/summary.json`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/checkpoint.pt`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/best.pt`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/split.json`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/notes.md` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/config.yaml` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/git_commit.txt` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/environment.json`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/environment.txt` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/seeds.txt` | mock dry-run train/val split; not real LIBERO evidence | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/command.sh` | Status: engineering smoke test, not scientific result. GRU action-only mock dry run, hidden_dim=256, parameter_count=210,716, trainable_parameter_count=210,716. Parameter count differs from MLP; this row is not a performance comparison. |
| R-G4-WAM-GRU-FUTURE-SMOKE-001 | `g4_wam_gru_future_smoke_seed0` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/metrics.csv`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/eval_offline.csv`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/train.log`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/summary.json`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/eval_summary.json`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/checkpoint.pt`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/best.pt`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/split.json`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/notes.md` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/config.yaml` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/git_commit.txt` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/environment.json`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/environment.txt`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/eval_environment.json` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/seeds.txt` | mock dry-run train/val split; not real LIBERO evidence; eval split `val` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/command.sh`; `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/eval_command.txt` | **Status: NOT REPORTABLE.** Engineering smoke test only. WAM-GRU with `lambda_future=1.0`, **smoke latents (not real frozen visual encoder latents)**, seed 0, max_steps=1. Recorded commit has `dirty=True`, which prevents reportable use. Offline val action_mse=86.12704032, future_latent_cosine_error=1.168717265. Must not be cited as real-data, closed-loop, robustness, success, or WAM evidence. |
| R-G4-WAM-GRU-NO-FUTURE-SMOKE-001 | `g4_wam_gru_no_future_smoke_seed0` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/metrics.csv`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/eval_offline.csv`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/train.log`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/summary.json`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/eval_summary.json`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/checkpoint.pt`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/best.pt`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/split.json`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/notes.md` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/config.yaml` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/git_commit.txt` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/environment.json`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/environment.txt`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/eval_environment.json` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/seeds.txt` | mock dry-run train/val split; not real LIBERO evidence; eval split `val` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/command.sh`; `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/eval_command.txt` | **Status: NOT REPORTABLE.** Engineering smoke test only. Same WAM-GRU architecture with `lambda_future=0.0`, **smoke latents (not real frozen visual encoder latents)**, seed 0, max_steps=1. Recorded commit has `dirty=True`, which prevents reportable use. Offline val action_mse=86.12520054, future_latent_cosine_error=1.24387157. Must not be cited as real-data, closed-loop, robustness, success, or WAM evidence. |
| R-G4-WAM-GRU-DINOV2S-REAL-OFFLINE-001 | `libero_spatial_wam_gru_dinov2s_future_seed0` | `<pending>` | `configs/reportable/libero_spatial_wam_gru_dinov2s_future.yaml` | `<pending>` | `<pending>` | `0,1,2` | `val` | `<pending>` | **Status: PENDING.** First reportable real-latent WAM-GRU ablation with future latent loss. Uses DINOv2 ViT-S/14 frozen latents (CLS token, dim=384, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`). Requires clean git, real LIBERO data, episode-level split, and preflight pass. Not yet executed. Metrics: global action MSE, per-horizon action MSE, per-dimension action MSE, future_latent_cosine_error, future_latent_mse, loss decomposition. |
| R-G4-WAM-GRU-DINOV2S-REAL-OFFLINE-002 | `libero_spatial_wam_gru_dinov2s_no_future_seed0` | `<pending>` | `configs/reportable/libero_spatial_wam_gru_dinov2s_no_future.yaml` | `<pending>` | `<pending>` | `0,1,2` | `val` | `<pending>` | **Status: PENDING.** First reportable real-latent WAM-GRU ablation without future latent loss. Uses DINOv2 ViT-S/14 frozen latents (CLS token, dim=384, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`). Requires clean git, real LIBERO data, episode-level split, and preflight pass. Not yet executed. Metrics: global action MSE, per-horizon action MSE, per-dimension action MSE, future_latent_cosine_error, future_latent_mse, loss decomposition. |

## Required Fields

- `Artifact ID`: Stable ID used by `docs/CLAIMS_LEDGER.md`.
- `Run ID`: Directory name under `results/runs/`, or under `results/smoke/`
  when the row is explicitly marked as engineering smoke only.
- `Result Files`: Concrete files under `results/`, not a broad directory.
- `Config`: The exact configuration file used for the run.
- `Commit`: The recorded git commit or dirty-state file.
- `Environment`: Python, OS, dependency, hardware, and runtime environment record.
- `Seeds`: Random seeds included in the artifact.
- `Evaluation Split`: Train/val/test/held-out task split.
- `Command`: Command used to produce the result.
- `Notes`: Link to `notes.md` or explain why it is absent.

## Rules

- Every registered result must include `config.yaml`, metrics, `git_commit.txt`, `environment.txt`, `seeds.txt`, `command.sh`, `split.json`, and `notes.md`.
- `checkpoint.pt` is required when a model checkpoint is applicable.
- A claim in `docs/CLAIMS_LEDGER.md` must reference one or more artifact IDs from this file.
- Template rows are examples, not scientific evidence.
- `R-G3A-001` is smoke-level engineering evidence only and must not be cited as
  a benchmark or scientific performance result.
- `R-G3B-MLP-MOCK-001` and `R-G3B-GRU-MOCK-001` are mock-data smoke checks only.
  They document shape, metric logging, checkpointing, and parameter counts; they
  must not be cited as real-data or benchmark results.
- `R-G4-WAM-GRU-FUTURE-SMOKE-001` and
  `R-G4-WAM-GRU-NO-FUTURE-SMOKE-001` are mock-data ablation smoke checks only.
  They document config parity, loss logging, checkpointing, and offline
  evaluator output; they must not be cited as real-data, closed-loop,
  robustness, or success-rate evidence.

## Reportability Rules

An artifact is **reportable** only if ALL of the following are true:

1. `git_commit.txt` records a commit where `dirty=False`.
2. The dataset type is `real-libero`, not `smoke` or `synthetic`.
3. The encoder type is `frozen-real`, not `stub` or `smoke`.
4. The run directory is under `results/runs/`, not `results/smoke/` or `results/debug/`.
5. The artifact is registered in this file with status `reportable` or `preliminary`.
6. The config has `reproducibility.require_clean_git=true`.
7. The experiment passed `scripts/preflight_reportable.py` before execution.

**Currently, no artifact is reportable.** All registered artifacts are engineering
smoke tests with deterministic mock data, smoke latents, and/or dirty git state.

## Preflight Check

Before running a reportable experiment, execute:

```bash
python scripts/preflight_reportable.py --config configs/your_config.yaml --artifact-id R-XXXX
```

This will fail closed if:
- Git working tree is dirty
- LIBERO_REPO_ROOT is not set
- LIBERO_DATASET_ROOT is not set
- Config does not have `reproducibility.require_clean_git=true`
- Output directory is under `results/smoke/` or `results/debug/`
- Artifact ID already exists in this registry
