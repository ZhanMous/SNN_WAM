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
| R-G4-WAM-GRU-DINOV2S-REAL-OFFLINE-001 | `libero_spatial_wam_gru_dinov2s_future_seed0` | `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/metrics.csv`; `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/eval_offline.csv`; `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/summary.json`; `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/best.pt` | `configs/reportable/libero_spatial_wam_gru_dinov2s_future.yaml` | `f53bd11` (dirty=True: untracked output dirs) | `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/environment.json` | `0` | `val` | `results/runs/libero_spatial_wam_gru_dinov2s_future/20260528_081325_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_future_seed0/command.sh` | **Status: PRELIMINARY.** Real-latent WAM-GRU ablation with future latent loss (lambda_future=1.0). DINOv2 ViT-S/14 frozen latents (CLS token, dim=384, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`). 100 epochs, episode-level split (40 train, 5 val, 5 test). Eval val: action_mse=0.01910, future_latent_cosine_error=0.00113, future_latent_mse=0.10306. Git dirty due to untracked output directories (code at commit was clean). Single seed (0); seeds 1,2 pending. |
| R-G4-WAM-GRU-DINOV2S-REAL-OFFLINE-002 | `libero_spatial_wam_gru_dinov2s_no_future_seed0` | `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/metrics.csv`; `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/eval_offline.csv`; `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/summary.json`; `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/best.pt` | `configs/reportable/libero_spatial_wam_gru_dinov2s_no_future.yaml` | `f53bd11` (dirty=True: untracked output dirs) | `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/environment.json` | `0` | `val` | `results/runs/libero_spatial_wam_gru_dinov2s_no_future/20260528_084146_libero_spatial_wam_gru_libero_spatial_wam_gru_dinov2s_no_future_seed0/command.sh` | **Status: PRELIMINARY.** Real-latent WAM-GRU ablation without future latent loss (lambda_future=0.0). DINOv2 ViT-S/14 frozen latents (CLS token, dim=384, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`). 100 epochs, episode-level split (40 train, 5 val, 5 test). Eval val: action_mse=0.01841, future_latent_cosine_error=1.00373, future_latent_mse=5.47144. Git dirty due to untracked output directories (code at commit was clean). Single seed (0); seeds 1,2 pending. |
| R-G5-MLP-LIBERO-ROLLOUT-SMOKE-001 | `20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout` | `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/metrics.csv`; `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/best.pt`; `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/eval_rollout.csv`; `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/summary.json`; `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/compatibility_report.json`; `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/failure_videos/episode_0000.npy`; `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/notes.md` | `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/config.yaml` | `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/git_commit.txt` | `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/environment.json`; `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/environment.txt` | training seed `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/seeds.txt`; eval seed `20260529` | closed-loop LIBERO smoke: suite `libero_spatial`, task_id `0`, init_state_id `0`, max_steps `3`, one episode | `results/runs/20260528_020408_libero_spatial_mlp_libero_spatial_mlp_action_placeholder_seed0/eval_rollout/command.txt` | **Status: CLOSED-LOOP SMOKE, NOT REPORTABLE.** Real LIBERO environment smoke of an MLP checkpoint whose training config was `dry_run: true` with mock data. Compatibility passed (`mismatches=[]`, strict state_dict load). Episode result: 0/1 success, success_rate=0.0, failure_counts=`max_steps_reached: 1`, failure media saved. Must not be cited as model performance, robustness, fair MLP/GRU/WAM-GRU comparison, or future-latent evidence. See `docs/ROLL_OUT_FINDINGS.md`. |

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

**Currently, no artifact is fully reportable.** Two preliminary real-latent artifacts exist (R-G4-WAM-GRU-DINOV2S-REAL-OFFLINE-001/002) but have `dirty=True` due to untracked output directories. All other registered artifacts are engineering smoke tests with deterministic mock data, smoke latents, and/or dirty git state.

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
