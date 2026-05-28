# Result Artifacts

This registry is the gate between raw experiment outputs and scientific claims. A result may not be cited in docs, slides, reports, or paper drafts unless it appears here.

| Artifact ID | Run ID | Result Files | Config | Commit | Environment | Seeds | Evaluation Split | Command | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R-000 | template | `results/runs/<run_id>/metrics.csv` | `results/runs/<run_id>/config.yaml` | `results/runs/<run_id>/git_commit.txt` | `results/runs/<run_id>/environment.txt` | `0,1,2` | `test` | `results/runs/<run_id>/command.sh` | Template row only. |
| R-G3A-001 | `g3a_real_action_only_smoke_seed0` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/metrics.csv`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/summary.json`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/checkpoint.pt`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/best.pt` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/config.yaml` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/git_commit.txt` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/environment.json`; `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/environment.txt` | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/seeds.txt` | train/val smoke split: 2 train trajectories, 1 val trajectory, 0 test trajectories | `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/command.sh` | Status: engineering smoke test, not scientific result. Stub image/text encoders; action-history MLP only; no WAM, VLA, SNN, GRU, or closed-loop claim. See `docs/ACTION_ONLY_SMOKE_REPORT.md`. |
| R-G3B-MLP-MOCK-001 | `g3b_mlp_mock_smoke_seed0` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/metrics.csv`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/summary.json`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/checkpoint.pt`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/best.pt`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/split.json`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/notes.md` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/config.yaml` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/git_commit.txt` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/environment.json`; `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/environment.txt` | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/seeds.txt` | mock dry-run train/val split; not real LIBERO evidence | `results/smoke/action_baselines/g3b_mlp_mock_smoke_seed0/command.sh` | Status: engineering smoke test, not scientific result. MLP action-only mock dry run, hidden_dim=256, parameter_count=80,412, trainable_parameter_count=80,412. |
| R-G3B-GRU-MOCK-001 | `g3b_gru_mock_smoke_seed0` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/metrics.csv`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/summary.json`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/checkpoint.pt`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/best.pt`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/split.json`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/notes.md` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/config.yaml` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/git_commit.txt` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/environment.json`; `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/environment.txt` | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/seeds.txt` | mock dry-run train/val split; not real LIBERO evidence | `results/smoke/action_baselines/g3b_gru_mock_smoke_seed0/command.sh` | Status: engineering smoke test, not scientific result. GRU action-only mock dry run, hidden_dim=256, parameter_count=210,716, trainable_parameter_count=210,716. Parameter count differs from MLP; this row is not a performance comparison. |

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
