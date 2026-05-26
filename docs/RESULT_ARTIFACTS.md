# Result Artifacts

This registry is the gate between raw experiment outputs and scientific claims. A result may not be cited in docs, slides, reports, or paper drafts unless it appears here.

| Artifact ID | Run ID | Result Files | Config | Commit | Environment | Seeds | Evaluation Split | Command | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R-000 | template | `results/runs/<run_id>/metrics.csv` | `results/runs/<run_id>/config.yaml` | `results/runs/<run_id>/git_commit.txt` | `results/runs/<run_id>/environment.txt` | `0,1,2` | `test` | `results/runs/<run_id>/command.sh` | Template row only. |

## Required Fields

- `Artifact ID`: Stable ID used by `docs/CLAIMS_LEDGER.md`.
- `Run ID`: Directory name under `results/runs/`.
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
