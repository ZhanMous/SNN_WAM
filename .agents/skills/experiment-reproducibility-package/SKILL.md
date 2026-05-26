---
name: experiment-reproducibility-package
description: Use when creating, auditing, or repairing experiment outputs, configs, checkpoints, logs, result tables, RESULT_ARTIFACTS.md, environment records, or reportable evidence packages.
---

# Experiment Reproducibility Package

A result is not reportable unless another run can identify exactly what code, data, config, command, split, seed, and checkpoint produced it.

## Required files per run

Every run directory should include:

```text
config.yaml
command.txt
git_commit.txt
environment.txt or pip_freeze.txt
metrics.csv
best.pt or checkpoint.pt when model training is involved
eval_rollout.csv when closed-loop evaluation is involved
summary.json or summary.md
notes.md
```

For failure analysis:

```text
failure_videos/
figures/
tables/
```

## Result directory naming

Prefer stable names:

```text
results/runs/YYYYMMDD_HHMM_<suite>_<model>_<short_goal>_<seed>/
```

Avoid overwriting reportable runs. Debug/smoke runs should live under:

```text
results/smoke/
results/debug/
```

## RESULT_ARTIFACTS.md contract

`docs/RESULT_ARTIFACTS.md` should have one entry per reportable result:

```markdown
## artifact_id

Status: reportable / preliminary / invalid / superseded
Run path:
Commit:
Config:
Command:
Dataset/split:
Checkpoint:
Metrics CSV:
Rollout CSV:
Seeds:
Main numbers:
Known limitations:
Used in claims:
```

Do not cite a run in paper/report docs unless it appears here.

## Anti-staleness audit

When updating summaries:

- Check that all referenced paths exist.
- Check that checkpoint metadata matches summary metadata.
- Check that metrics were produced by the same checkpoint being cited.
- Check that clean and robustness results use the intended base checkpoint.
- Mark old overwritten or mismatched runs as `invalid` or `superseded`, not silently corrected.

## Reproducibility script

Create or maintain:

```bash
scripts/check_result_artifacts.py
scripts/quality_gate.sh
```

The checker should verify:

- Path existence.
- Required files exist.
- Commit recorded.
- Metrics CSV has required columns.
- No reportable artifact points to `results/debug` or `results/smoke` unless explicitly allowed.

## Review output format

Return:

- PASS / FAIL / PASS WITH RISKS.
- Missing artifacts.
- Stale or inconsistent references.
- Commands run.
- Which results are reportable.
- Which results must not be cited.
