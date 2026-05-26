---
name: weekly-research-report
description: Use when generating or updating SNN-WAM weekly advisor reports, experiment logs, meeting summaries, progress memos, or next-week plans from code changes and result artifacts.
---

# Weekly Research Report

The report must be evidence-first, not narrative-first. It should help an advisor quickly see what worked, what failed, and what decision comes next.

## Inputs to inspect

Before writing, inspect if available:

- `docs/WEEKLY_REPORT.md`.
- `docs/EXPERIMENT_LOG.md`.
- `docs/RESULT_ARTIFACTS.md`.
- Recent `results/runs/*/summary.*`.
- Recent `metrics.csv` and `eval_rollout.csv`.
- `git log --oneline -n 10`.

## Weekly report structure

Use this structure:

```markdown
# Weekly Report: YYYY-MM-DD

## 1. This week's goal

## 2. Completed work

## 3. Evidence

| Artifact | Model/Setting | Key metric | Path | Status |

## 4. What we learned

## 5. Problems and risks

## 6. Next week's plan

## 7. One-sentence advisor summary
```

## Writing rules

- Prefer exact artifact paths and metric names.
- Distinguish smoke, preliminary, and reportable results.
- Do not hide failed experiments.
- Do not inflate results into paper claims.
- If no formal experiment ran, say so and report engineering progress.

## One-sentence advisor summary

Format:

```text
本周我完成了 X，并用 Y 证据确认 Z；目前主要风险是 R，下一步将做 N。
```

Example:

```text
本周我完成了 LIBERO trajectory window 数据读取与无未来泄漏测试，并跑通 GRU action-prediction smoke training；目前主要风险是 closed-loop action postprocessing 仍未验证，下一步将接入最小 rollout evaluator。
```

## Output format

When asked to write a report:

- Update the report file if editing the repo.
- Also return the one-sentence summary and top three risks.
