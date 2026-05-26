---
name: closed-loop-rollout-eval
description: Use when implementing or reviewing LIBERO closed-loop rollout, policy wrappers, fixed initial states, episode CSVs, success-rate computation, video recording, or failure analysis.
---

# Closed-loop Rollout Evaluation

Offline action prediction is not enough. Closed-loop rollout is the key evidence because actions change the next observation.

## Evaluation contract

A rollout evaluator must specify:

- LIBERO suite and task IDs/names.
- Number of episodes.
- Initial-state selection method.
- Max steps per episode.
- Action normalization/denormalization.
- Observation preprocessing.
- Whether SNN state persists within episode.
- Seed handling.
- Success criterion.

## Required outputs

For every rollout run, save:

- `eval_rollout.csv` with one row per episode.
- `summary.json` with aggregate success rate and settings.
- `config.yaml` or link to training config.
- `checkpoint_path.txt`.
- `git_commit.txt`.
- `command.txt`.
- `failure_videos/` or key frames for failures when `record_video=true`.
- `notes.md` listing failure modes and known evaluator limitations.

Suggested `eval_rollout.csv` columns:

```text
run_id,model,checkpoint,suite,task_id,task_name,episode_id,init_state_id,seed,success,steps,total_reward,terminated,truncated,failure_reason,video_path
```

## Policy wrapper rules

- The same model checkpoint must use the same preprocessing as training.
- Do not teacher-force actions during rollout.
- Do not peek at demonstration actions.
- Do not use future observations.
- If action chunking is used, document whether the policy executes the first action, receding horizon, or an averaged/smoothed action.

## Fair comparison rules

When comparing models:

- Use identical task IDs.
- Use identical initial states.
- Use identical seeds.
- Use identical max steps.
- Use identical action postprocessing.
- Report the number of episodes.

## Failure analysis

Classify failures when possible:

- Grasp miss.
- Object collision.
- Wrong object/location.
- Early termination.
- Oscillation/jitter.
- Slow but plausible.
- Environment/evaluator error.

Do not hide environment failures. Mark them separately.

## Smoke rollout

Before full evaluation, run:

- One task.
- One or five initial states.
- Small max steps.
- Video off first, then on.

Success is not required for smoke; no crash and a valid CSV are required.

## Review output format

Return:

- PASS / FAIL / PASS WITH RISKS.
- Whether demonstration leakage is impossible.
- Whether comparison is fair.
- Output artifact paths.
- Failure modes observed.
