# Split Policy

Status: policy defined for G2.5. Split-aware real-data loading is not implemented yet.

## Split Source

Phase 1 uses an explicit repository-generated `split.json` saved with every run. If a LIBERO file provides official `mask/<split>` keys, those keys may be imported into `split.json`, but the resolved split file remains the source of truth for SNN-WAM runs.

Required split file fields:

```json
{
  "suite": "libero_spatial",
  "split_unit": "trajectory",
  "train": ["task_file:data/demo_0"],
  "val": ["task_file:data/demo_40"],
  "test": ["task_file:data/demo_45"],
  "heldout_initial_states": ["task_file:data/demo_45:init_state"],
  "seed": 0,
  "method": "deterministic_sorted_demo_ids"
}
```

## Split Unit

Default Phase-1 split unit is trajectory within task file:

- Sort `data/demo_*` keys numerically.
- Train: first 80 percent.
- Val: next 10 percent.
- Test: final 10 percent.
- Each trajectory id belongs to exactly one split.

Task-level held-out evaluation is a separate experiment setting and must use a different `split_unit`, such as `task`, with task names recorded explicitly.

## Forbidden

- Do not mix windows from one trajectory across train/val/test.
- Do not compute action, state, image, or latent normalization statistics on val/test trajectories.
- Do not select best checkpoints based on test split.
- Do not use reward, done, success, or episode outcome labels as model inputs.
- Do not use closed-loop held-out initial states for offline training windows.

## Held-Out Trajectories

For Phase 1, held-out trajectories are selected deterministically by sorted demo id unless an official split exists. Random split generation is allowed only if:

- seed is recorded in `split.json`;
- exact trajectory ids per split are saved;
- rerunning split generation produces the same file.

## Closed-Loop Initial States

Closed-loop initial states must be separated from offline training demonstrations:

- Training may use only train trajectories.
- Validation rollouts may use validation initial states.
- Final reported closed-loop success may use test initial states only after model selection is finished.
- Initial-state ids used in closed-loop evaluation must be saved in `eval_rollout.csv` and in the run `split.json`.

## Reproducibility Requirements

Every training or evaluation run must save:

- `split.json`
- `normalization_stats.json`
- `command.txt` or `command.sh`
- `git_commit.txt`
- `environment.txt`
- `notes.md`

No split result is reportable unless it is registered in `docs/RESULT_ARTIFACTS.md`.
