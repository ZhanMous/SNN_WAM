# Official DINO-WM Upstream Reproduction

This package records commands for the unmodified upstream DINO-WM repo. It is not evidence of reproduction until `summary.json` is accompanied by official training metrics/checkpoints and planning outputs from `plan.py`.

Required success evidence:
- official repo commit
- DATASET_DIR with official task data
- train.py logs/checkpoint under ckpt_base_path/outputs/<model_name>
- plan.py outputs under upstream plan_outputs

## 2026-06-05 Plan Smoke Status

Status: execution_failed.

The command attempted to run official `plan.py` against the successful train
smoke checkpoint with:

```text
n_evals=1
planner=cem
goal_H=5
goal_source=random_state
planner.opt_steps=1
```

It failed before checkpoint loading or CEM execution. The import chain enters
PointMaze, then `gym.envs.mujoco`, then `mujoco_py`, which requires a separately
installed MuJoCo 2.1 runtime at:

```text
/home/zhan_shaoji/.mujoco/mujoco210
```

This package records the current planning blocker only. It is not evidence of
DINO-WM planning reproduction.
