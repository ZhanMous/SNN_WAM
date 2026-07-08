# Official DINO-WM Upstream Reproduction

This package records commands for the unmodified upstream DINO-WM repo. It is not evidence of reproduction until `summary.json` is accompanied by official training metrics/checkpoints and planning outputs from `plan.py`.

Required success evidence:
- official repo commit
- DATASET_DIR with official task data
- train.py logs/checkpoint under ckpt_base_path/outputs/<model_name>
- plan.py outputs under upstream plan_outputs

## 2026-06-05 Plan Smoke Status

Status: executed.

This smoke run used the checkpoint from:

```text
results/upstream/official_dinowm_pointmaze_train_smoke_20260605_cu128_dinov2b483/
```

It ran official `plan.py` with:

```text
n_evals=1
planner=cem
goal_H=5
goal_source=random_state
planner.opt_steps=1
```

Additional host-compatibility environment variables were required:

```text
PATH=/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin:...
LD_LIBRARY_PATH=/home/zhan_shaoji/.mujoco/mujoco210/bin:/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/lib:...
MUJOCO_GL=osmesa
D4RL_DATASET_DIR=/tmp/d4rl
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

The run loaded the smoke checkpoint, created plan targets, executed one CEM
planning/evaluation pass, and wrote upstream outputs under:

```text
external/dino_wm/plan_outputs/20260605152413_point_maze_official_frameskip5_hist3_seed0_gH5/
```

Observed smoke metrics are recorded in `metrics.csv`; success_rate was 0.0 for
this one-episode smoke. This is expected to be weak because the model was trained
for only 1 epoch on 2 rollouts and the planner used only 1 optimization step.
This package proves the official train->plan path can execute locally, but it is
not a full DINO-WM reproduction or planning performance result.
