# Resource-Limited Full-Data Config Audit

Status: prepared only; no train or plan execution was launched.

This audit was generated with upstream Hydra `--cfg job` using the unmodified
upstream DINO-WM entrypoints. It confirms the local fallback path keeps full
PointMaze data while reducing host resource pressure.

## Train

Command shape:

```text
/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin/python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3 num_pred=1 training.seed=0 ckpt_base_path=... hydra.run.dir=... hydra.sweep.dir=... training.batch_size=1 env.num_workers=0 training.num_reconstruct_samples=1 training.reconstruct_every_x_batch=999999 --cfg job
```

Resolved key fields:

```text
training.epochs: 100
training.batch_size: 1
training.reconstruct_every_x_batch: 999999
training.num_reconstruct_samples: 1
env.dataset.n_rollout: null
env.num_workers: 0
has_decoder: true
```

## Plan

Command shape:

```text
/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin/python plan.py model_name=point_maze_official_frameskip5_hist3_seed0 n_evals=5 planner=cem goal_H=5 goal_source=random_state planner.opt_steps=30 ckpt_base_path=... seed=0 --cfg job
```

Resolved key fields:

```text
n_evals: 5
goal_H: 5
planner.opt_steps: 30
```

## Interpretation

This is the recommended local diagnostic route if work must continue on this
WSL host. It is not the strict default upstream reproduction because
`training.batch_size`, `env.num_workers`, and reconstruction sampling differ
from the audited default full-run config.
