# Resolved Hydra Config Audit

Status: prepared; no training or planning execution was launched.

Date: 2026-06-05

## Train Config

Command audited:

```bash
/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin/python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3 num_pred=1 training.seed=0 ckpt_base_path=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_full_preflight_20260605_cu128_dinov2b483/official_ckpts hydra.run.dir=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_full_preflight_20260605_cu128_dinov2b483/official_ckpts/outputs/point_maze_official_frameskip5_hist3_seed0 hydra.sweep.dir=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_full_preflight_20260605_cu128_dinov2b483/official_ckpts/outputs/point_maze_official_frameskip5_hist3_seed0 --cfg job
```

Resolved key fields:

- `training.epochs: 100`
- `training.batch_size: 32`
- `frameskip: 5`
- `num_hist: 3`
- `num_pred: 1`
- `env.name: point_maze`
- `env.dataset.n_rollout: null`
- `env.dataset.data_path: ${oc.env:DATASET_DIR}/point_maze`
- `encoder.name: dinov2_vits14`
- `encoder.feature_key: x_norm_patchtokens`
- `predictor._target_: models.vit.ViTPredictor`
- `predictor.depth: 6`
- `predictor.heads: 16`
- `predictor.mlp_dim: 2048`

Interpretation: this is the official full PointMaze training configuration, not a smoke override.

## Plan Config

Command audited:

```bash
/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin/python plan.py model_name=point_maze_official_frameskip5_hist3_seed0 n_evals=5 planner=cem goal_H=5 goal_source=random_state planner.opt_steps=30 ckpt_base_path=/home/zhan_shaoji/code/SNN_WAM/results/upstream/official_dinowm_pointmaze_full_preflight_20260605_cu128_dinov2b483/official_ckpts seed=0 --cfg job
```

Resolved key fields:

- `model_name: point_maze_official_frameskip5_hist3_seed0`
- `model_epoch: latest`
- `seed: 0`
- `n_evals: 5`
- `goal_source: random_state`
- `goal_H: 5`
- `objective.alpha: 1`
- `objective.mode: last`
- `planner._target_: planning.cem.CEMPlanner`
- `planner.horizon: 5`
- `planner.topk: 30`
- `planner.num_samples: 300`
- `planner.opt_steps: 30`
- `planner.name: cem`

Interpretation: this matches the upstream README's trained-model planning example:
`python plan.py model_name=<model_name> n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30`.

`conf/plan_point_maze.yaml` is a separate official checkpoint-launch config. It defaults to a PointMaze-specific MPC-style planner wrapper and `n_evals=50`; it is useful for official pretrained checkpoints, but it is not the README's generic trained-model planning command.
