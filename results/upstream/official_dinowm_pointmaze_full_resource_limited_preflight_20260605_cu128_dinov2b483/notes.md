# Official DINO-WM Upstream Reproduction

This package records commands for the unmodified upstream DINO-WM repo. It is not evidence of reproduction until `summary.json` is accompanied by official training metrics/checkpoints and planning outputs from `plan.py`.

Required success evidence:
- official repo commit
- DATASET_DIR with official task data
- train.py logs/checkpoint under ckpt_base_path/outputs/<model_name>
- plan.py outputs under upstream plan_outputs

## Resource-Adjusted Local Route

This package is a local fallback path after the audited default full train
attempt restarted the WSL VM. It keeps the unmodified upstream DINO-WM code,
official PointMaze data, full dataset selection (`env.dataset.n_rollout=null`),
host-compatible `dino_wm_cu128` environment, and artifact-local DINOv2 pin.

It intentionally changes train resource pressure with explicit Hydra overrides:

- `training.batch_size=1`
- `env.num_workers=0`
- `training.num_reconstruct_samples=1`
- `training.reconstruct_every_x_batch=999999`

This package is prepared only (`execute=false`). It is not a training result and
must not be called a strict default upstream reproduction.
