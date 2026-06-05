# Official DINO-WM Upstream Reproduction

This package records commands for the unmodified upstream DINO-WM repo. It is not evidence of reproduction until `summary.json` is accompanied by official training metrics/checkpoints and planning outputs from `plan.py`.

Required success evidence:
- official repo commit
- DATASET_DIR with official task data
- train.py logs/checkpoint under ckpt_base_path/outputs/<model_name>
- plan.py outputs under upstream plan_outputs

## 2026-06-05 Train Smoke Status

Status: execution_failed.

The command reached CUDA device selection, wandb offline setup, and PointMaze
data loading with `env.dataset.n_rollout=2`. It failed while constructing the
DINOv2 encoder because upstream DINO-WM calls
`torch.hub.load("facebookresearch/dinov2", name)`, which downloads the current
DINOv2 `main` branch. The downloaded code uses Python 3.10-style union type
syntax, but the DINO-WM environment is Python 3.9.19.

Observed error:

```text
TypeError("unsupported operand type(s) for |: 'type' and 'NoneType'")
```

This package is diagnostic evidence only. It is not a model result and must not
be cited as upstream DINO-WM reproduction.
