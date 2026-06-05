# Official DINO-WM Upstream Reproduction

This package records commands for the unmodified upstream DINO-WM repo. It is not evidence of reproduction until `summary.json` is accompanied by official training metrics/checkpoints and planning outputs from `plan.py`.

Required success evidence:
- official repo commit
- DATASET_DIR with official task data
- train.py logs/checkpoint under ckpt_base_path/outputs/<model_name>
- plan.py outputs under upstream plan_outputs

## 2026-06-05 Train Smoke Status

Status: executed.

This smoke run used the unmodified upstream DINO-WM repo at commit
`0a9492fa12044b852ae9e001cc74604b79c8bb0c`, official PointMaze data, and the
host-compatible `dino_wm_cu128` environment. Because the official DINO-WM code
loads DINOv2 through an unpinned `torch.hub.load("facebookresearch/dinov2",
name)`, this package installs DINOv2 GitHub commit
`b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8` as an artifact-local
`TORCH_HOME/hub/facebookresearch_dinov2_main` cache.

Smoke overrides:

```text
training.epochs=1
training.batch_size=1
env.dataset.n_rollout=2
env.num_workers=0
debug=true
```

Observed final train/val losses are recorded in `metrics.csv`. Checkpoints were
written under:

```text
official_ckpts/outputs/point_maze_official_frameskip5_hist3_seed0/checkpoints/
```

This package proves the official train.py path can execute on this host with a
recorded DINOv2 code pin. It is not a full upstream DINO-WM reproduction because
it uses only 2 rollouts, 1 epoch, a newer PyTorch/CUDA stack, and no plan.py run.
