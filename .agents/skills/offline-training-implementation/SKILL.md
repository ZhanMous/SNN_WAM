---
name: offline-training-implementation
description: Use when implementing or modifying offline training for MLP, GRU, WAM-GRU, WAM-SNN, frozen encoders, losses, configs, checkpoints, or small smoke-run training scripts.
---

# Offline Training Implementation

This skill implements the phase-1 training path without jumping prematurely into full VLA, hardware, or long ES optimization.

## Mandatory development order

Follow this order unless the user explicitly overrides it:

1. Dataset inspection and trajectory window tests.
2. Minimal MLP action baseline.
3. GRU temporal baseline.
4. Future latent prediction added to GRU.
5. SNN temporal adapter.
6. Closed-loop evaluation.
7. Robustness and ES post-training.

Do not implement SNN before MLP/GRU data and training contracts are stable.

## Required config fields

Every training config should include:

```yaml
experiment:
  name: string
  seed: int
  tags: []
data:
  suite: libero_spatial_or_object
  dataset_root: path_or_env
  history_len: int
  action_horizon: int
  future_horizon: int
  image_size: int
  split: train_val_test_definition
model:
  visual_encoder: frozen_resnet_or_clip_or_stub
  text_encoder: frozen_clip_text_or_stub
  temporal_adapter: mlp_or_gru_or_snn_lif
  hidden_dim: int
training:
  batch_size: int
  epochs: int
  optimizer: adamw
  lr: float
  lambda_action: float
  lambda_future: float
  lambda_spike: float
  grad_clip_norm: optional_float
output:
  output_dir: path
  save_best_by: metric_name
```

## Required training outputs

Training must save:

- `config.yaml` copy.
- `command.txt` or `run_command.sh`.
- `git_commit.txt`.
- `environment.txt` or `pip_freeze.txt`.
- `metrics.csv` with epoch-level rows.
- `best.pt` and optionally `last.pt`.
- `notes.md` with known limitations.

## Loss contract

Default total loss:

```text
L_total = lambda_action * L_action + lambda_future * L_future + lambda_spike * L_spike
```

Rules:

- If `lambda_future=0`, future latent head may be disabled or ignored, but output schema must stay documented.
- If not using SNN, `L_spike` must be zero or absent, never a fake value.
- Loss components must be logged separately.

## Smoke training

Before long training, add a smoke command:

```bash
python src/train/train_offline.py --config configs/smoke_gru.yaml --max_steps 5 --output_dir results/smoke/gru
```

Smoke success requires:

- Forward pass completes.
- Loss is finite.
- Metrics CSV is written.
- Checkpoint is saved if configured.
- No GPU-only assumption unless documented.

## Tiny overfit test

When feasible, add a tiny-batch overfit test or script:

- Use 1-4 synthetic or tiny real samples.
- Train for a few steps.
- Confirm training loss decreases or gradients are nonzero.

## Implementation output format

After changes, report:

- Models/configs added.
- Loss components implemented.
- Output artifacts saved.
- Commands run.
- Runtime risks.
- Scientific risks.
