# Action-Only Smoke Report

## Status

This is an engineering smoke test, not a scientific result.

This is not WAM. No future latent or future state prediction is implemented.
This is not VLA. The run does not train or evaluate a vision-language policy.
This is not SNN. No spiking module, spike rate, or neuromorphic claim is used.
This is not GRU. The temporal adapter is the action-history MLP baseline only.
This is not closed-loop evaluation.

## Purpose

The purpose is only to verify that the real-data action-only path can load a
small LIBERO subset, create a deterministic trajectory split, fit train-only
action normalization, calculate action MSE, save checkpoints, and write
reproducibility artifacts.

Stub image and text encoders are used. Images and language are not used as
scientific evidence in this smoke run.

## Data And Split

- Dataset root used: `/home/zhan_shaoji/data/libero/datasets`
- Suite: `libero_spatial`
- Config: `configs/libero_spatial_action_only_smoke.yaml`
- Split rule: deterministic sorted trajectory ids, then config limits.
- Train trajectories: 2
- Validation trajectories: 1
- Test trajectories: 0
- Train windows: 212
- Validation windows: 156
- Seed: 0

The run directory records exact trajectory ids in `split.json`.

## Metric Definition

`action_mse` compares predicted action chunks and demonstration action chunks
with shape `[B, H, A]`. It is a global mean over batch/window, horizon, and
action dimensions. Lower is better.

For this run, the training loss is in normalized action units because action
standardization was fit on train trajectories only. Reported `action_mse` is in
raw action units.

## Smoke Run

- Run path: `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0`
- Command: `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/command.sh`
- Metrics: `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/metrics.csv`
- Checkpoint: `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/checkpoint.pt`
- Best checkpoint: `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/best.pt`
- Summary: `results/smoke/action_only_mlp/g3a_real_action_only_smoke_seed0/summary.json`

Final epoch smoke numbers:

| Split | Action MSE | Units |
| --- | ---: | --- |
| train | 0.1330477918 | raw action units |
| val | 0.1519946751 | raw action units |

Training total loss decreased from `0.9880120439` to `0.5793314014` in
normalized action units.

## Claims Not Supported

- No claim that SNN improves performance.
- No claim that WAM improves future prediction.
- No claim that a vision-language policy works.
- No claim that closed-loop success is improved.
- No claim that the method generalizes on LIBERO.
- No benchmark claim or hyperparameter tuning claim.
