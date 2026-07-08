# WSL Crash Analysis

Status: interrupted by WSL VM restart; not a Python-level training failure.

Date: 2026-06-05

## What Happened

The full upstream PointMaze train command was launched with the audited official
configuration:

- `training.epochs: 100`
- `training.batch_size: 32`
- `env.dataset.n_rollout: null`
- `env.num_workers: 16`
- `has_decoder: true`
- `encoder.name: dinov2_vits14`
- `encoder.feature_key: x_norm_patchtokens`

The run reached CUDA initialization, wandb offline startup, and full PointMaze
dataset loading:

```text
[2026-06-05 16:11:34,447] Loading dataset from .../data/dino_wm/point_maze ...
Loaded 2000 rollouts
```

There is no epoch log, no checkpoint, and no Python traceback. The wrapper
`summary.json` was last written before execution status could be updated.

## System Evidence

- WSL uptime after the crash was about 1 minute, confirming VM restart.
- `journalctl` reported unclean shutdown and journal replacement.
- Windows System log showed WSL Hyper-V networking being recreated at
  approximately 16:08 and 16:15, matching the two WSL restarts.
- Linux `dmesg` after restart showed WSL 2.7.3.0 and dynamic memory cap around
  16200 MB.
- Current WSL memory: 15 GiB RAM, 4 GiB swap.
- Host memory from Windows: about 32 GiB physical RAM.
- GPU: RTX 5060 Ti with 8 GiB VRAM.
- User-level `.wslconfig` contains only `[wsl2]`; no explicit memory/swap limit
  is configured.

## Interpretation

This host is not stable for retrying the strict default full train command. The
failure happens after loading full data and before the first epoch/checkpoint,
with no Python exception. The most likely trigger is resource pressure or
WSL/NVIDIA driver instability from the combination of full PointMaze data,
`batch_size=32`, `num_workers=16`, decoder/image reconstruction path, CUDA, and
8 GiB VRAM inside WSL.

Do not keep retrying the same default full train command on this WSL instance.

## Recommended Next Paths

1. Strictest upstream route:
   run the official upstream command on a native Linux machine or remote server
   with enough RAM/VRAM and a GPU supported by the official environment.

2. Local host-compatible route:
   keep upstream code/data but lower resource pressure with explicit Hydra
   overrides such as `training.batch_size=1`, `env.num_workers=0`,
   `training.num_reconstruct_samples=1`, and possibly reduced reconstruction
   plotting. This must be labelled resource-adjusted and must not be called a
   strict default reproduction.

3. Diagnostic-only route:
   use official pretrained checkpoints for `plan.py` to validate planning
   plumbing. This does not satisfy "train.py reproduced upstream".
