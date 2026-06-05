# Official DINO-WM Upstream Reproduction

Status: prepared but not yet reproduced in this workspace.

As of 2026-06-05:

- official repo is cloned at `external/dino_wm`;
- upstream commit is `0a9492fa12044b852ae9e001cc74604b79c8bb0c`;
- official PointMaze data is downloaded and extracted under
  `data/dino_wm/point_maze`;
- `data/dino_wm/point_maze.zip` sha256 is
  `6c48ccf22c90b9af8dcf0e2cd70849aec8dd8e214ac5f1f09552bf8bc9494acc`;
- extracted PointMaze data is about 29G and contains 2003 files;
- official `dino_wm` conda environment is installed after adding missing
  `cmake` for `egl-probe`;
- core training imports pass in `/home/zhan_shaoji/miniconda3/envs/dino_wm`;
- full GPU training is blocked on this host because official
  `torch==2.3.0+cu121` cannot execute CUDA kernels on the local RTX 5060 Ti
  (`sm_120`);
- a separate host-compatible environment,
  `/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128`, is cloned from the
  official environment and upgraded to `torch==2.8.0+cu128`,
  `torchvision==0.23.0+cu128`, and `torchaudio==2.8.0+cu128`;
- the `dino_wm_cu128` environment passes core dependency imports and a GPU smoke
  test on the RTX 5060 Ti: CUDA is available, device capability is `(12, 0)`,
  and a minimal CUDA tensor operation succeeds;
- a minimal official-code train smoke using `dino_wm_cu128` reached CUDA, wandb
  offline setup, and PointMaze data loading, then failed while constructing the
  DINOv2 encoder because upstream DINO-WM calls `torch.hub.load` on the current
  DINOv2 `main` branch from a Python 3.9 environment;
- planning is not ready until MuJoCo 2.1 runtime exists at
  `~/.mujoco/mujoco210` and `LD_LIBRARY_PATH` is configured.

The current strict-official prepared evidence package is:

```text
results/upstream/official_dinowm_pointmaze_preflight_20260605_env_ready/
```

It records a valid upstream command path using the official pinned environment,
but it is not a training result.

The current host-compatible prepared evidence package is:

```text
results/upstream/official_dinowm_pointmaze_preflight_20260605_cu128/
```

It uses the unmodified upstream code and official data, but a newer PyTorch/CUDA
stack. Treat it as the local execution route, not as strict official-environment
evidence.

The current train-smoke evidence package is:

```text
results/upstream/official_dinowm_pointmaze_train_smoke_20260605_cu128/
```

Status: `execution_failed`. This is useful diagnostic evidence, not a model
result and not a reproduction.

## Source Of Truth

- Official repo: https://github.com/gaoyuezhou/dino_wm
- Official data/checkpoints: https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28
- Project page: https://dino-wm.github.io/
- Paper: https://arxiv.org/abs/2411.04983

The upstream README defines the required dataset layout under `DATASET_DIR`:

```text
DATASET_DIR/
  point_maze/
  pusht_noise/
  wall_single/
  deformable/
```

## Priority Target

Use PointMaze first. It is an official supported environment and matches the
requested smoke-to-full reproduction command:

```bash
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
```

For stricter reproducibility, this repository's helper additionally records
`num_pred=1`, `training.seed`, `ckpt_base_path`, and deterministic Hydra output
directories without changing the intended upstream task.

## Helper Script

Prepare a reproducibility package without launching training:

```bash
python scripts/reproduce_official_dinowm_upstream.py \
  --upstream_dir /path/to/dino_wm \
  --dataset_dir /path/to/data \
  --env point_maze \
  --frameskip 5 \
  --num_hist 3 \
  --num_pred 1 \
  --stage preflight
```

Run upstream training only after the official repo, conda environment, Mujoco,
and dataset are present:

```bash
python scripts/reproduce_official_dinowm_upstream.py \
  --upstream_dir /path/to/dino_wm \
  --dataset_dir /path/to/data \
  --env point_maze \
  --frameskip 5 \
  --num_hist 3 \
  --num_pred 1 \
  --stage train \
  --execute
```

Run CEM planning from the trained model:

```bash
python scripts/reproduce_official_dinowm_upstream.py \
  --upstream_dir /path/to/dino_wm \
  --dataset_dir /path/to/data \
  --env point_maze \
  --stage plan \
  --execute
```

The helper writes `command.sh`, `summary.json`, `sources.json`,
`environment.txt`, and `notes.md` under `results/upstream/...`.

The helper exports `WANDB_MODE=offline` by default so train/plan smoke runs do
not require interactive wandb login or network logging.

## Host-Specific Blockers

The official environment pins `torch==2.3.0+cu121`. On this host, CUDA sees an
RTX 5060 Ti with compute capability `sm_120`, but a minimal CUDA tensor
operation fails with:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Strict upstream reproduction therefore needs one of these routes:

- run the official environment unchanged on a GPU supported by
  `torch==2.3.0+cu121` (`sm_90` or older);
- run only CPU-level smoke checks here, not full training;
- create a clearly labeled hardware-compatible environment with newer
  PyTorch/CUDA 12.8+ support, and keep it separate from "strict official"
  reproduction evidence.

The third route is prepared on this host as `dino_wm_cu128`. Use it to decide
whether the official code/data path can proceed locally, while keeping the
PyTorch version difference visible in every result package.

There is also an upstream dependency reproducibility blocker that is independent
of the local GPU. Upstream DINO-WM constructs its encoder with:

```python
torch.hub.load("facebookresearch/dinov2", name)
```

Because no DINOv2 code revision is pinned, a fresh run on 2026-06-05 downloads
the current DINOv2 `main` branch. That code now contains Python 3.10-style type
syntax such as `float | None`, while the official DINO-WM environment pins
Python 3.9.19. The train smoke therefore fails at encoder construction with:

```text
TypeError("unsupported operand type(s) for |: 'type' and 'NoneType'")
```

Do not launch full upstream training until this is resolved by an explicitly
recorded DINOv2 code revision or by a separately labeled Python-3.10 diagnostic
environment.

Planning for PointMaze imports `d4rl`, which imports `mujoco_py`; this requires
MuJoCo 2.1 installed separately. Set `D4RL_DATASET_DIR` to a writable path if
running inside a restricted filesystem.

## Recommended Execution Path

1. Keep the official `dino_wm` environment unchanged as the strict reference.
2. Use `dino_wm_cu128` for local GPU smoke because it is the only verified CUDA
   route on this RTX 5060 Ti host.
3. Resolve the unpinned DINOv2 hub dependency before any full training:
   prefer pinning a Python-3.9-compatible DINOv2 GitHub revision and recording it
   in the result package; use a Python-3.10/cu128 environment only as a clearly
   labeled diagnostic route.
4. Re-run an official-code train smoke, with limited rollout count and one
   epoch, and record it under `results/upstream/..._train_smoke`.
5. If train smoke completes and writes a checkpoint, run a minimal `plan.py`
   import/config smoke only after MuJoCo 2.1 is installed.
6. Only after upstream train and plan evidence exists, proceed to the local
   minimal equivalent baseline. Do not proceed to LIBERO or SNN before that.

## Acceptance Evidence

Do not mark "DINO-WM reproduced upstream" until all of these exist:

- official repo commit hash;
- official dataset path with `point_maze` or `pusht_noise`;
- upstream `train.py` command and logs;
- upstream checkpoint under `ckpt_base_path/outputs/<model_name>`;
- documented PyTorch/CUDA/MuJoCo compatibility route;
- upstream `plan.py` command;
- planning outputs under upstream `plan_outputs`;
- summary noting planner, `n_evals`, `goal_H`, goal source, and pass/fail status.

This evidence is separate from this repository's DWM-G3/G4 baseline gates.
