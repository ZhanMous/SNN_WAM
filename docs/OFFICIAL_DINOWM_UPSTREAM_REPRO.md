# Official DINO-WM Upstream Reproduction

Status: upstream train/plan smoke runnable; full DINO-WM reproduction not yet complete.

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
- a minimal official-code train smoke using `dino_wm_cu128` and the unpinned
  current DINOv2 `main` branch reached CUDA, wandb offline setup, and PointMaze
  data loading, then failed while constructing the DINOv2 encoder because the
  downloaded DINOv2 code used Python 3.10 syntax from a Python 3.9 environment;
- DINOv2 GitHub commit `b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8`, the parent
  of the commit that introduced the incompatible `float | None` syntax, was
  validated as Python-3.9-loadable through Torch Hub;
- a second minimal official-code train smoke using `dino_wm_cu128`, official
  PointMaze data, and artifact-local `TORCH_HOME` pinning DINOv2 to
  `b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8` executed successfully for 1 epoch
  on 2 rollouts and wrote official checkpoints;
- MuJoCo 2.1 was installed at `/home/zhan_shaoji/.mujoco/mujoco210`; the
  `dino_wm_cu128` environment needed `mesalib`, `libgl-devel`, and `patchelf`
  to compile/import `mujoco_py`;
- a minimal official `plan.py` smoke against the train-smoke checkpoint executed
  with `n_evals=1`, `planner=cem`, `goal_H=5`, `goal_source=random_state`, and
  `planner.opt_steps=1`; it loaded the checkpoint, created plan targets, wrote
  upstream planning outputs, and reported success_rate `0.0` for the single
  smoke episode.

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

The failed unpinned train-smoke evidence package is:

```text
results/upstream/official_dinowm_pointmaze_train_smoke_20260605_cu128/
```

Status: `execution_failed`. This is useful diagnostic evidence, not a model
result and not a reproduction.

The current successful pinned-DINOv2 train-smoke evidence package is:

```text
results/upstream/official_dinowm_pointmaze_train_smoke_20260605_cu128_dinov2b483/
```

Status: `executed`. This proves the official `train.py` path can execute on this
host with a recorded DINOv2 code pin. It is still not a full reproduction because
it uses `env.dataset.n_rollout=2`, `training.epochs=1`, a host-compatible
PyTorch/CUDA stack, and no `plan.py` run.

The earlier plan-smoke failure before MuJoCo setup is:

```text
results/upstream/official_dinowm_pointmaze_plan_smoke_20260605_cu128_dinov2b483/
```

Status: `execution_failed`. It records that PointMaze planning is blocked by the
missing MuJoCo 2.1 runtime before model loading or CEM execution.

The current successful plan-smoke evidence package is:

```text
results/upstream/official_dinowm_pointmaze_plan_smoke_20260605_cu128_dinov2b483_mujoco_weights/
```

Status: `executed`. It records a minimal official `plan.py` run against the
train-smoke checkpoint. Upstream outputs were written under:

```text
external/dino_wm/plan_outputs/20260605152413_point_maze_official_frameskip5_hist3_seed0_gH5/
```

This is still not a full planning reproduction because it uses the 1-epoch
2-rollout smoke checkpoint and `planner.opt_steps=1`.

The current full-run dry-run package is:

```text
results/upstream/official_dinowm_pointmaze_full_preflight_20260605_cu128_dinov2b483/
```

Status: `execution_interrupted_wsl_crash`. It uses the same upstream commit,
official PointMaze data, host-compatible `dino_wm_cu128` environment,
artifact-local DINOv2 pin, MuJoCo environment variables, and full planning
settings (`n_evals=5`, `planner.opt_steps=30`). It includes a
`resolved_config_audit.md` generated from upstream Hydra `--cfg job` checks. The
audited train config resolves to `training.epochs=100`,
`training.batch_size=32`, `env.dataset.n_rollout=null`, and
`env.num_workers=16`.

The default full train command was launched on 2026-06-05 and WSL restarted
before the wrapper could record a Python return code. The last captured train
log reached full PointMaze loading (`Loaded 2000 rollouts`) with no epoch
metric, no checkpoint, and no Python traceback. Crash notes are recorded in:

```text
results/upstream/official_dinowm_pointmaze_full_preflight_20260605_cu128_dinov2b483/wsl_crash_analysis.md
```

Do not keep retrying this same default full train command on the current WSL
host.

The current local resource-adjusted full-data preflight package is:

```text
results/upstream/official_dinowm_pointmaze_full_resource_limited_preflight_20260605_cu128_dinov2b483/
```

Status: `prepared`. It keeps unmodified upstream code, official PointMaze data,
full dataset selection (`env.dataset.n_rollout=null`), host-compatible
`dino_wm_cu128`, artifact-local DINOv2 pin, and full trained-model planning
settings. It explicitly changes resource pressure to `training.batch_size=1`,
`env.num_workers=0`, `training.num_reconstruct_samples=1`, and
`training.reconstruct_every_x_batch=999999`. Its `resolved_config_audit.md`
confirms those overrides. This route can be used for a local diagnostic run, but
it is not a strict default upstream reproduction.

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

This follows the upstream README's trained-model planning example:

```bash
python plan.py model_name=<model_name> n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30
```

The separate upstream `plan_point_maze.yaml` config is for launching official
pretrained PointMaze checkpoints. It defaults to a PointMaze-specific MPC-style
wrapper and `n_evals=50`; keep that distinct from the trained-model README
planning path above.

The helper writes `config.yaml`, `command.sh`, `command.txt`, `summary.json`,
`sources.json`, `environment.txt`, `git_commit.txt`, `seeds.txt`, `split.json`,
and `notes.md` under `results/upstream/...`. After a successful train stage it
also records `metrics.csv`, copies the official Hydra train config, and creates
a stable `checkpoint.pt` pointer to upstream `model_latest.pth`. After a
successful plan stage it records the upstream `plan_outputs` directory, copies
the official Hydra plan config, and parses `logs.json` into `metrics.csv`.

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

This is resolved for local train-smoke execution by installing DINOv2 GitHub
commit `b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8` as an artifact-local
`TORCH_HOME/hub/facebookresearch_dinov2_main` cache. The DINO-WM source remains
unmodified; the transitive DINOv2 dependency is pinned and recorded in
`prepared_caches.json`, `summary.json`, `environment.txt`, and `notes.md`.

Planning for PointMaze imports `gym.envs.mujoco`, which imports `mujoco_py`.
The local route now requires these host-compatibility environment variables:

```text
PATH=/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/bin:...
LD_LIBRARY_PATH=/home/zhan_shaoji/.mujoco/mujoco210/bin:/home/zhan_shaoji/miniconda3/envs/dino_wm_cu128/lib:...
MUJOCO_GL=osmesa
D4RL_DATASET_DIR=/tmp/d4rl
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is required only for the
hardware-compatible PyTorch 2.8 environment, because official DINO-WM checkpoints
store full module objects and upstream `plan.py` calls `torch.load` without an
explicit `weights_only` argument.

## Recommended Execution Path

1. Keep the official `dino_wm` environment unchanged as the strict reference.
2. Use `dino_wm_cu128` for local GPU smoke because it is the only verified CUDA
   route on this RTX 5060 Ti host.
3. Use the artifact-local DINOv2 code pin for local train runs unless upstream
   DINO-WM publishes an official DINOv2 revision.
4. For strict "upstream DINO-WM reproduced" evidence, run the default full
   upstream PointMaze training command on native Linux or a remote GPU host with
   enough RAM/VRAM. The current WSL host already crashed twice on the audited
   default full train command.
5. If local progress is needed before remote compute is available, create a
   clearly labelled resource-adjusted full-data run using the same upstream
   code/data/DINOv2 pin but lower resource pressure, for example
   `training.batch_size=1`, `env.num_workers=0`, and reduced reconstruction
   sampling. The prepared local fallback package is
   `results/upstream/official_dinowm_pointmaze_full_resource_limited_preflight_20260605_cu128_dinov2b483/`.
   This is diagnostic local evidence, not a strict default upstream
   reproduction.
6. Run a full official PointMaze planning job only after full upstream training
   completes. Reuse the recorded MuJoCo, DINOv2, and PyTorch compatibility
   environment variables.
7. Only after upstream train and plan evidence exists, proceed to the local
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
