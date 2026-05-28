# Environment

This project is LIBERO-first. G1 Environment Gate proves that the local machine can import PyTorch and LIBERO, record versions, and run a minimal LIBERO demo log. It does not train models.

## Recommended Environments

### `snnwam-libero`

Use this first. It is the default Phase-1 environment for repository smoke tests, offline adapter experiments, and the first LIBERO import/demo checks.

Create the base environment when conda is available:

```bash
conda env create -f environment.yml
conda activate snnwam-libero
python scripts/check_environment.py
```

The base `environment.yml` includes Python, pytest, PyYAML, NumPy, h5py, tqdm, and PyTorch packages. Adjust the PyTorch install line for the machine CUDA version if needed, following the official PyTorch instructions.

If `conda` is not installed, the repository checks can still use the system Python entrypoints:

```bash
python3 --version
python3 -m pip --version
python3 -m venv .venv-libero
source .venv-libero/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install pytest pyyaml
python3 scripts/check_environment.py
```

Use `python3 -m pip ...` instead of bare `pip` on systems where the `pip` command is absent.

LIBERO should be installed after the base environment from the official LIBERO repository and README that matches the target machine. Re-run:

```bash
python scripts/check_environment.py --require torch --require libero
```

Save that output as `environment.txt` in any reportable G1/G2+ run directory.

Outside a conda environment, use `python3` if the machine does not provide a `python` alias. The repository scripts also honor `PYTHON=/path/to/python`.

### `snnwam-maniskill`

Create this later only when the project reaches the appropriate rollout or large-scale simulation phase. It must be a separate environment because simulation dependencies can conflict with the LIBERO stack.

Do not add ManiSkill, OpenVLA, or Unitree dependencies to the default `snnwam-libero` environment.

## Required Dependencies

Required for G0/G1 repository checks:

- Python
- git
- pytest
- PyYAML

Required for actual G1 LIBERO-first validation:

- PyTorch
- LIBERO installed from its official source
- Any LIBERO-required simulator/runtime dependencies documented by the official LIBERO README

## Optional Dependencies

Optional and later-phase only:

- SNN framework packages for G3+ adapter work.
- ManiSkill for later simulation work in a separate environment.
- OpenVLA only for later baseline/backbone work, not Phase-1 default.
- Unitree tools only for future hardware work, not Phase 1.

## Environment Checker

`scripts/check_environment.py` reports:

- Python version, executable, and platform
- git branch, short commit, and dirty status
- PyTorch import status, version, CUDA availability, CUDA version, and GPU count
- LIBERO import status
- key utility imports such as PyYAML, pytest, NumPy, and h5py

The checker exits successfully by default even when optional packages are missing. Use `--require torch --require libero` when validating the G1 LIBERO machine.

## LIBERO Smoke Checks

These commands do not train models, do not download datasets, and do not modify LIBERO source code.

Import smoke:

```bash
python scripts/smoke_libero_import.py
```

Dry LIBERO env-step smoke. This checks imports and task API only; it does not instantiate an environment:

```bash
python scripts/smoke_libero_env_step.py --suite libero_spatial --task-id 0
```

Offscreen reset/step smoke. This is the first command that actually creates `OffScreenRenderEnv`, resets it, and runs a tiny zero-action step:

```bash
MUJOCO_GL=egl python scripts/smoke_libero_env_step.py \
  --suite libero_spatial \
  --task-id 0 \
  --max-steps 1 \
  --run-step
```

Smoke logs are written to `results/smoke/` only when the scripts are actually run. `--help` is side-effect free.

Environment variables:

- `LIBERO_DATASET_ROOT`: preferred path to local LIBERO datasets for this repository.
- `LIBERO_DATA_ROOT`: optional compatibility alias for LIBERO versions or scripts that use this name.
- `LIBERO_REPO_ROOT`: optional path to an installed or cloned official LIBERO repository containing `benchmark_scripts/download_libero_datasets.py`.
- `MUJOCO_GL`: offscreen MuJoCo backend; prefer `egl` on GPU Linux machines and `osmesa` when EGL is unavailable.
- `PYOPENGL_PLATFORM`: optional OpenGL backend override if required by the simulator stack.

Do not use GUI-only rendering assumptions for Phase 1 smoke checks.

## Minimal Dataset Locate/Download

First check whether the dataset root is already configured:

```bash
echo "$LIBERO_DATASET_ROOT"
python scripts/inspect_libero_demo.py --allow-missing
```

If missing, copy [LOCAL_PATHS_TEMPLATE.md](LOCAL_PATHS_TEMPLATE.md) into your local shell setup and set `LIBERO_DATASET_ROOT`.
See [LIBERO_BOOTSTRAP.md](LIBERO_BOOTSTRAP.md) for the G1.5 LIBERO Bootstrap Gate and the full manual command sequence.

Use the official downloader from the installed LIBERO repository:

```bash
cd "$LIBERO_REPO_ROOT"
python3 benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
```

Or use this repository's safe minimal wrapper:

```bash
bash scripts/download_libero_minimal.sh libero_spatial
python scripts/bootstrap_libero_check.py --json --allow-fail
```

Then inspect one real demonstration:

```bash
python scripts/inspect_libero_demo.py \
  --dataset-root "$LIBERO_DATASET_ROOT" \
  --suite libero_spatial \
  --update-docs
```

Do not proceed to `TrajectoryWindowDataset` until `docs/LIBERO_DATA_CONTRACT.md` contains an observed real schema. After that schema exists, keep v1 limited to causal windowing until action alignment and split policy are resolved.
