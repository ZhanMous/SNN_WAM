# LIBERO Bootstrap Gate

## Purpose

G1.5 LIBERO Bootstrap Gate exists between G1 Environment Gate and G2 Dataset Gate. It verifies that this workspace can see the official LIBERO repository, import `libero`, find a configured dataset root, locate at least one real `.hdf5` demonstration file, and inspect that file with `scripts/inspect_libero_demo.py`.

This gate does not implement `TrajectoryWindowDataset`, model code, SNN/GRU baselines, training, rollout policies, or paper claims.

## Required Local Variables

Set these in your shell, environment activation script, or another local-only setup file:

```bash
export LIBERO_REPO_ROOT="$HOME/code/third_party/LIBERO"
export LIBERO_DATASET_ROOT="$HOME/data/libero"
export LIBERO_DATA_ROOT="$LIBERO_DATASET_ROOT"
```

- `LIBERO_REPO_ROOT`: path to the cloned official LIBERO repository. It must contain `benchmark_scripts/download_libero_datasets.py`.
- `LIBERO_DATASET_ROOT`: preferred dataset root for this repository.
- `LIBERO_DATA_ROOT`: compatibility alias for LIBERO scripts or local setups that use this name.

Do not commit local absolute paths. Keep them in shell configuration or a private local file.

## Recommended Local Layout

Recommended third-party repository location:

```text
$HOME/code/third_party/LIBERO
```

Recommended downloaded dataset location:

```text
$HOME/data/libero
```

If the official downloader stores files under `$LIBERO_REPO_ROOT/datasets`, either set `LIBERO_DATASET_ROOT` to that directory or move/symlink the downloaded suite into the configured data root.

## Manual Bootstrap Commands

Use these commands when Codex has no network access, cannot create conda environments, or you want to keep third-party setup fully manual. On machines where `pip` is not a shell command, use `python3 -m pip`.

```bash
mkdir -p "$HOME/code/third_party" "$HOME/data/libero"
cd "$HOME/code/third_party"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

# Use the official LIBERO README for the exact dependency versions required by your machine.
# If conda is unavailable, a Python venv can still run repository checks:
# python3 -m venv .venv-libero
# source .venv-libero/bin/activate
# python3 -m pip install --upgrade pip
# python3 -m pip install -r requirements.txt
# python3 -m pip install -e .

export LIBERO_REPO_ROOT="$HOME/code/third_party/LIBERO"
export LIBERO_DATASET_ROOT="$HOME/data/libero"
export LIBERO_DATA_ROOT="$LIBERO_DATASET_ROOT"

cd /path/to/SNN_WAM
python3 scripts/check_environment.py --require libero
bash scripts/download_libero_minimal.sh libero_spatial
python3 scripts/bootstrap_libero_check.py --json --allow-fail
python3 scripts/inspect_libero_demo.py \
  --dataset-root "$LIBERO_DATASET_ROOT" \
  --suite libero_spatial \
  --update-docs
```

The official LIBERO documentation describes targeted suite download with:

```bash
python3 benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
```

Optional Hugging Face mirror mode, when supported by the installed LIBERO checkout:

```bash
USE_HUGGINGFACE=1 bash scripts/download_libero_minimal.sh libero_spatial
```

## Gate Commands

Run the non-mutating bootstrap check:

```bash
python3 scripts/bootstrap_libero_check.py --json --allow-fail
```

Run the repository quality gate:

```bash
bash scripts/quality_gate.sh
```

`quality_gate.sh` must not fail ordinary unit tests just because local LIBERO data is absent. It must print `G1.5 LIBERO Bootstrap Gate: FAIL` and state that G2 real dataset implementation is blocked until one real LIBERO HDF5 demonstration has been inspected.

## G2 Entry Rule

G2 TrajectoryWindowDataset v1 may begin only after G1.5 has inspected at least one real LIBERO HDF5 demonstration file.
