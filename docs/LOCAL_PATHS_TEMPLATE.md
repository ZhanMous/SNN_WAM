# Local Paths Template

Copy the relevant exports into your shell profile, `.envrc`, or job script. Do not commit machine-specific absolute paths.

```bash
# Required before G2 real data inspection.
export LIBERO_DATASET_ROOT="/absolute/path/to/libero/datasets"

# Optional compatibility alias used by some LIBERO docs/scripts.
export LIBERO_DATA_ROOT="$LIBERO_DATASET_ROOT"

# Optional path to an installed or cloned official LIBERO repository.
export LIBERO_REPO_ROOT="/absolute/path/to/LIBERO"
```

Expected first-suite layout examples:

```text
$LIBERO_DATASET_ROOT/libero_spatial/*.hdf5
$LIBERO_DATASET_ROOT/datasets/libero_spatial/*.hdf5
$LIBERO_DATASET_ROOT/libero_spatial_no_noops/*.hdf5
```

Official download reminder:

```bash
cd "$LIBERO_REPO_ROOT"
python3 benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
```

Use the official LIBERO README for the exact command supported by your installed version. Download only the first minimal suite until G2 inspection passes.
