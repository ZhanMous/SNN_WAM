#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/download_libero_minimal.sh [libero_spatial|libero_object]

Safe wrapper around the official LIBERO downloader for one small suite.

Required:
  LIBERO_REPO_ROOT      Official LIBERO checkout containing benchmark_scripts/download_libero_datasets.py

Optional:
  LIBERO_DATASET_ROOT   Preferred local dataset root for this repository
  LIBERO_DATA_ROOT      Compatibility alias for LIBERO data root
  PYTHON                Python executable inside the LIBERO environment, default: python3
  USE_HUGGINGFACE=1     Add --use-huggingface when supported by the installed LIBERO checkout

This script does not download all LIBERO datasets by default.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

suite="${1:-libero_spatial}"
case "$suite" in
  libero_spatial|libero_object)
    ;;
  *)
    echo "ERROR: unsupported minimal suite '$suite'. Use libero_spatial or libero_object." >&2
    exit 2
    ;;
esac

if [[ -z "${LIBERO_REPO_ROOT:-}" ]]; then
  echo "ERROR: LIBERO_REPO_ROOT is not set." >&2
  exit 1
fi

if [[ ! -d "$LIBERO_REPO_ROOT" ]]; then
  echo "ERROR: LIBERO_REPO_ROOT does not exist: $LIBERO_REPO_ROOT" >&2
  exit 1
fi

downloader="$LIBERO_REPO_ROOT/benchmark_scripts/download_libero_datasets.py"
if [[ ! -f "$downloader" ]]; then
  echo "ERROR: official LIBERO downloader not found: $downloader" >&2
  exit 1
fi

python_bin="${PYTHON:-python3}"
expected_data_root="${LIBERO_DATASET_ROOT:-${LIBERO_DATA_ROOT:-$LIBERO_REPO_ROOT/datasets}}"

cd "$LIBERO_REPO_ROOT"

cmd=("$python_bin" "benchmark_scripts/download_libero_datasets.py" "--datasets" "$suite")
if [[ "${USE_HUGGINGFACE:-0}" == "1" || "${LIBERO_USE_HUGGINGFACE:-0}" == "1" ]]; then
  cmd+=("--use-huggingface")
fi

echo "[download_libero_minimal] LIBERO_REPO_ROOT=$LIBERO_REPO_ROOT"
echo "[download_libero_minimal] suite=$suite"
echo "[download_libero_minimal] command=${cmd[*]}"
"${cmd[@]}"

echo "[download_libero_minimal] expected data location: $expected_data_root"
echo "[download_libero_minimal] next check:"
echo "  python scripts/bootstrap_libero_check.py --json --allow-fail"
