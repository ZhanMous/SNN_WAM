#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON:-python3}"

echo "[quality_gate] environment report"
"$python_bin" scripts/check_environment.py --json

echo "[quality_gate] G1.5 LIBERO Bootstrap Gate"
if "$python_bin" scripts/bootstrap_libero_check.py --json; then
  echo "G1.5 LIBERO Bootstrap Gate: PASS"
else
  echo "G1.5 LIBERO Bootstrap Gate: FAIL"
  echo "G2 real dataset implementation is blocked until one real LIBERO HDF5 demonstration file is inspected."
fi

echo "[quality_gate] smoke tests"
PYTHON="$python_bin" bash scripts/smoke_check.sh

echo "[quality_gate] ok"
