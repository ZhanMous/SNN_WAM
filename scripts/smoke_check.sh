#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON:-python3}"

echo "[smoke] repo: $repo_root"
echo "[smoke] running pytest"
PYTHONDONTWRITEBYTECODE=1 "$python_bin" -m pytest -q -p no:cacheprovider "$@"
echo "[smoke] ok"
