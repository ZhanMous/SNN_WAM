#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "[smoke] repo: $repo_root"
echo "[smoke] running pytest"
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider "$@"
echo "[smoke] ok"
