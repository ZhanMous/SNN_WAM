#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON:-python3}"

echo "============================================"
echo "  SNN-WAM Quality Gate (safe checks only)"
echo "============================================"
echo ""

rc=0

# ------------------------------------------------------------------
# 1. pytest (if tests/ exists)
# ------------------------------------------------------------------
echo "[1/4] tests"
if [ -d tests/ ]; then
  if "$python_bin" -m pytest -q 2>&1; then
    echo "  PASS"
  else
    echo "  FAIL (see above)"
    rc=1
  fi
else
  echo "  pytest not configured"
fi
echo ""

# ------------------------------------------------------------------
# 2. artifact checker (if scripts/check_artifacts.py exists)
# ------------------------------------------------------------------
echo "[2/4] artifact checker"
if [ -f scripts/check_artifacts.py ]; then
  if "$python_bin" scripts/check_artifacts.py 2>&1; then
    echo "  PASS"
  else
    echo "  FAIL (see above)"
    rc=1
  fi
else
  echo "  artifact checker not configured"
fi
echo ""

# ------------------------------------------------------------------
# 3. claims checker (if scripts/check_claims.py exists)
# ------------------------------------------------------------------
echo "[3/4] claims checker"
if [ -f scripts/check_claims.py ]; then
  if "$python_bin" scripts/check_claims.py 2>&1; then
    echo "  PASS"
  else
    echo "  FAIL (see above)"
    rc=1
  fi
else
  echo "  claims checker not configured"
fi
echo ""

# ------------------------------------------------------------------
# 4. git status
# ------------------------------------------------------------------
echo "[4/4] git status"
git status --short
echo ""

echo "============================================"
echo "  Quality Gate Summary"
echo "============================================"
if [ "$rc" -eq 0 ]; then
  echo "  Overall: PASS"
else
  echo "  Overall: FAIL"
fi
echo "[quality_gate] done"
exit "$rc"
