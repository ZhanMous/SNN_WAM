#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON:-python3}"

echo "============================================"
echo "  SNN-WAM Quality Gate"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# 1. Environment report
# ------------------------------------------------------------------
echo "[1/5] environment report"
"$python_bin" scripts/check_environment.py --json
echo ""

# ------------------------------------------------------------------
# 2. G1.5 LIBERO Bootstrap Gate
# ------------------------------------------------------------------
echo "[2/5] G1.5 LIBERO Bootstrap Gate"

g15_status="BLOCKED"
g15_detail=""

if env | grep -q "^LIBERO_REPO_ROOT=" && env | grep -q "^LIBERO_DATASET_ROOT=\|^LIBERO_DATA_ROOT="; then
  if "$python_bin" scripts/bootstrap_libero_check.py --json; then
    g15_status="PASS"
    g15_detail="All G1.5 checks passed. Real-data experiments are unblocked."
  else
    g15_status="FAIL"
    g15_detail="G1.5 checks ran but failed. See bootstrap report above."
  fi
else
  missing_vars=""
  env | grep -q "^LIBERO_REPO_ROOT=" || missing_vars="LIBERO_REPO_ROOT"
  env | grep -q "^LIBERO_DATASET_ROOT=\|^LIBERO_DATA_ROOT=" || missing_vars="$missing_vars LIBERO_DATASET_ROOT/LIBERO_DATA_ROOT"
  g15_detail="Environment variables not set:$missing_vars"
  g15_detail="$g15_detail"
  g15_detail="$g15_detail Real-data experiments are BLOCKED."
  g15_detail="$g15_detail To unblock: source scripts/load_local_env.sh (requires .env.local)"
fi

echo ""
echo "  G1.5 Bootstrap Status: $g15_status"
echo "  $g15_detail"
echo ""

# ------------------------------------------------------------------
# 3. Smoke tests (always run, do not require LIBERO)
# ------------------------------------------------------------------
echo "[3/5] smoke tests"
PYTHON="$python_bin" bash scripts/smoke_check.sh
echo ""

# ------------------------------------------------------------------
# 4. Result artifact registry
# ------------------------------------------------------------------
echo "[4/5] result artifact registry"
"$python_bin" scripts/check_result_artifacts.py
echo ""

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
echo "============================================"
echo "  Quality Gate Summary"
echo "============================================"
echo "  Unit tests:        PASS"
echo "  Artifact registry: PASS"
echo "  G1.5 Bootstrap:    $g15_status"

if [[ "$g15_status" == "PASS" ]]; then
  echo "  Real-data status:  UNBLOCKED"
else
  echo "  Real-data status:  BLOCKED"
  echo ""
  echo "  Current artifacts are engineering smoke only."
  echo "  No artifact is reportable scientific evidence."
fi

echo ""
echo "[quality_gate] done"
