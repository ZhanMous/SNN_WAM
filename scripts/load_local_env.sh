#!/usr/bin/env bash
# Source local environment variables from .env.local.
# Usage: source scripts/load_local_env.sh
#
# .env.local is git-ignored. Copy .env.local.example to .env.local
# and fill in your paths.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="$repo_root/.env.local"

if [[ ! -f "$env_file" ]]; then
  echo "[load_local_env] .env.local not found at $env_file" >&2
  echo "[load_local_env] Copy .env.local.example to .env.local and fill in your paths." >&2
  return 1 2>/dev/null || exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do
  # Skip comments and empty lines
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ -z "$line" ]] && continue
  # Skip lines without =
  [[ "$line" != *=* ]] && continue

  key="${line%%=*}"
  value="${line#*=}"

  # Remove surrounding quotes if present
  if [[ "$value" =~ ^\"(.*)\"$ ]] || [[ "$value" =~ ^\'(.*)\'$ ]]; then
    value="${BASH_REMATCH[1]}"
  fi

  # Only export non-empty values
  if [[ -n "$value" ]]; then
    export "$key=$value"
  fi
done < "$env_file"

echo "[load_local_env] loaded $env_file"
