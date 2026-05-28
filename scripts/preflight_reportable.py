#!/usr/bin/env python3
"""Preflight check for reportable experiments.

Fails closed if any precondition for a reportable experiment is missing:
- Git working tree is dirty
- LIBERO_REPO_ROOT is not set
- LIBERO_DATASET_ROOT is not set
- Config does not have reproducibility.require_clean_git=true
- Output artifact ID already exists in RESULT_ARTIFACTS.md

Exit codes:
- 0: All preflight checks passed
- 1: One or more preflight checks failed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ARTIFACTS_PATH = REPO_ROOT / "docs" / "RESULT_ARTIFACTS.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config YAML file.",
    )
    parser.add_argument(
        "--artifact-id",
        type=str,
        default=None,
        help="Artifact ID to register (checks for duplicates).",
    )
    return parser.parse_args()


def check_git_clean() -> list[str]:
    """Check that the git working tree is clean."""
    errors: list[str] = []
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            errors.append(
                "Git working tree is dirty. Commit or stash changes before "
                "running reportable experiments."
            )
    except subprocess.CalledProcessError as exc:
        errors.append(f"Failed to check git status: {exc}")
    return errors


def check_env_vars() -> list[str]:
    """Check that required environment variables are set."""
    errors: list[str] = []
    import os

    if not os.environ.get("LIBERO_REPO_ROOT"):
        errors.append(
            "LIBERO_REPO_ROOT is not set. "
            "Set it to the LIBERO repository root directory."
        )
    if not os.environ.get("LIBERO_DATASET_ROOT") and not os.environ.get("LIBERO_DATA_ROOT"):
        errors.append(
            "LIBERO_DATASET_ROOT (or LIBERO_DATA_ROOT) is not set. "
            "Set it to the LIBERO dataset directory."
        )
    return errors


def check_config_require_clean_git(config_path: Path) -> list[str]:
    """Check that the config has reproducibility.require_clean_git=true."""
    errors: list[str] = []
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Failed to load config {config_path}: {exc}")
        return errors

    if not isinstance(config, dict):
        errors.append(f"Config {config_path} is not a valid YAML mapping")
        return errors

    reproducibility = config.get("reproducibility")
    if not isinstance(reproducibility, dict):
        errors.append(
            f"Config {config_path} is missing 'reproducibility' section. "
            "Reportable configs must have reproducibility.require_clean_git=true"
        )
        return errors

    if not reproducibility.get("require_clean_git"):
        errors.append(
            f"Config {config_path} has reproducibility.require_clean_git != true. "
            "Reportable configs must set require_clean_git=true"
        )
    return errors


def check_artifact_id_not_duplicate(artifact_id: str) -> list[str]:
    """Check that the artifact ID doesn't already exist in RESULT_ARTIFACTS.md."""
    errors: list[str] = []
    if not RESULT_ARTIFACTS_PATH.exists():
        return errors

    text = RESULT_ARTIFACTS_PATH.read_text(encoding="utf-8")
    if f"| {artifact_id} " in text or f"| {artifact_id}|" in text:
        errors.append(
            f"Artifact ID '{artifact_id}' already exists in {RESULT_ARTIFACTS_PATH}. "
            "Choose a unique artifact ID."
        )
    return errors


def check_output_dir_not_smoke(config_path: Path) -> list[str]:
    """Check that output_dir is not under results/smoke/ or results/debug/."""
    errors: list[str] = []
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return errors

    if not isinstance(config, dict):
        return errors

    output = config.get("output", {})
    output_dir = output.get("output_dir", "")
    if isinstance(output_dir, str) and (
        "results/smoke/" in output_dir or "results/debug/" in output_dir
    ):
        errors.append(
            f"Reportable config has output_dir='{output_dir}'. "
            "Reportable experiments must not use results/smoke/ or results/debug/."
        )
    return errors


def main() -> int:
    args = parse_args()
    all_errors: list[str] = []

    # Check that config file exists
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 1

    # Run all checks
    all_errors.extend(check_git_clean())
    all_errors.extend(check_env_vars())
    all_errors.extend(check_config_require_clean_git(args.config))
    all_errors.extend(check_output_dir_not_smoke(args.config))

    if args.artifact_id:
        all_errors.extend(check_artifact_id_not_duplicate(args.artifact_id))

    # Report results
    if all_errors:
        print("PREFLIGHT FAILED:", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("PREFLIGHT PASSED")
    print(f"  Config: {args.config}")
    if args.artifact_id:
        print(f"  Artifact ID: {args.artifact_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
