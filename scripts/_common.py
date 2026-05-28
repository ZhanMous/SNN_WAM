"""Shared helpers for scripts in the SNN-WAM repository."""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_git(args: list[str]) -> str | None:
    """Run a git command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def package_version(module_name: str) -> str | None:
    """Return the installed package version for a module name.

    Handles cases where the import name differs from the package name
    (e.g. ``yaml`` -> ``pyyaml``).
    """
    candidates = {
        "yaml": ["pyyaml", "yaml"],
        "libero": ["libero"],
    }.get(module_name, [module_name])
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None
