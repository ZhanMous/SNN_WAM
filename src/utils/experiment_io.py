"""Reproducible experiment output directory helpers.

This module creates run directories and writes infrastructure artifacts only.
It does not start training, write metrics, or create checkpoints.
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.utils.config import validate_config


def create_experiment_dir(
    config: Mapping[str, Any],
    *,
    command: Sequence[str] | str | None = None,
    notes: str = "No notes provided.\n",
    run_id: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Create a non-overwriting run directory and write core artifacts.

    Required artifacts:

    - `config.yaml`: exact config used to create the run directory.
    - `command.txt`: command string for reproduction.
    - `git_commit.txt`: current commit plus dirty flag when available.
    - `environment.txt`: Python/platform/runtime environment summary.
    - `notes.md`: human notes and limitations.

    Args:
        config: Validated experiment config mapping.
        command: Command string or argv sequence. Defaults to current process
            argv for smoke tests.
        notes: Markdown notes content.
        run_id: Optional stable run id for tests or scripted runs.
        timestamp: Optional timestamp prefix for deterministic tests. If
            omitted, current UTC time is used.

    Returns:
        The newly created run directory path.

    Raises:
        FileExistsError: If the target run directory already exists.
    """

    validate_config(config)
    output_root = Path(str(config["output"]["output_dir"]))
    resolved_run_id = run_id or format_run_id(config, timestamp=timestamp)
    run_dir = output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    write_yaml(run_dir / "config.yaml", config)
    (run_dir / "command.txt").write_text(format_command(command), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(capture_git_commit(), encoding="utf-8")
    (run_dir / "environment.txt").write_text(capture_environment(), encoding="utf-8")
    (run_dir / "notes.md").write_text(_ensure_trailing_newline(notes), encoding="utf-8")
    return run_dir


def format_run_id(config: Mapping[str, Any], *, timestamp: str | None = None) -> str:
    """Return `YYYYMMDD_HHMM_<suite>_<adapter>_<name>_seed<seed>`."""

    validate_config(config)
    prefix = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    suite = slug(config["data"]["suite"])
    adapter = slug(config["model"]["temporal_adapter"])
    name = slug(config["experiment"]["name"])
    seed = int(config["experiment"]["seed"])
    return f"{prefix}_{suite}_{adapter}_{name}_seed{seed}"


def write_yaml(path: Path, config: Mapping[str, Any]) -> None:
    """Write a YAML mapping with stable key order preserved."""

    path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")


def format_command(command: Sequence[str] | str | None) -> str:
    """Format a reproducible command string for `command.txt`."""

    if command is None:
        command = sys.argv
    if isinstance(command, str):
        return _ensure_trailing_newline(command)
    return _ensure_trailing_newline(" ".join(shlex.quote(part) for part in command))


def capture_git_commit() -> str:
    """Capture current git commit and dirty state, or `unknown` outside git."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"commit={commit}\ndirty={bool(status)}\n"
    except (OSError, subprocess.CalledProcessError):
        return "commit=unknown\ndirty=unknown\n"


def capture_environment() -> str:
    """Capture minimal environment text for reproducibility."""

    lines = [
        f"python_executable={sys.executable}",
        f"python_version={platform.python_version()}",
        f"platform={platform.platform()}",
        f"cwd={Path.cwd()}",
    ]
    for name in ("PYTHONHASHSEED", "LIBERO_DATASET_ROOT", "LIBERO_DATA_ROOT"):
        lines.append(f"{name}={os.environ.get(name, '')}")
    return "\n".join(lines) + "\n"


def slug(value: Any) -> str:
    """Return a filesystem-safe lowercase slug."""

    text = str(value).strip().lower()
    output = []
    previous_underscore = False
    for char in text:
        if char.isalnum():
            output.append(char)
            previous_underscore = False
        elif not previous_underscore:
            output.append("_")
            previous_underscore = True
    return "".join(output).strip("_") or "unnamed"


def _ensure_trailing_newline(value: str) -> str:
    return value if value.endswith("\n") else f"{value}\n"


__all__ = [
    "capture_environment",
    "capture_git_commit",
    "create_experiment_dir",
    "format_command",
    "format_run_id",
    "slug",
]
