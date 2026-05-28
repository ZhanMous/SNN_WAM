#!/usr/bin/env python3
"""Smoke-check LIBERO import state and write a small JSON log."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_git(args: list[str]) -> str | None:
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
    candidates = {"yaml": ["pyyaml", "yaml"], "libero": ["libero"]}.get(
        module_name,
        [module_name],
    )
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def import_record(module_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "available": False,
        "version": None,
        "origin": None,
        "error": None,
    }
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as exc:
        record["error"] = repr(exc)
        return record
    if spec is None:
        return record

    record["origin"] = spec.origin
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on LIBERO install.
        record["error"] = repr(exc)
        return record

    record["available"] = True
    record["version"] = getattr(module, "__version__", None) or package_version(module_name)
    return record


def build_report() -> dict[str, Any]:
    status_short = run_git(["status", "--short"])
    return {
        "smoke": "libero_import",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "git": {
            "commit": run_git(["rev-parse", "--short", "HEAD"]),
            "branch": run_git(["branch", "--show-current"]),
            "dirty": bool(status_short),
        },
        "environment": {
            "LIBERO_DATA_ROOT": os.environ.get("LIBERO_DATA_ROOT"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "imports": {
            "libero": import_record("libero"),
            "libero.libero": import_record("libero.libero"),
            "libero.libero.benchmark": import_record("libero.libero.benchmark"),
            "libero.libero.envs": import_record("libero.libero.envs"),
        },
    }


def write_log(report: dict[str, Any], log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{timestamp}_libero_import.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/smoke"),
        help="Directory for JSON smoke logs. Default: results/smoke.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 even when LIBERO imports are unavailable.",
    )
    args = parser.parse_args()

    report = build_report()
    missing = [
        name
        for name, record in report["imports"].items()
        if not record.get("available", False)
    ]
    report["status"] = "pass" if not missing else "missing_imports"
    report["missing_imports"] = missing

    log_path = write_log(report, args.log_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"smoke_log={log_path}")

    if missing and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
