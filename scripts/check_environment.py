#!/usr/bin/env python3
"""Report the local SNN-WAM environment without requiring optional packages."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any


KEY_IMPORTS = ["torch", "libero", "yaml", "pytest", "numpy", "h5py"]

from _common import package_version, run_git  # noqa: E402


def check_import(module_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "available": False,
        "version": None,
        "error": None,
    }

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as exc:
        record["error"] = repr(exc)
        return record

    if spec is None:
        return record

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on optional packages.
        record["error"] = repr(exc)
        return record

    record["available"] = True
    record["version"] = getattr(module, "__version__", None) or package_version(module_name)

    if module_name == "torch":
        cuda = getattr(module, "cuda", None)
        cuda_available = bool(cuda and cuda.is_available())
        record["cuda_available"] = cuda_available
        record["cuda_version"] = getattr(getattr(module, "version", None), "cuda", None)
        record["cuda_device_count"] = int(cuda.device_count()) if cuda else 0

    return record


def build_report() -> dict[str, Any]:
    status_short = run_git(["status", "--short"])
    return {
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
        "imports": {name: check_import(name) for name in KEY_IMPORTS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        choices=KEY_IMPORTS,
        help="Fail if this import is unavailable. May be passed multiple times.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON. This is also the default format.",
    )
    args = parser.parse_args()

    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))

    missing_required = [
        name
        for name in args.require
        if not report["imports"].get(name, {}).get("available", False)
    ]
    if missing_required:
        print(
            "Missing required imports: " + ", ".join(missing_required),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
