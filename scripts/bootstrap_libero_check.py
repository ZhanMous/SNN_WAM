#!/usr/bin/env python3
"""Check G1.5 LIBERO bootstrap prerequisites without installing or downloading."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = "libero_spatial"

from _common import run_git  # noqa: E402


def env_or_cli_path(cli_value: Path | None, env_names: tuple[str, ...]) -> tuple[Path | None, str | None]:
    if cli_value is not None:
        return cli_value.expanduser(), "cli"
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser(), env_name
    return None, None


def check_libero_import() -> dict[str, Any]:
    record: dict[str, Any] = {
        "ok": False,
        "available": False,
        "version": None,
        "module_file": None,
        "error": None,
    }
    try:
        spec = importlib.util.find_spec("libero")
    except Exception as exc:  # pragma: no cover - depends on optional package state.
        record["error"] = repr(exc)
        return record
    if spec is None:
        record["error"] = "ModuleNotFoundError('libero')"
        return record

    try:
        module = importlib.import_module("libero")
    except Exception as exc:  # pragma: no cover - depends on optional package state.
        record["error"] = repr(exc)
        return record

    record["ok"] = True
    record["available"] = True
    record["version"] = getattr(module, "__version__", None)
    record["module_file"] = getattr(module, "__file__", None)
    return record


def candidate_demo_roots(dataset_root: Path, suite: str) -> list[Path]:
    return [
        dataset_root / suite,
        dataset_root / f"{suite}_no_noops",
        dataset_root / "datasets" / suite,
        dataset_root / "datasets" / f"{suite}_no_noops",
        dataset_root,
    ]


def find_first_hdf5(dataset_root: Path, suite: str) -> Path | None:
    seen: set[Path] = set()
    for root in candidate_demo_roots(dataset_root, suite):
        resolved = root.expanduser()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        matches = sorted(resolved.rglob("*.hdf5"))
        if matches:
            return matches[0]
    return None


def run_inspector(dataset_root: Path, suite: str, output_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "inspect_libero_demo.py"),
        "--dataset-root",
        str(dataset_root),
        "--suite",
        suite,
        "--output-dir",
        str(output_dir),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": repr(exc),
        }

    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    libero_repo_root, repo_source = env_or_cli_path(
        args.libero_repo_root,
        ("LIBERO_REPO_ROOT",),
    )
    dataset_root, dataset_source = env_or_cli_path(
        args.dataset_root,
        ("LIBERO_DATASET_ROOT", "LIBERO_DATA_ROOT"),
    )

    repo_exists = bool(libero_repo_root and libero_repo_root.exists())
    downloader_path = (
        libero_repo_root / "benchmark_scripts" / "download_libero_datasets.py"
        if libero_repo_root
        else None
    )
    downloader_exists = bool(downloader_path and downloader_path.exists())

    dataset_exists = bool(dataset_root and dataset_root.exists())
    hdf5_demo = find_first_hdf5(dataset_root, args.suite) if dataset_exists and dataset_root else None
    import_record = check_libero_import()

    inspector_record: dict[str, Any]
    if hdf5_demo is None or dataset_root is None:
        inspector_record = {
            "ok": False,
            "skipped": True,
            "reason": "no_hdf5_demo_file_found",
        }
    else:
        inspector_record = run_inspector(dataset_root, args.suite, args.output_dir)
        inspector_record["skipped"] = False

    checks = {
        "libero_repo_root": {
            "ok": repo_exists,
            "path": str(libero_repo_root) if libero_repo_root else None,
            "source": repo_source,
        },
        "official_downloader": {
            "ok": downloader_exists,
            "path": str(downloader_path) if downloader_path else None,
        },
        "libero_import": import_record,
        "dataset_root": {
            "ok": dataset_exists,
            "path": str(dataset_root) if dataset_root else None,
            "source": dataset_source,
        },
        "hdf5_demo": {
            "ok": hdf5_demo is not None,
            "path": str(hdf5_demo) if hdf5_demo else None,
            "suite": args.suite,
        },
        "inspect_libero_demo": inspector_record,
    }

    blockers: list[str] = []
    if libero_repo_root is None:
        blockers.append("LIBERO_REPO_ROOT is not set.")
    elif not repo_exists:
        blockers.append("LIBERO_REPO_ROOT does not exist.")
    if not downloader_exists:
        blockers.append("benchmark_scripts/download_libero_datasets.py was not found under LIBERO_REPO_ROOT.")
    if not import_record["ok"]:
        blockers.append("import libero failed.")
    if dataset_root is None:
        blockers.append("LIBERO_DATASET_ROOT or LIBERO_DATA_ROOT is not set.")
    elif not dataset_exists:
        blockers.append("Configured LIBERO dataset root does not exist.")
    if hdf5_demo is None:
        blockers.append("No .hdf5 demonstration file was found under the configured dataset root.")
    if hdf5_demo is not None and not inspector_record["ok"]:
        blockers.append("scripts/inspect_libero_demo.py could not inspect the discovered .hdf5 file.")

    status = "PASS" if not blockers else "FAIL"
    return {
        "gate": "G1.5 LIBERO Bootstrap Gate",
        "status": status,
        "g2_blocked": status != "PASS",
        "suite": args.suite,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "blockers": blockers,
        "manual_next_steps": [
            "Set LIBERO_REPO_ROOT to the official LIBERO checkout.",
            "Install LIBERO from that checkout in the active environment.",
            "Set LIBERO_DATASET_ROOT or LIBERO_DATA_ROOT to the local dataset directory.",
            "Run: bash scripts/download_libero_minimal.sh libero_spatial",
            "Run: python scripts/inspect_libero_demo.py --dataset-root \"$LIBERO_DATASET_ROOT\" --suite libero_spatial --update-docs",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-repo-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/inspections"))
    parser.add_argument("--json", action="store_true", help="Emit JSON. JSON is the default output.")
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Exit 0 even when G1.5 is blocked. Intended for non-blocking quality reports.",
    )
    args = parser.parse_args()

    report = build_report(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "PASS" or args.allow_fail:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
