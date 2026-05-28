#!/usr/bin/env python3
"""Validate registered result artifact paths and basic metric columns.

Checks:
- All referenced result paths exist.
- Metric CSVs have required columns.
- Reportable artifacts are not under smoke/debug.
- Reportable artifacts have git_commit.txt with dirty=False.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "docs" / "RESULT_ARTIFACTS.md"
METRIC_COLUMNS = {"action_mse", "total_loss", "action_loss", "future_loss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Artifact registry markdown file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = check_registry(args.registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"artifact_registry_ok={args.registry}")
    return 0


def check_registry(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("| R-") and not line.startswith("| R-000 ")
    ]
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        artifact_id = cells[0] if cells else "<unknown>"
        notes = cells[-1] if cells else ""
        result_paths = re.findall(r"`(results/[^`]+)`", row)
        if not result_paths:
            errors.append(f"{artifact_id} has no concrete results/ paths")
            continue
        is_reportable = "Status: reportable" in notes and "NOT REPORTABLE" not in notes
        if is_reportable and any(
            item.startswith(("results/smoke/", "results/debug/"))
            for item in result_paths
        ):
            errors.append(f"{artifact_id} is reportable but points to smoke/debug paths")
        for relative in result_paths:
            artifact_path = REPO_ROOT / relative
            if not artifact_path.exists():
                errors.append(f"{artifact_id} missing path: {relative}")
            elif artifact_path.name in {"metrics.csv", "eval_offline.csv"}:
                errors.extend(check_metric_csv(artifact_id, artifact_path))
        if is_reportable:
            errors.extend(check_clean_git(artifact_id, result_paths))
    return errors


def check_clean_git(artifact_id: str, result_paths: list[str]) -> list[str]:
    """Check that reportable artifacts have clean git state."""
    errors: list[str] = []
    # Find the run directory from the result paths
    run_dirs: set[Path] = set()
    for relative in result_paths:
        path = REPO_ROOT / relative
        # Walk up to find the run directory (contains git_commit.txt)
        for parent in [path] + list(path.parents):
            if (parent / "git_commit.txt").exists():
                run_dirs.add(parent)
                break
    for run_dir in run_dirs:
        git_commit_path = run_dir / "git_commit.txt"
        content = git_commit_path.read_text(encoding="utf-8").strip()
        if "dirty=True" in content:
            errors.append(
                f"{artifact_id} has dirty=True in {git_commit_path.relative_to(REPO_ROOT)}; "
                "reportable artifacts require clean git state"
            )
        if "dirty=False" not in content and "dirty=True" not in content:
            errors.append(
                f"{artifact_id} git_commit.txt at {git_commit_path.relative_to(REPO_ROOT)} "
                "does not record dirty state"
            )
    return errors


def check_metric_csv(artifact_id: str, path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        rows = list(reader)
    errors: list[str] = []
    if not rows:
        errors.append(f"{artifact_id} has empty metric CSV: {path}")
    missing = sorted(METRIC_COLUMNS - columns)
    if missing:
        errors.append(f"{artifact_id} metric CSV {path} missing columns: {missing}")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
