#!/usr/bin/env python3
"""Verify docs/RESULT_ARTIFACTS.md exists and contains required columns/fields.

Required fields (checked as case-insensitive substrings in table headers):
    artifact_id, path, stage, command, git_commit, git_dirty, reportable, notes

Exit code 0 on pass, 1 on fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = REPO_ROOT / "docs" / "RESULT_ARTIFACTS.md"

REQUIRED_FIELDS = [
    "artifact_id",
    "path",
    "stage",
    "command",
    "git_commit",
    "git_dirty",
    "reportable",
    "notes",
]


def _extract_table_headers(text: str) -> list[str]:
    """Return the first markdown table header row split into cell strings."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip().lower() for c in stripped.strip("|").split("|")]
            return cells
    return []


def check_artifacts(path: Path) -> list[str]:
    errors: list[str] = []

    if not path.exists():
        return [f"File not found: {path}"]

    text = path.read_text(encoding="utf-8")
    headers = _extract_table_headers(text)

    if not headers:
        return [f"No markdown table found in {path}"]

    for field in REQUIRED_FIELDS:
        field_lower = field.lower()
        found = any(field_lower in header for header in headers)
        if not found:
            errors.append(
                f"Required field '{field}' not found in table headers: {headers}"
            )

    return errors


def main() -> int:
    artifacts_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ARTIFACTS
    )
    errors = check_artifacts(artifacts_path)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"artifact_check_ok={artifacts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
