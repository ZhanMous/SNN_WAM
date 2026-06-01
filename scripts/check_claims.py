#!/usr/bin/env python3
"""Verify docs/CLAIMS_LEDGER.md exists and contains required sections.

Sections: Supported, Diagnostic-only, Unsupported, Forbidden.

Also checks that forbidden phrases do not appear in supported claims
unless they are explicitly in the Forbidden or Diagnostic-only sections.

Forbidden phrases:
    "future-latent improves"
    "SNN outperforms"
    "closed-loop success"
    "DINO-WM invalid"
    "DINOv2 unsuitable"

Exit code 0 on pass, 1 on fail.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLAIMS = REPO_ROOT / "docs" / "CLAIMS_LEDGER.md"

REQUIRED_SECTIONS = [
    "supported",
    "diagnostic-only",
    "unsupported",
    "forbidden",
]

FORBIDDEN_PHRASES = [
    "future-latent improves",
    "SNN outperforms",
    "closed-loop success",
    "DINO-WM invalid",
    "DINOv2 unsuitable",
]


def _section_names(text: str) -> list[str]:
    """Return lowercase section names from ## headers."""
    return [
        line.lstrip("#").strip().lower()
        for line in text.splitlines()
        if re.match(r"^##\s+", line)
    ]


def _find_sections(text: str) -> dict[str, str]:
    """Return mapping of section-name -> section-body (text between consecutive ## headers)."""
    sections: dict[str, str] = {}
    lines = text.splitlines()
    current_name: str | None = None
    current_body: list[str] = []

    for line in lines:
        m = re.match(r"^##\s+(.+)", line)
        if m:
            if current_name is not None:
                sections[current_name] = "\n".join(current_body)
            current_name = m.group(1).strip().lower()
            current_body = []
        elif current_name is not None:
            current_body.append(line)

    if current_name is not None:
        sections[current_name] = "\n".join(current_body)

    return sections


def _extract_table_rows(text: str, status_filter: str | None = None) -> list[str]:
    """Extract claim-text cells from markdown table rows, optionally filtering by status."""
    rows: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("| C-"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        status = cells[1].strip().lower()
        claim_text = cells[2].strip()
        if status_filter and status != status_filter.lower():
            continue
        rows.append(claim_text)
    return rows


def check_claims(path: Path) -> list[str]:
    errors: list[str] = []

    if not path.exists():
        return [f"File not found: {path}"]

    text = path.read_text(encoding="utf-8")

    # --- check required sections ---
    section_names = _section_names(text)
    for section in REQUIRED_SECTIONS:
        if section not in section_names:
            errors.append(
                f"Required section '## {section.title()}' not found. "
                f"Found sections: {section_names}"
            )

    # --- check forbidden phrases in supported claims ---
    supported_claims = _extract_table_rows(text, status_filter="supported")
    lower_supported = [c.lower() for c in supported_claims]

    forbidden_in_supported: list[str] = []
    for phrase in FORBIDDEN_PHRASES:
        phrase_lower = phrase.lower()
        for i, claim_text in enumerate(lower_supported):
            if phrase_lower in claim_text:
                forbidden_in_supported.append(
                    f"Phrase '{phrase}' found in supported claim: "
                    f"{supported_claims[i][:120]}"
                )

    if forbidden_in_supported:
        errors.extend(forbidden_in_supported)

    return errors


def main() -> int:
    claims_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLAIMS
    errors = check_claims(claims_path)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"claims_check_ok={claims_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
