from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_artifacts import check_artifacts
from scripts.check_claims import check_claims


# ---------------------------------------------------------------------------
# check_artifacts tests
# ---------------------------------------------------------------------------

class TestCheckArtifacts:
    def test_missing_file(self, tmp_path: Path) -> None:
        errors = check_artifacts(tmp_path / "nope.md")
        assert any("File not found" in e for e in errors)

    def test_no_table(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.md"
        f.write_text("# No table here\n", encoding="utf-8")
        errors = check_artifacts(f)
        assert any("No markdown table" in e for e in errors)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.md"
        f.write_text(
            "# Bad\n\n"
            "| artifact_id | notes |\n"
            "|---|---|\n"
            "| R-001 | test |\n",
            encoding="utf-8",
        )
        errors = check_artifacts(f)
        assert any("path" in e for e in errors)
        assert any("stage" in e for e in errors)
        assert any("command" in e for e in errors)
        assert any("git_commit" in e for e in errors)
        assert any("git_dirty" in e for e in errors)
        assert any("reportable" in e for e in errors)

    def test_all_fields_present(self, tmp_path: Path) -> None:
        f = tmp_path / "good.md"
        f.write_text(
            "# Good\n\n"
            "| artifact_id | path | stage | command | git_commit | git_dirty "
            "| reportable | notes |\n"
            "|---|---|---|---|---|---|---|---|\n"
            "| R-001 | results/runs/x/metrics.csv | smoke | python train | abc123 "
            "| False | False | smoke test |\n",
            encoding="utf-8",
        )
        errors = check_artifacts(f)
        assert errors == []

    def test_case_insensitive_field_match(self, tmp_path: Path) -> None:
        f = tmp_path / "case.md"
        f.write_text(
            "# Case\n\n"
            "| ARTIFACT_ID | Path | STAGE | COMMAND | GIT_COMMIT | GIT_DIRTY "
            "| REPORTABLE | Notes |\n"
            "|---|---|---|---|---|---|---|---|\n"
            "| R-001 | results/x | smoke | cmd | commit | False | False | n |\n",
            encoding="utf-8",
        )
        errors = check_artifacts(f)
        assert errors == []

    def test_extra_columns_ok(self, tmp_path: Path) -> None:
        f = tmp_path / "extra.md"
        f.write_text(
            "# Extra\n\n"
            "| artifact_id | run_id | path | stage | config | git_commit | "
            "git_dirty | env | seeds | eval_split | command | reportable | notes |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
            "| R-001 | run1 | results/x | smoke | c.yaml | abc | False | "
            "e.txt | 0 | val | cmd | False | n |\n",
            encoding="utf-8",
        )
        errors = check_artifacts(f)
        assert errors == []


# ---------------------------------------------------------------------------
# check_claims tests
# ---------------------------------------------------------------------------

VALID_CLAIMS = (
    "# Claims Ledger\n\n"
    "| Claim ID | Status | Claim | Artifact IDs | Evidence Files |\n"
    "|---|---|---|---|---|\n"
    "| C-001 | supported | Pipeline runs end-to-end. | R-001 | results/x/m.csv |\n"
    "\n"
    "## Supported\n\n"
    "Supported claims are listed in the table.\n\n"
    "## Diagnostic-only\n\n"
    "Diagnostic claims listed in the table.\n\n"
    "## Unsupported\n\n"
    "No unsupported claims currently.\n\n"
    "## Forbidden\n\n"
    "See forbidden phrases section.\n\n"
    "## Status Values\n\n"
    "- supported: has evidence.\n"
)


class TestCheckClaims:
    def test_missing_file(self, tmp_path: Path) -> None:
        errors = check_claims(tmp_path / "nope.md")
        assert any("File not found" in e for e in errors)

    def test_missing_sections(self, tmp_path: Path) -> None:
        f = tmp_path / "no_sections.md"
        f.write_text("# Claims\n\nNo sections.\n", encoding="utf-8")
        errors = check_claims(f)
        missing = [e for e in errors if "Required section" in e]
        assert len(missing) == 4

    def test_all_sections_present(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.md"
        f.write_text(VALID_CLAIMS, encoding="utf-8")
        errors = check_claims(f)
        assert errors == []

    def test_forbidden_phrase_in_supported_claim(self, tmp_path: Path) -> None:
        f = tmp_path / "bad_claim.md"
        f.write_text(
            "# Claims Ledger\n\n"
            "| Claim ID | Status | Claim | Artifact IDs | Evidence Files |\n"
            "|---|---|---|---|---|\n"
            "| C-001 | supported | SNN outperforms GRU. | R-001 | results/x/m.csv |\n"
            "\n"
            "## Supported\n\n"
            "## Diagnostic-only\n\n"
            "## Unsupported\n\n"
            "## Forbidden\n\n",
            encoding="utf-8",
        )
        errors = check_claims(f)
        assert any("SNN outperforms" in e for e in errors)

    def test_forbidden_phrase_in_forbidden_section_ok(self, tmp_path: Path) -> None:
        """Forbidden phrases in the Forbidden section should not trigger errors."""
        f = tmp_path / "forbidden_ok.md"
        f.write_text(
            "# Claims Ledger\n\n"
            "| Claim ID | Status | Claim | Artifact IDs | Evidence Files |\n"
            "|---|---|---|---|---|\n"
            "| C-001 | supported | Pipeline runs. | R-001 | results/x/m.csv |\n"
            "\n"
            "## Supported\n\n"
            "## Diagnostic-only\n\n"
            "## Unsupported\n\n"
            "## Forbidden\n\n"
            "- SNN outperforms is forbidden.\n",
            encoding="utf-8",
        )
        errors = check_claims(f)
        assert errors == []

    def test_forbidden_phrase_in_unsupported_claim_ok(self, tmp_path: Path) -> None:
        """Forbidden phrases in non-supported claims should not trigger errors."""
        f = tmp_path / "unsupported_ok.md"
        f.write_text(
            "# Claims Ledger\n\n"
            "| Claim ID | Status | Claim | Artifact IDs | Evidence Files |\n"
            "|---|---|---|---|---|\n"
            "| C-001 | rejected | SNN outperforms GRU. | R-001 | results/x/m.csv |\n"
            "\n"
            "## Supported\n\n"
            "## Diagnostic-only\n\n"
            "## Unsupported\n\n"
            "## Forbidden\n\n",
            encoding="utf-8",
        )
        errors = check_claims(f)
        assert errors == []

    def test_multiple_forbidden_phrases(self, tmp_path: Path) -> None:
        f = tmp_path / "multi.md"
        f.write_text(
            "# Claims Ledger\n\n"
            "| Claim ID | Status | Claim | Artifact IDs | Evidence Files |\n"
            "|---|---|---|---|---|\n"
            "| C-001 | supported | future-latent improves loss. | R-001 | r/m.csv |\n"
            "| C-002 | supported | closed-loop success achieved. | R-002 | r/m2.csv |\n"
            "| C-003 | supported | DINO-WM invalid result. | R-003 | r/m3.csv |\n"
            "| C-004 | supported | DINOv2 unsuitable for tasks. | R-004 | r/m4.csv |\n"
            "| C-005 | supported | SNN outperforms baseline. | R-005 | r/m5.csv |\n"
            "\n"
            "## Supported\n\n"
            "## Diagnostic-only\n\n"
            "## Unsupported\n\n"
            "## Forbidden\n\n",
            encoding="utf-8",
        )
        errors = check_claims(f)
        forbidden_found = [e for e in errors if "Phrase" in e]
        assert len(forbidden_found) == 5

    def test_forbidden_phrase_case_insensitive(self, tmp_path: Path) -> None:
        f = tmp_path / "case.md"
        f.write_text(
            "# Claims Ledger\n\n"
            "| Claim ID | Status | Claim | Artifact IDs | Evidence Files |\n"
            "|---|---|---|---|---|\n"
            "| C-001 | supported | The SNN Outperforms all baselines. | R-001 | r/m.csv |\n"
            "\n"
            "## Supported\n\n"
            "## Diagnostic-only\n\n"
            "## Unsupported\n\n"
            "## Forbidden\n\n",
            encoding="utf-8",
        )
        errors = check_claims(f)
        assert any("SNN outperforms" in e for e in errors)


# ---------------------------------------------------------------------------
# integration: check against real repo files
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRepoFiles:
    def test_real_artifacts_file(self) -> None:
        path = REPO_ROOT / "docs" / "RESULT_ARTIFACTS.md"
        if not path.exists():
            pytest.skip("RESULT_ARTIFACTS.md not found")
        errors = check_artifacts(path)
        assert errors == []

    def test_real_claims_file(self) -> None:
        path = REPO_ROOT / "docs" / "CLAIMS_LEDGER.md"
        if not path.exists():
            pytest.skip("CLAIMS_LEDGER.md not found")
        errors = check_claims(path)
        assert errors == []
