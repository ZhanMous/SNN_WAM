from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_libero_smoke_scripts_have_help() -> None:
    for script in [
        "scripts/smoke_libero_import.py",
        "scripts/smoke_libero_env_step.py",
        "scripts/check_libero_action_alignment.py",
    ]:
        result = run_script(script, "--help")
        assert "usage:" in result.stdout


@pytest.mark.optional
@pytest.mark.skipif(
    importlib.util.find_spec("libero") is None,
    reason="LIBERO is optional until G1 is validated on a LIBERO machine.",
)
def test_optional_libero_import_smoke_runs_when_installed(tmp_path: Path) -> None:
    result = run_script("scripts/smoke_libero_import.py", "--log-dir", str(tmp_path))
    assert "smoke_log=" in result.stdout
    logs = list(tmp_path.glob("*_libero_import.json"))
    assert len(logs) == 1
    report = json.loads(logs[0].read_text(encoding="utf-8"))
    assert report["status"] == "pass"


@pytest.mark.optional
@pytest.mark.skipif(
    importlib.util.find_spec("libero") is None,
    reason="LIBERO is optional until G1 is validated on a LIBERO machine.",
)
def test_optional_libero_env_step_dry_run_when_installed(tmp_path: Path) -> None:
    result = run_script(
        "scripts/smoke_libero_env_step.py",
        "--log-dir",
        str(tmp_path),
    )
    assert "smoke_log=" in result.stdout
    logs = list(tmp_path.glob("*_libero_env_step.json"))
    assert len(logs) == 1
    report = json.loads(logs[0].read_text(encoding="utf-8"))
    assert report["status"] == "dry_run_ok"
