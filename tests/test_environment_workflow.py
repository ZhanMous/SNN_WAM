import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_environment_checker_runs_without_optional_imports() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_environment.py", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert "python" in report
    assert "git" in report
    assert "imports" in report
    for module_name in ["torch", "libero", "yaml", "pytest", "numpy", "h5py"]:
        assert module_name in report["imports"]
        assert "available" in report["imports"][module_name]


@pytest.mark.optional
@pytest.mark.skipif(
    importlib.util.find_spec("libero") is None,
    reason="LIBERO is optional until G1 is validated on a LIBERO machine.",
)
def test_optional_libero_import_when_installed() -> None:
    assert importlib.import_module("libero") is not None


@pytest.mark.optional
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is optional in bootstrap CI but required for G1 validation.",
)
def test_optional_torch_cuda_query_when_installed() -> None:
    torch = importlib.import_module("torch")
    assert hasattr(torch, "cuda")


def test_environment_manifest_stays_libero_first() -> None:
    text = (ROOT / "environment.yml").read_text(encoding="utf-8").lower()
    assert "snnwam-libero" in text
    assert "torch" in text
    for forbidden in ["maniskill", "openvla", "unitree"]:
        assert forbidden not in text
