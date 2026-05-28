from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_bootstrap(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/bootstrap_libero_check.py", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_bootstrap_libero_check_has_help() -> None:
    result = run_bootstrap("--help")
    assert "usage:" in result.stdout
    assert "--libero-repo-root" in result.stdout
    assert "--dataset-root" in result.stdout
    assert "--allow-fail" in result.stdout


def test_bootstrap_libero_check_json_reports_blocked_when_unconfigured() -> None:
    env = os.environ.copy()
    for name in ["LIBERO_REPO_ROOT", "LIBERO_DATASET_ROOT", "LIBERO_DATA_ROOT"]:
        env.pop(name, None)

    result = run_bootstrap("--json", "--allow-fail", env=env)
    report = json.loads(result.stdout)

    assert report["gate"] == "G1.5 LIBERO Bootstrap Gate"
    assert report["status"] == "FAIL"
    assert report["g2_blocked"] is True
    assert report["checks"]["libero_repo_root"]["ok"] is False
    assert report["checks"]["dataset_root"]["ok"] is False
    assert report["checks"]["hdf5_demo"]["ok"] is False
    assert report["checks"]["inspect_libero_demo"]["skipped"] is True
    assert any("LIBERO_REPO_ROOT" in blocker for blocker in report["blockers"])
    assert any("No .hdf5 demonstration file" in blocker for blocker in report["blockers"])


def test_download_libero_minimal_help_is_side_effect_free() -> None:
    result = subprocess.run(
        ["bash", "scripts/download_libero_minimal.sh", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Usage:" in result.stdout
    assert "libero_spatial" in result.stdout
    assert "LIBERO_REPO_ROOT" in result.stdout


def test_download_libero_minimal_fails_fast_without_repo_root() -> None:
    env = os.environ.copy()
    env.pop("LIBERO_REPO_ROOT", None)
    result = subprocess.run(
        ["bash", "scripts/download_libero_minimal.sh", "libero_spatial"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "LIBERO_REPO_ROOT is not set" in result.stderr


@pytest.mark.optional
def test_bootstrap_checker_can_run_inspector_on_tiny_hdf5(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")

    libero_repo_root = tmp_path / "LIBERO"
    downloader = libero_repo_root / "benchmark_scripts" / "download_libero_datasets.py"
    downloader.parent.mkdir(parents=True)
    downloader.write_text("# official downloader placeholder for bootstrap test\n", encoding="utf-8")

    dataset_root = tmp_path / "datasets"
    suite_root = dataset_root / "libero_spatial"
    suite_root.mkdir(parents=True)
    demo_path = suite_root / "demo.hdf5"
    with h5py.File(demo_path, "w") as handle:
        group = handle.create_group("data").create_group("demo_0")
        obs = group.create_group("obs")
        obs.create_dataset("agentview_rgb", data=[[[[0, 0, 0]]], [[[1, 1, 1]]]])
        obs.create_dataset("robot_state", data=[[0.0, 0.1], [1.0, 1.1]])
        group.create_dataset("actions", data=[[0.0, 0.0], [1.0, 1.0]])

    result = run_bootstrap(
        "--json",
        "--allow-fail",
        "--libero-repo-root",
        str(libero_repo_root),
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(tmp_path / "reports"),
    )
    report = json.loads(result.stdout)

    assert report["checks"]["official_downloader"]["ok"] is True
    assert report["checks"]["hdf5_demo"]["path"] == str(demo_path)
    assert report["checks"]["inspect_libero_demo"]["ok"] is True
    if report["checks"]["libero_import"]["ok"]:
        assert report["status"] == "PASS"
        assert report["blockers"] == []
    else:
        assert report["status"] == "FAIL"
        assert any("import libero failed" in blocker for blocker in report["blockers"])
