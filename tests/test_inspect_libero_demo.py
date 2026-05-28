from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_demo_inspector(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/inspect_libero_demo.py", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inspect_libero_demo_has_help() -> None:
    result = run_demo_inspector("--help")
    assert "usage:" in result.stdout
    assert "--dataset-root" in result.stdout
    assert "--allow-missing" in result.stdout


def test_inspect_libero_demo_missing_root_writes_missing_report(tmp_path: Path) -> None:
    result = run_demo_inspector(
        "--dataset-root",
        str(tmp_path / "missing_root"),
        "--output-dir",
        str(tmp_path),
        "--allow-missing",
    )
    assert "dataset_root_missing" in result.stdout
    reports = list(tmp_path.glob("*_libero_demo_missing.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "missing"
    assert report["reason"] == "dataset_root_missing"


def test_inspect_libero_demo_locates_json_demo_and_updates_docs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_root"
    suite_root = dataset_root / "libero_spatial"
    suite_root.mkdir(parents=True)
    demo_path = suite_root / "demo.json"
    demo_path.write_text(
        json.dumps(
            {
                "obs": {
                    "agentview_rgb": [[[[0, 0, 0]]], [[[1, 1, 1]]]],
                    "robot_state": [[0.0, 0.1], [1.0, 1.1]],
                },
                "actions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                "language": "synthetic instruction",
            }
        ),
        encoding="utf-8",
    )

    result = run_demo_inspector(
        "--dataset-root",
        str(dataset_root),
        "--suite",
        "libero_spatial",
        "--output-dir",
        str(tmp_path),
    )
    assert str(demo_path) in result.stdout
    reports = list(tmp_path.glob("*_libero_data_inspection_real.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["mode"] == "real"
    assert "obs/agentview_rgb" in report["trajectory_keys"]
