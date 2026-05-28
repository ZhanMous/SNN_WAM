from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_inspector(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/inspect_libero_data.py", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inspect_libero_data_has_help() -> None:
    result = run_inspector("--help")
    assert "usage:" in result.stdout
    assert "--mock" in result.stdout


def test_inspect_libero_data_mock_mode_writes_labeled_report(tmp_path: Path) -> None:
    result = run_inspector("--mock", "--output-dir", str(tmp_path))
    assert "mock=True" in result.stdout
    assert "trajectory_keys:" in result.stdout
    assert "time_convention: axis=0" in result.stdout
    reports = list(tmp_path.glob("*_libero_data_inspection_mock.json"))
    assert len(reports) == 1

    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["mock"] is True
    assert report["mode"] == "mock"
    assert "trajectory/obs/agentview_rgb" in report["trajectory_keys"]
    assert "trajectory/actions" in report["trajectory_keys"]

    action_record = next(
        item for item in report["datasets"] if item["path"] == "trajectory/actions"
    )
    assert action_record["shape"] == [8, 7]
    assert action_record["dtype"] == "float32"
    assert action_record["source_type"] == "dataset"
    assert action_record["time_axis"] == 0
    assert any("future target actions" in risk for risk in report["future_leakage_risks"])


@pytest.mark.optional
@pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None,
    reason="h5py is optional until real LIBERO data inspection is run.",
)
def test_optional_hdf5_inspection_with_synthetic_file(tmp_path: Path) -> None:
    import h5py  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    demo_path = tmp_path / "demo.hdf5"
    with h5py.File(demo_path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps(
            {
                "problem_name": "synthetic_problem",
                "language_instruction": "synthetic instruction",
            }
        )
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((5, 16, 16, 3), dtype="uint8"))
        obs.create_dataset("robot_state", data=np.zeros((5, 4), dtype="float32"))
        demo.create_dataset("actions", data=np.zeros((5, 2), dtype="float32"))

    result = run_inspector(
        "--path",
        str(demo_path),
        "--trajectory",
        "data/demo_0",
        "--output-dir",
        str(tmp_path),
    )
    assert "mode=real" in result.stdout
    reports = list(tmp_path.glob("*_libero_data_inspection_real.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["mock"] is False
    assert report["trajectory_id"] == "data/demo_0"
    language_record = next(
        item for item in report["datasets"] if item["field_role"] == "language"
    )
    assert language_record["path"] == "attrs/data/problem_info/language_instruction"
    assert language_record["dtype"] == "str"
    assert language_record["source_type"] == "attribute"
    assert language_record["value_preview"] == "synthetic instruction"
