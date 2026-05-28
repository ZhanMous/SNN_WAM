from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.utils.config import load_config
from src.utils.experiment_io import create_experiment_dir, format_run_id


ROOT = Path(__file__).resolve().parents[1]


def config_with_output_root(tmp_path: Path) -> dict:
    config = deepcopy(load_config(ROOT / "configs/libero_spatial_mlp.yaml"))
    config["output"]["output_dir"] = str(tmp_path / "runs")
    return config


def test_create_experiment_dir_writes_reproducibility_files(tmp_path: Path) -> None:
    config = config_with_output_root(tmp_path)

    run_dir = create_experiment_dir(
        config,
        command=["python3", "-m", "pytest", "-q"],
        notes="placeholder config infrastructure only\n",
        run_id="fixed_run",
    )

    assert run_dir == tmp_path / "runs" / "fixed_run"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "command.txt").read_text(encoding="utf-8") == (
        "python3 -m pytest -q\n"
    )
    assert "commit=" in (run_dir / "git_commit.txt").read_text(encoding="utf-8")
    assert "python_version=" in (run_dir / "environment.txt").read_text(
        encoding="utf-8"
    )
    assert (run_dir / "notes.md").read_text(encoding="utf-8") == (
        "placeholder config infrastructure only\n"
    )


def test_create_experiment_dir_does_not_overwrite_existing_dir(tmp_path: Path) -> None:
    config = config_with_output_root(tmp_path)
    create_experiment_dir(config, run_id="same_run")

    with pytest.raises(FileExistsError):
        create_experiment_dir(config, run_id="same_run")


def test_format_run_id_is_stable_with_timestamp() -> None:
    config = load_config(ROOT / "configs/libero_spatial_gru.yaml")

    run_id = format_run_id(config, timestamp="20260527_1200")

    assert run_id == (
        "20260527_1200_libero_spatial_gru_"
        "libero_spatial_gru_action_placeholder_seed0"
    )
