"""Tests for the official upstream DINO-WM reproduction helper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.reproduce_official_dinowm_upstream import (
    build_commands,
    normalize_paths,
    preflight_checks,
    update_execution_status,
    write_repro_package,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        upstream_dir=tmp_path / "dino_wm",
        dataset_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifact",
        env="point_maze",
        frameskip=5,
        num_hist=3,
        num_pred=1,
        seed=0,
        ckpt_base_path=tmp_path / "ckpts",
        model_name="point_maze_test_seed0",
        python_exe="python",
        wandb_mode="offline",
        stage="preflight",
        execute=False,
        n_evals=5,
        planner="cem",
        goal_H=5,
        goal_source="random_state",
        planner_opt_steps=30,
        extra_train_arg=[],
        extra_plan_arg=[],
    )


def test_build_commands_match_official_pointmaze_entrypoint(tmp_path: Path) -> None:
    args = _args(tmp_path)
    commands = build_commands(args)

    assert commands.train[:4] == ["python", "train.py", "--config-name", "train.yaml"]
    assert "env=point_maze" in commands.train
    assert "frameskip=5" in commands.train
    assert "num_hist=3" in commands.train
    assert "num_pred=1" in commands.train
    assert "training.seed=0" in commands.train

    assert commands.plan[:2] == ["python", "plan.py"]
    assert "model_name=point_maze_test_seed0" in commands.plan
    assert "planner=cem" in commands.plan
    assert "goal_H=5" in commands.plan
    assert "goal_source=random_state" in commands.plan
    assert "planner.opt_steps=30" in commands.plan


def test_normalize_paths_keeps_outputs_outside_upstream_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = _args(tmp_path)
    args.upstream_dir = Path("external/dino_wm")
    args.dataset_dir = Path("data/dino_wm")
    args.artifact_dir = Path("results/upstream/run")
    args.ckpt_base_path = Path("results/upstream/run/official_ckpts")

    normalize_paths(args)
    commands = build_commands(args)

    assert args.upstream_dir == tmp_path / "external/dino_wm"
    assert args.dataset_dir == tmp_path / "data/dino_wm"
    assert args.artifact_dir == tmp_path / "results/upstream/run"
    expected_ckpt = tmp_path / "results/upstream/run/official_ckpts"
    assert args.ckpt_base_path == expected_ckpt
    assert f"ckpt_base_path={expected_ckpt}" in commands.train


def test_preflight_checks_official_repo_and_dataset_layout(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (args.upstream_dir / "conf").mkdir(parents=True)
    (args.upstream_dir / "train.py").write_text("")
    (args.upstream_dir / "plan.py").write_text("")
    (args.upstream_dir / "conf" / "train.yaml").write_text("")
    (args.upstream_dir / "conf" / "plan.yaml").write_text("")
    (args.dataset_dir / "point_maze").mkdir(parents=True)

    checks = preflight_checks(args)

    assert checks
    assert all(check["ok"] for check in checks)


def test_write_repro_package_records_commands_and_sources(tmp_path: Path) -> None:
    args = _args(tmp_path)
    commands = build_commands(args)
    checks = [{"name": "synthetic", "ok": True, "detail": "test"}]

    write_repro_package(args.artifact_dir, args=args, commands=commands, checks=checks)

    command_text = (args.artifact_dir / "command.sh").read_text()
    assert "export WANDB_MODE=offline" in command_text
    assert "python train.py --config-name train.yaml" in command_text
    assert "python plan.py model_name=point_maze_test_seed0" in command_text

    summary = json.loads((args.artifact_dir / "summary.json").read_text())
    assert summary["status"] == "prepared"
    assert summary["env"] == "point_maze"
    assert summary["python_exe"] == "python"
    assert summary["wandb_mode"] == "offline"
    assert summary["model_name"] == "point_maze_test_seed0"
    assert summary["extra_train_arg"] == []
    assert summary["extra_plan_arg"] == []
    environment = json.loads((args.artifact_dir / "environment.txt").read_text())
    assert environment["target_python_exe"] == "python"
    assert (args.artifact_dir / "sources.json").exists()
    assert (args.artifact_dir / "environment.txt").exists()
    assert (args.artifact_dir / "notes.md").exists()


def test_update_execution_status_records_failed_stage(tmp_path: Path) -> None:
    args = _args(tmp_path)
    commands = build_commands(args)
    checks = [{"name": "synthetic", "ok": True, "detail": "test"}]
    write_repro_package(args.artifact_dir, args=args, commands=commands, checks=checks)

    update_execution_status(args.artifact_dir, [{"stage": "train", "return_code": 1}])

    summary = json.loads((args.artifact_dir / "summary.json").read_text())
    assert summary["status"] == "execution_failed"
    assert summary["execution_results"] == [{"stage": "train", "return_code": 1}]
