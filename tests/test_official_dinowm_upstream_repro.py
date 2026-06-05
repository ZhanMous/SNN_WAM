"""Tests for the official upstream DINO-WM reproduction helper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from scripts.reproduce_official_dinowm_upstream import (
    build_commands,
    collect_plan_artifacts,
    collect_train_artifacts,
    execute_command,
    find_latest_plan_output_dir,
    normalize_paths,
    parse_extra_env,
    prepare_dinov2_main_cache,
    preflight_checks,
    update_execution_status,
    write_plan_metrics_csv,
    write_train_metrics_csv,
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
        torch_home=None,
        dinov2_github_ref=None,
        dinov2_cache_source=None,
        dinov2_checkpoint_source=None,
        pin_dinov2_main_cache=False,
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
        extra_env=[],
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
    command_txt = (args.artifact_dir / "command.txt").read_text()
    assert "export WANDB_MODE=offline" in command_text
    assert f"cd {args.upstream_dir}" in command_text
    assert "python train.py --config-name train.yaml" in command_text
    assert "python plan.py model_name=point_maze_test_seed0" in command_text
    assert command_txt == command_text

    config = json.loads((args.artifact_dir / "config.yaml").read_text())
    assert config["env"] == "point_maze"
    assert config["commands"]["train"] == commands.train

    summary = json.loads((args.artifact_dir / "summary.json").read_text())
    assert summary["status"] == "prepared"
    assert summary["env"] == "point_maze"
    assert summary["python_exe"] == "python"
    assert summary["torch_home"] is None
    assert summary["dinov2_github_ref"] is None
    assert summary["pin_dinov2_main_cache"] is False
    assert summary["wandb_mode"] == "offline"
    assert summary["model_name"] == "point_maze_test_seed0"
    assert summary["extra_train_arg"] == []
    assert summary["extra_plan_arg"] == []
    assert summary["extra_env"] == []
    environment = json.loads((args.artifact_dir / "environment.txt").read_text())
    assert environment["target_python_exe"] == "python"
    assert environment["target_torch_home"] is None
    assert (args.artifact_dir / "sources.json").exists()
    assert (args.artifact_dir / "environment.txt").exists()
    assert (args.artifact_dir / "git_commit.txt").exists()
    assert (args.artifact_dir / "seeds.txt").read_text() == "training.seed=0\n"
    assert json.loads((args.artifact_dir / "split.json").read_text())["env"] == "point_maze"
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


def test_prepare_dinov2_main_cache_installs_artifact_local_cache(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.torch_home = args.artifact_dir / "torch_home"
    args.dinov2_github_ref = "b" * 40
    args.dinov2_cache_source = tmp_path / "source_dinov2"
    args.dinov2_checkpoint_source = tmp_path / "dinov2_vits14_pretrain.pth"
    args.pin_dinov2_main_cache = True
    args.dinov2_cache_source.mkdir()
    (args.dinov2_cache_source / "hubconf.py").write_text("dependencies = ['torch']\n")
    args.dinov2_checkpoint_source.write_bytes(b"checkpoint")

    records = prepare_dinov2_main_cache(args)

    cache_dir = args.torch_home / "hub" / "facebookresearch_dinov2_main"
    assert (cache_dir / "hubconf.py").exists()
    assert (cache_dir / "DINOV2_GITHUB_REF.txt").read_text() == ("b" * 40) + "\n"
    checkpoint = args.torch_home / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
    assert checkpoint.read_bytes() == b"checkpoint"
    assert [record["name"] for record in records] == ["dinov2_main_cache", "dinov2_checkpoint"]

    checks = preflight_checks(args)
    assert any(check["name"] == "dinov2_main_cache_installed" and check["ok"] for check in checks)


def test_prepare_dinov2_main_cache_rejects_global_torch_home(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.torch_home = tmp_path / "outside_torch_home"
    args.dinov2_github_ref = "b" * 40
    args.dinov2_cache_source = tmp_path / "source_dinov2"
    args.pin_dinov2_main_cache = True
    args.dinov2_cache_source.mkdir()

    with pytest.raises(ValueError, match="inside --artifact_dir"):
        prepare_dinov2_main_cache(args)


def test_write_train_metrics_csv_parses_official_train_log(tmp_path: Path) -> None:
    args = _args(tmp_path)
    output_dir = args.ckpt_base_path / "outputs" / args.model_name
    output_dir.mkdir(parents=True)
    (output_dir / "train.log").write_text(
        "[2026-06-05][__main__][INFO] - Epoch 1  Training loss: 2.5175"
        "                  Validation loss: 2.2880\n"
    )

    metrics_path = write_train_metrics_csv(args.artifact_dir, args)

    assert metrics_path == args.artifact_dir / "metrics.csv"
    rows = (args.artifact_dir / "metrics.csv").read_text().splitlines()
    assert rows[0] == "epoch,train_loss,val_loss,source_log"
    assert rows[1].startswith("1,2.5175,2.288,")


def test_collect_train_artifacts_records_config_and_checkpoint_pointer(tmp_path: Path) -> None:
    args = _args(tmp_path)
    output_dir = args.ckpt_base_path / "outputs" / args.model_name
    (output_dir / ".hydra").mkdir(parents=True)
    (output_dir / ".hydra" / "config.yaml").write_text("env: point_maze\n")
    (output_dir / ".hydra" / "overrides.yaml").write_text("- env=point_maze\n")
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "model_latest.pth").write_bytes(b"checkpoint")

    records = collect_train_artifacts(args.artifact_dir, args)

    assert (args.artifact_dir / "official_train_config.yaml").read_text() == "env: point_maze\n"
    assert (args.artifact_dir / "official_train_overrides.yaml").exists()
    assert (args.artifact_dir / "checkpoint.pt").read_bytes() == b"checkpoint"
    assert records["checkpoint"] == str(args.artifact_dir / "checkpoint.pt")
    assert records["checkpoint_source"].endswith("model_latest.pth")


def test_collect_plan_artifacts_and_metrics_from_upstream_output(tmp_path: Path) -> None:
    args = _args(tmp_path)
    plan_output = (
        args.upstream_dir
        / "plan_outputs"
        / f"20260605152413_{args.model_name}_gH{args.goal_H}"
    )
    (plan_output / ".hydra").mkdir(parents=True)
    (plan_output / ".hydra" / "config.yaml").write_text("planner: cem\n")
    (plan_output / ".hydra" / "overrides.yaml").write_text("- planner=cem\n")
    (plan_output / "logs.json").write_text(
        json.dumps(
            {
                "plan_0/success_rate": 0.0,
                "plan_0/mean_state_dist": 1.7,
                "plan_0/loss": 2.5,
            }
        )
        + "\n"
        + json.dumps(
            {
                "final_eval/success_rate": 1.0,
                "final_eval/mean_state_dist": 0.2,
                "final_eval/mean_visual_dist": 0.1,
                "final_eval/mean_proprio_dist": 0.3,
            }
        )
        + "\n"
    )
    log_path = args.artifact_dir / "plan.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(f"Planning result saved dir: {plan_output}\n")

    assert find_latest_plan_output_dir(args, log_path) == plan_output
    records = collect_plan_artifacts(args.artifact_dir, args, log_path)
    metrics_path = write_plan_metrics_csv(
        args.artifact_dir,
        args=args,
        plan_output_dir=plan_output,
        log_path=log_path,
    )

    assert records["plan_output_dir"] == str(plan_output)
    assert (args.artifact_dir / "plan_outputs.txt").read_text() == str(plan_output) + "\n"
    assert (args.artifact_dir / "official_plan_config.yaml").read_text() == "planner: cem\n"
    assert metrics_path == args.artifact_dir / "metrics.csv"
    rows = (args.artifact_dir / "metrics.csv").read_text().splitlines()
    assert rows[0] == (
        "stage,success_rate,state_dist,visual_dist,proprio_dist,plan_loss,"
        "source_log,plan_output_dir"
    )
    assert rows[1].startswith("plan,1.0,0.2,0.1,0.3,2.5,")


def test_execute_command_writes_combined_log(tmp_path: Path) -> None:
    log_path = tmp_path / "stage.log"

    rc = execute_command(
        [sys.executable, "-c", "print('stage-ok')"],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=log_path,
    )

    assert rc == 0
    assert "stage-ok" in log_path.read_text()


def test_parse_extra_env_requires_key_value() -> None:
    assert parse_extra_env(["A=1", "B=two=parts"]) == {"A": "1", "B": "two=parts"}

    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_extra_env(["broken"])
