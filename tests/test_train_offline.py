from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src.data.trajectory_window import RawTrajectory
from src.train.train_offline import (
    build_action_transform,
    run_training,
    split_gripper_action_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def test_train_offline_dry_run_writes_metrics_and_checkpoints(tmp_path: Path) -> None:
    run_dir = run_training(
        ROOT / "configs/libero_spatial_mlp.yaml",
        dry_run=True,
        max_steps=1,
        output_dir=tmp_path / "runs",
        run_id="dry_run",
        command=["python", "src/train/train_offline.py", "--dry_run"],
    )

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "command.txt").exists()
    assert (run_dir / "git_commit.txt").exists()
    assert (run_dir / "environment.txt").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "notes.md").exists()
    assert (run_dir / "command.sh").exists()
    assert (run_dir / "train.log").exists()
    assert (run_dir / "split.json").exists()
    assert (run_dir / "normalization_stats.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "summary.json").exists()
    assert "total_loss,action_loss,future_loss" in (
        run_dir / "train.log"
    ).read_text(encoding="utf-8")

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["split"] for row in rows] == ["train", "val"]
    assert all(float(row["action_mse"]) >= 0.0 for row in rows)
    assert all(row["action_mse_units"] == "raw_action_units" for row in rows)
    assert all(int(row["parameter_count"]) > 0 for row in rows)
    assert all(int(row["trainable_parameter_count"]) > 0 for row in rows)
    assert all(row["lower_is_better"] == "true" for row in rows)


def test_train_offline_gru_dry_run_writes_metrics_and_parameter_count(
    tmp_path: Path,
) -> None:
    run_dir = run_training(
        ROOT / "configs/libero_spatial_gru.yaml",
        dry_run=True,
        max_steps=1,
        output_dir=tmp_path / "runs",
        run_id="gru_dry_run",
        command=["python3", "src/train/train_offline.py", "--dry_run"],
    )

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["split"] for row in rows] == ["train", "val"]
    assert all(float(row["action_mse"]) >= 0.0 for row in rows)
    assert all(row["action_mse_units"] == "raw_action_units" for row in rows)
    assert all(int(row["parameter_count"]) > 0 for row in rows)
    assert all(int(row["trainable_parameter_count"]) > 0 for row in rows)
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "best.pt").exists()


def test_train_offline_wam_gru_dry_run_writes_future_latent_metrics(
    tmp_path: Path,
) -> None:
    run_dir = run_training(
        ROOT / "configs/smoke/libero_spatial_wam_gru.yaml",
        dry_run=True,
        max_steps=1,
        output_dir=tmp_path / "runs",
        run_id="wam_gru_dry_run",
        command=["python3", "src/train/train_offline.py", "--dry_run"],
    )

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["split"] for row in rows] == ["train", "val"]
    assert all(float(row["future_loss"]) >= 0.0 for row in rows)
    assert all(float(row["future_latent_cosine_error"]) >= 0.0 for row in rows)
    assert all(
        len(json.loads(row["future_latent_cosine_error_by_horizon"])) == 4
        for row in rows
    )
    normalization = json.loads(
        (run_dir / "normalization_stats.json").read_text(encoding="utf-8")
    )
    assert normalization["visual_latents"]["encoder"]["encoder_id"] == "smoke_time_index"
    assert normalization["visual_latents"]["target_only_future"] is True
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "best.pt").exists()


def test_train_offline_wam_gru_no_future_dry_run_keeps_future_eval_metrics(
    tmp_path: Path,
) -> None:
    run_dir = run_training(
        ROOT / "configs/smoke/libero_spatial_gru_no_future.yaml",
        dry_run=True,
        max_steps=1,
        output_dir=tmp_path / "runs",
        run_id="wam_gru_no_future_dry_run",
        command=["python3", "src/train/train_offline.py", "--dry_run"],
    )

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["split"] for row in rows] == ["train", "val"]
    assert all(float(row["future_loss"]) >= 0.0 for row in rows)
    assert all(float(row["future_latent_cosine_error"]) >= 0.0 for row in rows)
    assert all(
        float(row["total_loss"]) == pytest.approx(float(row["action_loss"]))
        for row in rows
    )
    assert (run_dir / "train.log").exists()


def test_train_offline_bc_gru_dry_run_uses_latent_proprio_task_inputs(
    tmp_path: Path,
) -> None:
    run_dir = run_training(
        ROOT / "configs/smoke/libero_spatial_bc_gru.yaml",
        dry_run=True,
        max_steps=1,
        output_dir=tmp_path / "runs",
        run_id="bc_gru_dry_run",
        command=["python3", "src/train/train_offline.py", "--dry_run"],
    )

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["split"] for row in rows] == ["train", "val"]
    assert all(float(row["action_mse"]) >= 0.0 for row in rows)
    assert all(float(row["future_loss"]) == 0.0 for row in rows)
    normalization = json.loads(
        (run_dir / "normalization_stats.json").read_text(encoding="utf-8")
    )
    assert normalization["visual_latents"]["current_latent_input"] is True
    assert normalization["visual_latents"]["target_only_future"] is False
    assert (run_dir / "best.pt").exists()


def test_missing_libero_dataset_root_fails_with_clear_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIBERO_DATASET_ROOT", raising=False)

    with pytest.raises(OSError, match="LIBERO_DATASET_ROOT is not set"):
        run_training(
            ROOT / "configs/smoke/libero_spatial_action_only_smoke.yaml",
            output_dir=tmp_path / "runs",
            run_id="missing_env",
            command=["python3", "src/train/train_offline.py"],
        )

    assert not (tmp_path / "runs" / "missing_env").exists()


def test_trainer_action_standardization_uses_train_split_only() -> None:
    train = RawTrajectory(
        images=["train"] * 4,
        actions=[[0.0, 10.0], [2.0, 12.0], [4.0, 14.0], [6.0, 16.0]],
        language="train",
        trajectory_id="train_0",
        split="train",
    )
    val = RawTrajectory(
        images=["val"] * 4,
        actions=[[1000.0, 1010.0], [1002.0, 1012.0]],
        language="val",
        trajectory_id="val_0",
        split="val",
    )

    _, stats = build_action_transform(
        [train, val],
        {"normalization": {"actions": {"mode": "standardize_train"}}},
    )

    assert stats["actions"]["source_split"] == "train"
    assert stats["actions"]["count"] == 4
    assert stats["actions"]["mean"] == [3.0, 13.0]
    assert stats["actions"]["reported_action_mse_units"] == "raw_action_units"


def test_split_gripper_action_loss_uses_continuous_dims_and_gripper_logits() -> None:
    target_actions = torch.tensor(
        [
            [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]],
            [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, -1.0]],
        ]
    )
    outputs = {
        "pred_continuous_actions": target_actions[..., :6].clone(),
        "pred_gripper_logits": torch.tensor([[20.0], [-20.0]]),
    }

    loss = split_gripper_action_loss(outputs, target_actions)

    assert float(loss.item()) < 1e-6
