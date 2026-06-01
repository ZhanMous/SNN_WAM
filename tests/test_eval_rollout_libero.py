"""Tests for LIBERO closed-loop rollout evaluation scaffold.

Uses mock environment only. Mock rollouts must NOT be cited as real
success-rate evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

pytest.importorskip("torch")
import torch

from src.eval.eval_rollout_libero import (
    EpisodeResult,
    MockLIBEROEnv,
    extract_action_chunk,
    run_single_episode,
    run_rollout_evaluation,
)
from src.models.registry import build_offline_model


# ---------------------------------------------------------------------------
# MockLIBEROEnv tests
# ---------------------------------------------------------------------------


def test_mock_env_reset_returns_obs() -> None:
    env = MockLIBEROEnv()
    obs = env.reset(seed=42)
    assert "agentview_image" in obs
    assert obs["agentview_image"].shape == (128, 128, 3)


def test_mock_env_step_returns_tuple() -> None:
    env = MockLIBEROEnv(episode_length=5)
    env.reset()
    action = np.zeros(7, dtype=np.float32)
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_mock_env_done_after_episode_length() -> None:
    env = MockLIBEROEnv(episode_length=3)
    env.reset()
    for i in range(3):
        _, _, done, _ = env.step(np.zeros(7))
        if i < 2:
            assert not done
        else:
            assert done


def test_mock_env_success_on_configured_step() -> None:
    env = MockLIBEROEnv(episode_length=10, success_on_step=3)
    env.reset()
    for i in range(3):
        obs, reward, done, info = env.step(np.zeros(7))
    assert info.get("success", False) is True
    assert reward == 1.0


def test_mock_env_action_dim() -> None:
    env = MockLIBEROEnv(action_dim=7)
    assert env.action_dim == 7


def test_mock_env_task_name() -> None:
    env = MockLIBEROEnv(task_name="pick_up_bowl")
    assert env.task_name == "pick_up_bowl"


def test_extract_action_chunk_converts_split_gripper_logits_to_env_commands() -> None:
    outputs = {
        "pred_continuous_actions": torch.zeros(2, 1, 6),
        "pred_gripper_logits": torch.tensor([[-0.1], [0.0]]),
    }

    action_chunk, already_env_space = extract_action_chunk(outputs)

    assert already_env_space is True
    assert tuple(action_chunk.shape) == (2, 1, 7)
    assert action_chunk[:, 0, -1].tolist() == [-1.0, 1.0]


# ---------------------------------------------------------------------------
# Episode runner tests
# ---------------------------------------------------------------------------


def _make_mock_config(action_dim: int = 7, latent_dim: int = 384) -> dict:
    return {
        "experiment": {"name": "test", "seed": 0, "tags": []},
        "data": {
            "suite": "libero_spatial",
            "dataset_root": "env:LIBERO_DATASET_ROOT",
            "history_len": 2,
            "action_horizon": 4,
            "future_horizon": 4,
            "image_size": 128,
            "action_dim": action_dim,
            "latent_dim": latent_dim,
            "split": {"source": "splits/mock_split.json", "unit": "episode", "train": [], "val": [], "test": []},
        },
        "model": {
            "visual_encoder": "stub",
            "text_encoder": "stub",
            "temporal_adapter": "gru",
            "hidden_dim": 32,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "optimizer": "adamw",
            "lr": 0.001,
            "lambda_action": 1.0,
            "lambda_future": 0.0,
            "lambda_spike": 0.0,
            "grad_clip_norm": None,
        },
        "output": {"output_dir": "/tmp/mock_out", "save_best_by": "val/action_mse"},
    }


def _make_mock_model(config: dict) -> torch.nn.Module:
    action_dim = int(config["data"]["action_dim"])
    latent_dim = int(config["data"].get("latent_dim", 384))
    return build_offline_model(config, action_dim=action_dim, latent_dim=latent_dim)


def test_run_single_episode_returns_result() -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    env = MockLIBEROEnv(episode_length=5, action_dim=7, success_on_step=4)

    result = run_single_episode(
        model,
        env,
        device=torch.device("cpu"),
        max_steps=10,
        action_dim=7,
        history_len=2,
        action_horizon=4,
        action_transform=None,
        encoder=None,
        is_wam=False,
        seed=0,
    )

    assert isinstance(result, EpisodeResult)
    assert result.steps > 0
    assert result.steps <= 10
    assert result.task_name == "mock_task"


def test_run_single_episode_success_detected() -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    env = MockLIBEROEnv(episode_length=5, action_dim=7, success_on_step=3)

    result = run_single_episode(
        model,
        env,
        device=torch.device("cpu"),
        max_steps=10,
        action_dim=7,
        history_len=2,
        action_horizon=4,
        action_transform=None,
        encoder=None,
        is_wam=False,
        seed=0,
    )

    assert result.success is True
    assert result.total_reward > 0


def test_run_single_episode_max_steps_truncation() -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    env = MockLIBEROEnv(episode_length=100, action_dim=7)

    result = run_single_episode(
        model,
        env,
        device=torch.device("cpu"),
        max_steps=5,
        action_dim=7,
        history_len=2,
        action_horizon=4,
        action_transform=None,
        encoder=None,
        is_wam=False,
        seed=0,
    )

    assert result.steps == 5


def test_run_single_episode_records_video(tmp_path: Path) -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    env = MockLIBEROEnv(episode_length=3, action_dim=7)
    video_dir = tmp_path / "videos"

    result = run_single_episode(
        model,
        env,
        device=torch.device("cpu"),
        max_steps=5,
        action_dim=7,
        history_len=2,
        action_horizon=4,
        action_transform=None,
        encoder=None,
        is_wam=False,
        seed=0,
        record_video=True,
        media_dir=video_dir,
        episode_id=0,
    )

    assert result.video_path != ""
    assert Path(result.video_path).exists()
    assert "failure_videos" in result.video_path


def test_run_single_episode_moves_wam_latent_to_policy_device() -> None:
    class CpuLatentEncoder:
        def eval(self) -> None:
            pass

        def encode(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(obs.shape[0], 384, device=torch.device("cpu"))

    class DeviceCheckingWAM(torch.nn.Module):
        def forward(
            self,
            action_history: torch.Tensor,
            z_t: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            assert z_t.device == action_history.device
            batch_size = action_history.shape[0]
            return {
                "pred_actions": torch.zeros(
                    batch_size,
                    1,
                    action_history.shape[-1],
                    device=action_history.device,
                )
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MockLIBEROEnv(episode_length=1, action_dim=7)
    result = run_single_episode(
        DeviceCheckingWAM().to(device),
        env,
        device=device,
        max_steps=1,
        action_dim=7,
        history_len=2,
        action_horizon=1,
        action_transform=None,
        encoder=CpuLatentEncoder(),
        is_wam=True,
        seed=0,
    )

    assert result.steps == 1


# ---------------------------------------------------------------------------
# Full rollout evaluation tests (mock mode)
# ---------------------------------------------------------------------------


def _write_mock_checkpoint(
    run_dir: Path,
    config: dict,
    model: torch.nn.Module,
) -> None:
    """Write a minimal checkpoint for testing."""
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "config": config,
        "best_metric": 0.0,
        "best_epoch": 0,
        "metrics": {"train": {}, "val": {}},
    }
    torch.save(checkpoint, run_dir / "best.pt")
    (run_dir / "config.yaml").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    # Write normalization stats
    (run_dir / "normalization_stats.json").write_text(
        json.dumps({"actions": {"mode": "none"}}),
        encoding="utf-8",
    )


def test_run_rollout_evaluation_mock(tmp_path: Path) -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    run_dir = tmp_path / "run"
    _write_mock_checkpoint(run_dir, config, model)

    csv_path = run_rollout_evaluation(
        run_dir=run_dir,
        suite="libero_spatial",
        task_ids=[0],
        num_episodes=2,
        max_steps=5,
        seed=42,
        mock=True,
    )

    assert csv_path.exists()
    assert csv_path.name == "eval_rollout.csv"

    # Check CSV content
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 3  # header + 2 episodes
    assert "episode_id" in lines[0]
    assert "success" in lines[0]


def test_run_rollout_evaluation_writes_metadata(tmp_path: Path) -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    run_dir = tmp_path / "run"
    _write_mock_checkpoint(run_dir, config, model)

    run_rollout_evaluation(
        run_dir=run_dir,
        num_episodes=1,
        max_steps=3,
        mock=True,
    )

    eval_dir = run_dir / "eval_rollout"
    assert (eval_dir / "eval_command.txt").exists()
    assert (eval_dir / "command.txt").exists()
    assert (eval_dir / "git_commit.txt").exists()
    assert (eval_dir / "config.yaml").exists()
    assert (eval_dir / "environment.json").exists()
    assert (eval_dir / "environment.txt").exists()
    assert (eval_dir / "checkpoint_path.txt").exists()
    assert (eval_dir / "compatibility_report.json").exists()
    assert (eval_dir / "summary.json").exists()
    assert (eval_dir / "eval_summary.json").exists()
    assert (eval_dir / "notes.md").exists()

    summary = json.loads((eval_dir / "eval_summary.json").read_text())
    assert summary["status"] == "mock_eval_not_real_evidence"
    assert summary["mock"] is True
    assert "mock_env_not_real_libero" in summary["non_claims"]
    assert "failure_counts" in summary


def test_run_rollout_evaluation_multiple_tasks(tmp_path: Path) -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    run_dir = tmp_path / "run"
    _write_mock_checkpoint(run_dir, config, model)

    csv_path = run_rollout_evaluation(
        run_dir=run_dir,
        task_ids=[0, 1],
        num_episodes=2,
        max_steps=5,
        mock=True,
    )

    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 5  # header + 4 episodes (2 tasks x 2 episodes)


def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    """Verify --help exits with code 0."""
    from src.eval.eval_rollout_libero import parse_args
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--help"])
    assert exc_info.value.code == 0


def test_csv_fieldnames_match_data(tmp_path: Path) -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    run_dir = tmp_path / "run"
    _write_mock_checkpoint(run_dir, config, model)

    csv_path = run_rollout_evaluation(
        run_dir=run_dir,
        num_episodes=1,
        max_steps=3,
        mock=True,
    )

    import csv as csv_mod
    with csv_path.open() as f:
        reader = csv_mod.DictReader(f)
        row = next(reader)
        for field in [
            "run_id", "model", "checkpoint", "suite", "episode_id", "task_id",
            "task_name", "init_state_id", "seed", "success", "steps",
            "total_reward", "terminated", "truncated", "failure_reason",
            "video_path",
        ]:
            assert field in row, f"Missing field: {field}"


def test_mock_success_rate_not_cited_as_real() -> None:
    """Verify that mock eval summary always carries non-claims."""
    config = _make_mock_config()
    model = _make_mock_model(config)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        _write_mock_checkpoint(run_dir, config, model)
        run_rollout_evaluation(run_dir=run_dir, num_episodes=1, max_steps=3, mock=True)
        summary = json.loads((run_dir / "eval_rollout" / "eval_summary.json").read_text())
        assert "mock_env_not_real_libero" in summary["non_claims"]
        assert "mock_success_rate_not_real_evidence" in summary["non_claims"]


def test_run_rollout_evaluation_uses_requested_init_state_ids(tmp_path: Path) -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    run_dir = tmp_path / "run"
    _write_mock_checkpoint(run_dir, config, model)

    csv_path = run_rollout_evaluation(
        run_dir=run_dir,
        num_episodes=2,
        init_state_ids=[3, 7],
        max_steps=3,
        mock=True,
    )

    import csv as csv_mod
    with csv_path.open() as f:
        rows = list(csv_mod.DictReader(f))
    assert [int(row["init_state_id"]) for row in rows] == [3, 7]


def test_run_rollout_evaluation_records_failure_media(tmp_path: Path) -> None:
    config = _make_mock_config()
    model = _make_mock_model(config)
    run_dir = tmp_path / "run"
    _write_mock_checkpoint(run_dir, config, model)

    csv_path = run_rollout_evaluation(
        run_dir=run_dir,
        num_episodes=1,
        max_steps=3,
        record_video=True,
        mock=True,
    )

    import csv as csv_mod
    with csv_path.open() as f:
        row = next(csv_mod.DictReader(f))
    assert row["success"] == "False"
    assert "failure_videos" in row["video_path"]
    assert Path(row["video_path"]).exists()
