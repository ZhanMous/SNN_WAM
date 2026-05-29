"""LIBERO closed-loop rollout evaluation scaffold.

Loads a trained checkpoint, instantiates the LIBERO environment, and runs
closed-loop rollouts.  Writes one row per episode to ``eval_rollout.csv``.

This script is a scaffold: the real LIBERO environment path is guarded by
an import check.  A mock environment is used for testing when LIBERO is
unavailable.  Mock rollouts must NOT be cited as real success-rate evidence.

Usage::

    python -m src.eval.eval_rollout_libero \\
        --run_dir results/runs/<run_id> \\
        --suite libero_spatial \\
        --num_episodes 5 \\
        --max_steps 300

"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.encoders import build_frozen_visual_encoder  # noqa: E402
from src.models.registry import build_offline_model  # noqa: E402
from src.train.eval_offline import load_checkpoint  # noqa: E402
from src.train.train_offline import (  # noqa: E402
    ActionTransform,
    build_datasets,
    has_future_latent_targets,
    infer_action_dim,
    infer_latent_dim,
    validate_training_scope,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLOUT_FIELDNAMES = [
    "run_id",
    "model",
    "checkpoint",
    "suite",
    "episode_id",
    "task_id",
    "task_name",
    "init_state_id",
    "seed",
    "success",
    "steps",
    "total_reward",
    "terminated",
    "truncated",
    "failure_reason",
    "video_path",
]

REQUIRED_IMPORTS = [
    "libero",
    "libero.libero.benchmark",
    "libero.libero.envs",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    episode_id: int
    task_id: int
    task_name: str
    init_state_id: int
    seed: int
    success: bool
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    failure_reason: str = ""
    video_path: str = ""


@dataclass
class RolloutConfig:
    suite: str
    task_ids: list[int]
    num_episodes: int
    max_steps: int
    seed: int
    record_video: bool
    init_state_ids: list[int] = field(default_factory=list)
    camera_height: int = 128
    camera_width: int = 128
    mujoco_gl: str = "egl"
    action_chunk_exec: int = 1  # how many actions from chunk to execute before re-query
    settle_steps: int = 5


# ---------------------------------------------------------------------------
# LIBERO environment abstraction
# ---------------------------------------------------------------------------


class LIBEROEnvInterface:
    """Abstraction over LIBERO environment for testability."""

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def reset_to_init_state(
        self,
        init_state: Any | None,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        return self.reset(seed=seed)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        raise NotImplementedError

    def get_observation(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def task_name(self) -> str:
        return ""

    @property
    def task_id(self) -> int:
        return 0


class RealLIBEROEnv(LIBEROEnvInterface):
    """Real LIBERO OffScreenRenderEnv wrapper."""

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        camera_height: int = 128,
        camera_width: int = 128,
        mujoco_gl: str = "egl",
    ) -> None:
        os.environ.setdefault("MUJOCO_GL", mujoco_gl)

        from libero.libero import benchmark  # type: ignore[import-not-found]
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]
        from libero.libero.utils import get_libero_path  # type: ignore[import-not-found]

        benchmark_dict = benchmark.get_benchmark_dict()
        if suite_name not in benchmark_dict:
            available = sorted(benchmark_dict)
            raise ValueError(
                f"Unknown suite '{suite_name}'. Available: {available}"
            )
        suite = benchmark_dict[suite_name]()
        task = suite.get_task(task_id)
        self._task = task
        self._task_id = task_id

        bddl_name = task.bddl_file if hasattr(task, "bddl_file") else task.bddl_file_name
        bddl_file = Path(bddl_name)
        if not bddl_file.is_absolute():
            bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / bddl_name
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=camera_height,
            camera_widths=camera_width,
        )
        self._obs: dict[str, Any] | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        self._obs = self._env.reset()
        return self._obs

    def reset_to_init_state(
        self,
        init_state: Any | None,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        obs = self.reset(seed=seed)
        if init_state is not None:
            obs = self._env.set_init_state(init_state)
            self._obs = obs
        return obs

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        result = self._env.step(action)
        if isinstance(result, tuple):
            if len(result) == 5:
                obs, reward, done, truncated, info = result
                self._obs = obs
                return obs, reward, done or truncated, info
            obs, reward, done, info = result
            self._obs = obs
            if isinstance(info, dict):
                try:
                    info = dict(info)
                    info.setdefault("success", bool(self._env.check_success()))
                except Exception:
                    pass
            return obs, reward, done, info
        return result, 0.0, False, {}

    def get_observation(self) -> np.ndarray:
        if self._obs is None:
            raise RuntimeError("Call reset() before get_observation()")
        # LIBERO returns a dict with 'agentview_image' or similar
        for key in ("agentview_image", "agentview_rgb", "image", "rgb"):
            if key in self._obs:
                return np.array(self._obs[key])
        # Fallback: first array-like value
        for value in self._obs.values():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                return value
        raise RuntimeError("Could not find image observation in env output")

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_dim(self) -> int:
        action_space = getattr(self._env, "action_space", None)
        if action_space is not None and getattr(action_space, "shape", None):
            return action_space.shape[0]
        return 7  # LIBERO default

    @property
    def task_name(self) -> str:
        for attr in ("language", "task_name", "name"):
            value = getattr(self._task, attr, None)
            if value:
                return str(value)
        return f"task_{self._task_id}"

    @property
    def task_id(self) -> int:
        return self._task_id


class MockLIBEROEnv(LIBEROEnvInterface):
    """Mock environment for testing when LIBERO is unavailable."""

    def __init__(
        self,
        task_id: int = 0,
        task_name: str = "mock_task",
        action_dim: int = 7,
        episode_length: int = 10,
        success_on_step: int = -1,
    ) -> None:
        self._task_id = task_id
        self._task_name = task_name
        self._action_dim = action_dim
        self._episode_length = episode_length
        self._success_on_step = success_on_step
        self._step_count = 0
        self._seed: int | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._seed = seed
        self._step_count = 0
        if seed is not None:
            np.random.seed(seed)
        return {"agentview_image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)}

    def reset_to_init_state(
        self,
        init_state: Any | None,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        return self.reset(seed=seed)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._step_count += 1
        done = self._step_count >= self._episode_length
        success = self._step_count == self._success_on_step
        reward = 1.0 if success else 0.0
        obs = {"agentview_image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)}
        info: dict[str, Any] = {"success": success}
        return obs, reward, done, info

    def get_observation(self) -> np.ndarray:
        return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    def close(self) -> None:
        pass

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def task_id(self) -> int:
        return self._task_id


# ---------------------------------------------------------------------------
# Rollout runner
# ---------------------------------------------------------------------------


def run_single_episode(
    model: torch.nn.Module,
    env: LIBEROEnvInterface,
    *,
    device: torch.device,
    max_steps: int,
    action_dim: int,
    history_len: int,
    action_horizon: int,
    action_transform: ActionTransform | None,
    encoder: Any | None,
    is_wam: bool,
    init_state: Any | None = None,
    init_state_id: int = -1,
    seed: int | None = None,
    record_video: bool = False,
    media_dir: Path | None = None,
    episode_id: int = 0,
    settle_steps: int = 0,
    action_chunk_exec: int = 1,
) -> EpisodeResult:
    """Run one closed-loop episode and return the result."""

    model.eval()
    if encoder is not None:
        encoder.eval()
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if action_chunk_exec <= 0:
        raise ValueError("action_chunk_exec must be positive")

    obs = env.reset_to_init_state(init_state, seed=seed)
    if init_state is not None and settle_steps > 0:
        settle_action = np.zeros(action_dim, dtype=np.float32)
        for _ in range(settle_steps):
            obs, _, done, _ = env.step(settle_action)
            if done:
                break
        done = False

    done = False
    total_reward = 0.0
    step_count = 0
    failure_reason = ""
    frames: list[np.ndarray] = []

    # Action history buffer: zero-padded initially
    action_history = torch.zeros(1, history_len, action_dim, device=device)
    # Action chunk buffer for executing multiple actions per model query
    action_chunk: torch.Tensor | None = None
    chunk_step = 0

    while not done and step_count < max_steps:
        # Get current image observation for WAM encoder
        z_t: torch.Tensor | None = None
        if is_wam and encoder is not None:
            obs_image = env.get_observation()
            obs_tensor = torch.from_numpy(obs_image).float()
            if obs_tensor.max() > 1.0:
                obs_tensor = obs_tensor / 255.0
            # HWC -> CHW
            if obs_tensor.ndim == 3 and obs_tensor.shape[-1] in (1, 3):
                obs_tensor = obs_tensor.permute(2, 0, 1)
            obs_tensor = obs_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                z_t = encoder.encode(obs_tensor).to(device)

        # Query model if we need new actions
        if (
            action_chunk is None
            or chunk_step >= action_chunk.shape[1]
            or chunk_step >= action_chunk_exec
        ):
            with torch.no_grad():
                if is_wam and z_t is not None:
                    outputs = model(action_history, z_t)
                    if isinstance(outputs, dict):
                        action_chunk = outputs["pred_actions"]
                    else:
                        action_chunk = outputs
                else:
                    action_chunk = model(action_history)
            chunk_step = 0

        # Take action from chunk
        model_action = action_chunk[0, chunk_step, :action_dim].cpu()
        chunk_step += 1

        # Denormalize if needed
        env_action = model_action
        if action_transform is not None:
            env_action = action_transform.denormalize_tensor(model_action)

        action_np = env_action.numpy()

        # Step environment
        obs, reward, done, info = env.step(action_np)
        total_reward += reward
        step_count += 1

        # The action history stays in the same units the model saw at training.
        action_history = torch.roll(action_history, shifts=-1, dims=1)
        action_history[0, -1, :] = model_action.to(device)

        # Record frame
        if record_video and media_dir is not None:
            obs_image = env.get_observation()
            frames.append(obs_image.copy())

        # Check for explicit failure
        if isinstance(info, dict) and "failure_reason" in info:
            failure_reason = str(info["failure_reason"])

    # Determine success
    success = False
    if isinstance(info, dict):
        success = bool(info.get("success", False))
    if total_reward > 0:
        success = True
    truncated = step_count >= max_steps and not success
    terminated = done and not truncated
    if not success and not failure_reason:
        failure_reason = "max_steps_reached" if truncated else "environment_done_without_success"

    # Save video if recorded
    video_path = ""
    if record_video and frames and media_dir is not None:
        video_dir = media_dir / ("success_videos" if success else "failure_videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(video_dir / f"episode_{episode_id:04d}.npy")
        np.save(video_path, np.stack(frames))

    return EpisodeResult(
        episode_id=episode_id,
        task_id=env.task_id,
        task_name=env.task_name,
        init_state_id=init_state_id,
        seed=seed if seed is not None else -1,
        success=success,
        steps=step_count,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        failure_reason=failure_reason,
        video_path=video_path,
    )


def run_rollout_evaluation(
    *,
    run_dir: Path,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    suite: str = "libero_spatial",
    task_ids: list[int] | None = None,
    num_episodes: int = 5,
    max_steps: int = 300,
    seed: int = 0,
    init_state_ids: list[int] | None = None,
    record_video: bool = False,
    action_chunk_exec: int = 1,
    settle_steps: int = 5,
    device_name: str = "cpu",
    mock: bool = False,
    command: Sequence[str] | None = None,
) -> Path:
    """Run closed-loop LIBERO rollout evaluation.

    Returns the path to eval_rollout.csv.
    """

    run_dir = run_dir.expanduser()
    config_path = config_path or run_dir / "config.yaml"
    checkpoint_path = checkpoint_path or run_dir / "best.pt"
    output_dir = run_dir / "eval_rollout"
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if action_chunk_exec <= 0:
        raise ValueError("action_chunk_exec must be positive")
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    config = load_config(config_path)
    validate_training_scope(config)
    seed_everything(seed)

    # Build model from checkpoint
    device = torch.device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)

    # Infer action_dim and latent_dim from config
    action_dim = int(config["data"].get("action_dim", 7))
    latent_dim = int(config["data"].get("latent_dim", 384)) if config["model"]["temporal_adapter"] == "wam_gru" else None

    model = build_offline_model(config, action_dim=action_dim, latent_dim=latent_dim)
    compatibility_report = build_compatibility_report(
        config=config,
        checkpoint=checkpoint,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        action_dim=action_dim,
        latent_dim=latent_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    compatibility_report["state_dict_load"] = "strict_pass"
    compatibility_report["compatible"] = not compatibility_report["mismatches"]
    (output_dir / "compatibility_report.json").write_text(
        json.dumps(compatibility_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model.to(device)
    model.eval()

    is_wam = config["model"]["temporal_adapter"] == "wam_gru"

    # Build encoder for WAM models
    encoder = None
    if is_wam:
        encoder = build_frozen_visual_encoder(config["model"])
        encoder.to(device)
        encoder.eval()

    # Load action transform from normalization stats if available
    action_transform = None
    norm_stats_path = run_dir / "normalization_stats.json"
    if norm_stats_path.exists():
        norm_stats = json.loads(norm_stats_path.read_text(encoding="utf-8"))
        action_stats = norm_stats.get("actions", {})
        if action_stats.get("mode") == "standardize_train":
            action_transform = ActionTransform(
                mean=tuple(action_stats["mean"]),
                std=tuple(action_stats["std"]),
            )

    history_len = int(config["data"]["history_len"])
    action_horizon = int(config["data"]["action_horizon"])

    # Determine task IDs
    if task_ids is None:
        task_ids = [0]

    # Set up environment
    media_dir = output_dir if record_video else None

    results: list[EpisodeResult] = []
    episode_id = 0
    model_name = str(config["model"].get("name", config["model"]["temporal_adapter"]))

    for task_id in task_ids:
        task_init_states = None if mock else load_task_init_states(suite, task_id)
        selected_init_state_ids = select_init_state_ids(
            num_episodes=num_episodes,
            requested=init_state_ids,
            available_count=None if task_init_states is None else int(len(task_init_states)),
        )
        for ep_idx, init_state_id in enumerate(selected_init_state_ids):
            ep_seed = seed + episode_id
            env: LIBEROEnvInterface | None = None
            try:
                init_state = None
                episode_settle_steps = 0
                if mock:
                    env = MockLIBEROEnv(
                        task_id=task_id,
                        task_name=f"mock_task_{task_id}",
                        action_dim=action_dim,
                        episode_length=min(max_steps, 10),
                        success_on_step=5 if ep_idx == 0 else -1,
                    )
                else:
                    env = RealLIBEROEnv(
                        suite_name=suite,
                        task_id=task_id,
                        camera_height=128,
                        camera_width=128,
                    )
                    if task_init_states is not None:
                        init_state = task_init_states[init_state_id]
                    episode_settle_steps = settle_steps

                result = run_single_episode(
                    model,
                    env,
                    device=device,
                    max_steps=max_steps,
                    action_dim=action_dim,
                    history_len=history_len,
                    action_horizon=action_horizon,
                    action_transform=action_transform,
                    encoder=encoder,
                    is_wam=is_wam,
                    init_state=init_state,
                    init_state_id=init_state_id,
                    seed=ep_seed,
                    record_video=record_video,
                    media_dir=media_dir,
                    episode_id=episode_id,
                    settle_steps=episode_settle_steps,
                    action_chunk_exec=action_chunk_exec,
                )
                results.append(result)
            except Exception as exc:
                results.append(
                    EpisodeResult(
                        episode_id=episode_id,
                        task_id=task_id,
                        task_name=getattr(env, "task_name", f"task_{task_id}") if env else f"task_{task_id}",
                        init_state_id=init_state_id,
                        seed=ep_seed,
                        success=False,
                        steps=0,
                        total_reward=0.0,
                        terminated=True,
                        truncated=False,
                        failure_reason=f"environment/evaluator error: {type(exc).__name__}: {exc}",
                        video_path="",
                    )
                )
            finally:
                if env is not None:
                    env.close()

            episode_id += 1

    # Write CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eval_rollout.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROLLOUT_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "run_id": run_dir.name,
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "suite": suite,
                "episode_id": r.episode_id,
                "task_id": r.task_id,
                "task_name": r.task_name,
                "init_state_id": r.init_state_id,
                "seed": r.seed,
                "success": r.success,
                "steps": r.steps,
                "total_reward": r.total_reward,
                "terminated": r.terminated,
                "truncated": r.truncated,
                "failure_reason": r.failure_reason,
                "video_path": r.video_path,
            })

    # Save metadata
    _write_metadata(
        output_dir=output_dir,
        run_dir=run_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        config=config,
        suite=suite,
        task_ids=task_ids,
        num_episodes=num_episodes,
        max_steps=max_steps,
        seed=seed,
        init_state_ids=init_state_ids,
        record_video=record_video,
        mock=mock,
        results=results,
        action_chunk_exec=action_chunk_exec,
        settle_steps=settle_steps,
        command=command,
    )

    return csv_path


def _write_metadata(
    *,
    output_dir: Path,
    run_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    config: dict[str, Any],
    suite: str,
    task_ids: list[int],
    num_episodes: int,
    max_steps: int,
    seed: int,
    init_state_ids: list[int] | None,
    record_video: bool,
    mock: bool,
    results: list[EpisodeResult],
    action_chunk_exec: int,
    settle_steps: int,
    command: Sequence[str] | None,
) -> None:
    """Write eval metadata files."""

    # Command
    cmd_str = " ".join(command) if command else " ".join(sys.argv)
    (output_dir / "command.txt").write_text(cmd_str + "\n", encoding="utf-8")
    (output_dir / "eval_command.txt").write_text(cmd_str + "\n", encoding="utf-8")
    (output_dir / "checkpoint_path.txt").write_text(
        str(checkpoint_path) + "\n", encoding="utf-8"
    )

    # Git commit
    git_info = _get_git_info()
    (output_dir / "git_commit.txt").write_text(
        f"commit={git_info.get('commit', 'unknown')}\n"
        f"dirty={git_info.get('dirty', 'unknown')}\n",
        encoding="utf-8",
    )

    # Config copy
    (output_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8")
        if config_path.exists()
        else json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Environment
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(env_info, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "environment.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in env_info.items()) + "\n",
        encoding="utf-8",
    )

    # Summary
    total = len(results)
    successes = sum(1 for r in results if r.success)
    failure_counts = Counter(
        r.failure_reason or "unclassified_failure"
        for r in results
        if not r.success
    )
    success_steps = [r.steps for r in results if r.success]
    is_smoke = bool(mock or total < 10 or max_steps < 100)
    summary = {
        "status": "mock_eval_not_real_evidence" if mock else "closed_loop_smoke" if is_smoke else "closed_loop_eval",
        "suite": suite,
        "task_ids": task_ids,
        "num_episodes": num_episodes,
        "init_state_ids": init_state_ids,
        "max_steps": max_steps,
        "seed": seed,
        "total_episodes": total,
        "successes": successes,
        "success_rate": successes / total if total > 0 else 0.0,
        "success_completion_steps": success_steps,
        "mean_completion_steps_successes": (
            sum(success_steps) / len(success_steps) if success_steps else None
        ),
        "failure_counts": dict(sorted(failure_counts.items())),
        "mock": mock,
        "record_video": record_video,
        "recorded_media": any(r.video_path for r in results),
        "action_chunk_exec": action_chunk_exec,
        "action_chunk_policy": (
            "receding_horizon_first_action"
            if action_chunk_exec == 1
            else f"execute_{action_chunk_exec}_actions_then_requery"
        ),
        "settle_steps_after_fixed_init": 0 if mock else settle_steps,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "csv_path": str(output_dir / "eval_rollout.csv"),
        "is_reportable": False,
        "reportability_reasons": _reportability_reasons(
            mock=mock,
            total_episodes=total,
            max_steps=max_steps,
            git_dirty=git_info.get("dirty", "unknown"),
        ),
        "non_claims": _non_claims(mock),
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    (output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    (output_dir / "eval_summary.json").write_text(
        summary_text,
        encoding="utf-8",
    )
    (output_dir / "notes.md").write_text(
        build_eval_notes(summary=summary, results=results),
        encoding="utf-8",
    )


def build_compatibility_report(
    *,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    action_dim: int,
    latent_dim: int | None,
) -> dict[str, Any]:
    """Compare the runtime config against checkpoint metadata before rollout."""

    checkpoint_config = checkpoint.get("config", {})
    compared_fields = [
        ("data.history_len", ("data", "history_len")),
        ("data.action_horizon", ("data", "action_horizon")),
        ("data.future_horizon", ("data", "future_horizon")),
        ("data.latent_dim", ("data", "latent_dim")),
        ("model.temporal_adapter", ("model", "temporal_adapter")),
        ("model.hidden_dim", ("model", "hidden_dim")),
        ("model.visual_encoder", ("model", "visual_encoder")),
        ("model.visual_latent_dim", ("model", "visual_latent_dim")),
        ("training.lambda_future", ("training", "lambda_future")),
    ]
    mismatches: list[dict[str, Any]] = []
    for label, keys in compared_fields:
        runtime_value = nested_get(config, keys)
        checkpoint_value = nested_get(checkpoint_config, keys)
        if runtime_value != checkpoint_value:
            mismatches.append(
                {
                    "field": label,
                    "runtime_config": runtime_value,
                    "checkpoint_config": checkpoint_value,
                }
            )

    return {
        "compatible": False,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_metric": checkpoint.get("best_metric"),
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "mismatches": mismatches,
        "state_dict_load": "not_attempted",
    }


def nested_get(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def load_task_init_states(suite_name: str, task_id: int) -> Any:
    from libero.libero import benchmark  # type: ignore[import-not-found]
    from libero.libero.utils import get_libero_path  # type: ignore[import-not-found]

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(benchmark_dict)}")
    suite = benchmark_dict[suite_name]()
    task = suite.get_task(task_id)
    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    try:
        return torch.load(init_states_path, weights_only=False)
    except TypeError:  # pragma: no cover - for older torch.
        return torch.load(init_states_path)


def select_init_state_ids(
    *,
    num_episodes: int,
    requested: list[int] | None,
    available_count: int | None,
) -> list[int]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    ids = list(requested) if requested is not None else list(range(num_episodes))
    if len(ids) < num_episodes:
        raise ValueError(
            f"Need at least num_episodes={num_episodes} init_state_ids, got {ids}"
        )
    ids = ids[:num_episodes]
    if available_count is not None:
        for init_state_id in ids:
            if init_state_id < 0 or init_state_id >= available_count:
                raise ValueError(
                    f"init_state_id {init_state_id} out of range for "
                    f"{available_count} available states"
                )
    return ids


def _reportability_reasons(
    *,
    mock: bool,
    total_episodes: int,
    max_steps: int,
    git_dirty: str,
) -> list[str]:
    reasons: list[str] = ["closed_loop_rollout_smoke"]
    if mock:
        reasons.append("mock_environment")
    if total_episodes < 10:
        reasons.append("limited_episode_count")
    if max_steps < 100:
        reasons.append("limited_max_steps")
    if git_dirty != "False":
        reasons.append(f"git_dirty={git_dirty}")
    return reasons


def build_eval_notes(
    *,
    summary: Mapping[str, Any],
    results: Sequence[EpisodeResult],
) -> str:
    failure_counts = summary.get("failure_counts", {})
    lines = [
        "# Rollout Evaluation Notes",
        "",
        f"Status: {summary['status']}",
        f"Suite: {summary['suite']}",
        f"Task IDs: {summary['task_ids']}",
        f"Episodes: {summary['total_episodes']}",
        f"Max policy steps: {summary['max_steps']}",
        f"Success rate: {summary['success_rate']}",
        "",
        "Failure counts:",
    ]
    if failure_counts:
        lines.extend(f"- {reason}: {count}" for reason, count in failure_counts.items())
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Evaluator limitations:",
            "- This rollout is a small smoke run unless summary.json marks it otherwise.",
            "- No model comparison is implied by this single-checkpoint evaluation.",
            "- The policy consumes only action history and the current observation latent; no demonstration actions or future observations are used during rollout.",
            "- Failure episodes remain in eval_rollout.csv and are not filtered.",
        ]
    )
    if any(result.video_path for result in results):
        lines.extend(["", "Recorded media:"])
        lines.extend(
            f"- episode {result.episode_id}: {result.video_path}"
            for result in results
            if result.video_path
        )
    return "\n".join(lines) + "\n"


def _non_claims(mock: bool) -> list[str]:
    claims = [
        "not_offline_metric",
        "not_model_comparison",
        "not_future_latent_improvement_evidence",
    ]
    if mock:
        claims.extend([
            "mock_env_not_real_libero",
            "mock_success_rate_not_real_evidence",
            "not_reportable",
        ])
    return claims


def _get_git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Check for dirty state excluding eval_rollout output directories.
        # Eval creates new files that would always report dirty=False otherwise.
        has_staged = subprocess.call(
            ["git", "diff", "--cached", "--quiet"],
            stderr=subprocess.DEVNULL,
        ) != 0
        has_unstaged = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        ) != 0
        # Untracked files outside eval_rollout directories
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().splitlines()
        has_untracked = any(
            line for line in untracked
            if line and "eval_rollout" not in line
        )
        dirty = has_staged or has_unstaged or has_untracked
        return {"commit": commit, "dirty": str(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": "unknown"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir", required=True, type=Path,
        help="Training run directory containing config.yaml and best.pt.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--suite", default="libero_spatial",
        help="LIBERO benchmark suite. Default: libero_spatial.",
    )
    parser.add_argument(
        "--task_ids", type=int, nargs="*", default=None,
        help="Task IDs to evaluate. Default: [0].",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=5,
        help="Episodes per task. Default: 5.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=300,
        help="Max steps per episode. Default: 300.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init_state_ids",
        type=int,
        nargs="*",
        default=None,
        help="Fixed LIBERO benchmark init-state IDs. Default: first num_episodes states.",
    )
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument(
        "--action_chunk_exec", type=int, default=1,
        help="Actions to execute from chunk before re-querying model. Default: 1.",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=5,
        help="Zero-action physics settle steps after fixed init state. Default: 5.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock environment instead of real LIBERO. For testing only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    csv_path = run_rollout_evaluation(
        run_dir=args.run_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        suite=args.suite,
        task_ids=args.task_ids,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        init_state_ids=args.init_state_ids,
        record_video=args.record_video,
        action_chunk_exec=args.action_chunk_exec,
        settle_steps=args.settle_steps,
        device_name=args.device,
        mock=args.mock,
        command=[sys.executable, "-m", "src.eval.eval_rollout_libero", *(argv or sys.argv[1:])],
    )
    print(f"eval_rollout_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
