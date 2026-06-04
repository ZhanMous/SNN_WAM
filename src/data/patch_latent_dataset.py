"""Patch-latent transition dataset for DINO-WM training.

Loads cached DINOv2 patch latents from `.pt` files and creates
transition windows with shapes:

- z_context: `[B, T, P, D]` (current patch latents)
- actions: `[B, T, A]` (context action history)
- future_actions: `[B, H, A]` (candidate/action-conditioned future actions)
- z_target: `[B, H, P, D]` (future patch latents)

The dataset enforces causal alignment:
- Context at time t: patch_latents[t-T+1:t+1], actions before t
- Candidate future actions: actions[t:t+H]
- Targets: patch_latents[t+1:t+1+H]

No future patch latent target leaks into inputs. `future_actions` are explicit
world-model conditioning inputs for planning, not prediction targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data.trajectory_window import (
    RawTrajectory,
    TrajectoryWindowDataset,
    valid_time_indices,
)


def load_patch_latent_cache(
    cache_dir: Path,
    max_demos: int | None = None,
    max_frames: int | None = None,
) -> list[RawTrajectory]:
    """Load cached patch latents and return RawTrajectory list.

    Args:
        cache_dir: Directory containing .pt files and metadata.json.
        max_demos: Maximum number of demos to load.
        max_frames: Maximum number of frames per demo.

    Returns:
        List of RawTrajectory objects with patch_latents populated.
    """
    import json

    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {cache_dir}")

    metadata = json.loads(metadata_path.read_text())
    assert metadata["encoder_type"] == "dinov2_patch", (
        f"Expected dinov2_patch encoder, got {metadata['encoder_type']}"
    )

    trajectories: list[RawTrajectory] = []
    demo_count = 0

    for pt_file in sorted(cache_dir.glob("*.pt")):
        if max_demos is not None and demo_count >= max_demos:
            break

        data = torch.load(pt_file, weights_only=False)

        for demo_path, demo_data in data.items():
            if max_demos is not None and demo_count >= max_demos:
                break

            patch_latents = demo_data["patch_latents"]  # [T, N, D]
            actions = demo_data["actions"]  # [T, A]
            timesteps = demo_data["timesteps"]  # [T]

            T = patch_latents.shape[0]
            if max_frames is not None:
                T = min(T, max_frames)
                patch_latents = patch_latents[:T]
                actions = actions[:T] if actions is not None else None
                timesteps = timesteps[:T]

            # Keep cached patch latents as tensors. The cache is usually float16
            # and can be many GB; converting to Python lists multiplies memory use.
            patch_latents = torch.as_tensor(patch_latents).detach().cpu().contiguous()
            if actions is None:
                actions = torch.zeros(T, 7, dtype=torch.float32)
            else:
                actions = (
                    torch.as_tensor(actions, dtype=torch.float32)
                    .detach()
                    .cpu()
                    .contiguous()
                )

            # Create a dummy image list (we don't need real images for latent-only training)
            images = list(range(T))

            # Extract task name from demo path
            task_name = demo_path.split("/")[-1] if "/" in demo_path else demo_path

            trajectory = RawTrajectory(
                images=images,
                actions=actions,
                language=task_name,
                patch_latents=patch_latents,
                trajectory_id=f"{pt_file.stem}:{demo_path}",
                split="train",
                task_name=task_name,
            )

            trajectories.append(trajectory)
            demo_count += 1

    return trajectories


class PatchLatentTransitionDataset:
    """Transition dataset for DINO-WM from cached patch latents.

    Creates windows with:
    - z_context: `[T_ctx, P, D]` patch latents as context
    - actions: `[T_ctx, A]` action history
    - future_actions: `[H, A]` candidate future actions
    - z_target: `[H, P, D]` future patch latents as target

    The dataset enforces causal alignment:
    - Context at time t: patch_latents[t-T_ctx+1:t+1]
    - Future actions: actions[t:t+H]
    - Target: patch_latents[t+1:t+1+H]
    - No future patch latent target leaks into context.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        context_len: int = 3,
        future_horizon: int = 2,
        max_demos: int | None = None,
        max_frames: int | None = None,
        split: str = "train",
    ) -> None:
        """Initialize the dataset.

        Args:
            cache_dir: Directory containing cached patch latents.
            context_len: Number of past timesteps for context.
            future_horizon: Number of future timesteps to predict.
            max_demos: Maximum number of demos to load.
            max_frames: Maximum number of frames per demo.
            split: Dataset split label.
        """
        if context_len <= 0:
            raise ValueError("context_len must be positive")
        if future_horizon <= 0:
            raise ValueError("future_horizon must be positive")

        self.context_len = context_len
        self.future_horizon = future_horizon
        self.split = split

        # Load trajectories
        self.trajectories = load_patch_latent_cache(
            cache_dir, max_demos=max_demos, max_frames=max_frames
        )

        # Build index: (trajectory_index, time_index)
        self._index: list[tuple[int, int]] = []
        for traj_idx, traj in enumerate(self.trajectories):
            if traj.patch_latents is None:
                raise ValueError("patch_latents are required")
            T = len(traj.patch_latents)
            # Valid times: need context_len past + future_horizon future
            # At time t, context is [t-context_len+1:t+1], target is [t+1:t+1+future_horizon]
            for t in range(context_len - 1, T - future_horizon):
                self._index.append((traj_idx, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one transition window.

        Returns dict with:
        - z_context: [context_len, P, D] patch latents
        - actions: [context_len, A] action history
        - future_actions: [future_horizon, A] candidate future actions
        - z_target: [future_horizon, P, D] future patch latents
        - metadata: dict with trajectory_id, time_index, etc.
        """
        traj_idx, t = self._index[index]
        traj = self.trajectories[traj_idx]

        if traj.patch_latents is None:
            raise RuntimeError("patch_latents unexpectedly missing")

        patch_latents = torch.as_tensor(traj.patch_latents)
        actions = torch.as_tensor(traj.actions, dtype=torch.float32)

        # Context: [t-context_len+1:t+1]
        ctx_start = t - self.context_len + 1
        ctx_end = t + 1
        z_context = patch_latents[ctx_start:ctx_end].to(
            dtype=torch.float32
        )  # [context_len, P, D]
        # Action history is strictly before the candidate action at t. For the
        # first valid windows this needs left padding because there are fewer
        # than context_len previous actions.
        history_indices = list(range(t - self.context_len, t))
        action_history = torch.zeros(
            self.context_len,
            actions.shape[-1],
            dtype=actions.dtype,
            device=actions.device,
        )  # [context_len, A]
        valid_history_indices = [idx for idx in history_indices if idx >= 0]
        if valid_history_indices:
            valid_actions = actions[
                valid_history_indices[0] : valid_history_indices[-1] + 1
            ]
            action_history[-len(valid_history_indices) :] = valid_actions

        # Candidate future actions: action[t] should move z[t] -> z[t+1].
        future_action_start = t
        future_action_end = t + self.future_horizon
        future_actions = actions[future_action_start:future_action_end]  # [future_horizon, A]

        # Target: [t+1:t+1+future_horizon]
        tgt_start = t + 1
        tgt_end = t + 1 + self.future_horizon
        z_target = patch_latents[tgt_start:tgt_end].to(
            dtype=torch.float32
        )  # [future_horizon, P, D]

        return {
            "z_context": z_context,
            "actions": action_history,
            "future_actions": future_actions,
            "z_target": z_target,
            "metadata": {
                "trajectory_id": traj.trajectory_id,
                "time_index": t,
                "context_range": list(range(ctx_start, ctx_end)),
                "action_history_range": history_indices,
                "future_action_range": list(range(future_action_start, future_action_end)),
                "target_range": list(range(tgt_start, tgt_end)),
                "split": self.split,
            },
        }

    @property
    def patch_dim(self) -> tuple[int, int]:
        """Return (num_patches, feature_dim) from first trajectory."""
        if not self.trajectories:
            raise ValueError("No trajectories loaded")
        if self.trajectories[0].patch_latents is None:
            raise ValueError("No patch latents loaded")
        sample = torch.as_tensor(self.trajectories[0].patch_latents[0])
        return sample.shape[0], sample.shape[1]

    @property
    def action_dim(self) -> int:
        """Return action dimension from first trajectory."""
        if not self.trajectories:
            raise ValueError("No trajectories loaded")
        sample = torch.as_tensor(self.trajectories[0].actions[0])
        return sample.shape[0]


def create_dinowm_transition_dataset(
    cache_dir: Path,
    *,
    context_len: int = 3,
    future_horizon: int = 2,
    max_demos: int | None = None,
    max_frames: int | None = None,
    split: str = "train",
) -> PatchLatentTransitionDataset:
    """Factory function to create DINO-WM transition dataset.

    Args:
        cache_dir: Path to patch latent cache directory.
        context_len: Number of past timesteps for context.
        future_horizon: Number of future timesteps to predict.
        max_demos: Maximum number of demos to load.
        max_frames: Maximum number of frames per demo.
        split: Dataset split label.

    Returns:
        PatchLatentTransitionDataset instance.
    """
    return PatchLatentTransitionDataset(
        cache_dir,
        context_len=context_len,
        future_horizon=future_horizon,
        max_demos=max_demos,
        max_frames=max_frames,
        split=split,
    )


__all__ = [
    "PatchLatentTransitionDataset",
    "create_dinowm_transition_dataset",
    "load_patch_latent_cache",
]
