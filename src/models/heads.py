"""Prediction heads for offline adapter baselines."""

from __future__ import annotations

import torch
from torch import nn


class ActionChunkHead(nn.Module):
    """Project model features to a future action chunk.

    Shape contract:

    - input `features`: `[B, D]`, floating point tensor on any PyTorch device.
    - output: `[B, H, A]`, where `H` is `action_horizon` and `A` is
      `action_dim`.
    """

    def __init__(self, input_dim: int, action_horizon: int, action_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.input_dim = input_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.projection = nn.Linear(input_dim, action_horizon * action_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return predicted actions with shape `[B, H, A]`."""

        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [B, D], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"features dim {features.shape[1]} does not match {self.input_dim}"
            )

        batch_size = features.shape[0]
        actions = self.projection(features)
        return actions.reshape(batch_size, self.action_horizon, self.action_dim)


class FutureLatentChunkHead(nn.Module):
    """Project model features to future visual latent targets.

    Shape contract:

    - input `features`: `[B, D]`, floating point tensor.
    - output: `[B, H, Z]`, where `H` is `future_horizon` and `Z` is
      `latent_dim`.
    """

    def __init__(self, input_dim: int, future_horizon: int, latent_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if future_horizon <= 0:
            raise ValueError("future_horizon must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.input_dim = input_dim
        self.future_horizon = future_horizon
        self.latent_dim = latent_dim
        self.projection = nn.Linear(input_dim, future_horizon * latent_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return predicted future latents with shape `[B, H, Z]`."""

        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [B, D], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"features dim {features.shape[1]} does not match {self.input_dim}"
            )

        batch_size = features.shape[0]
        latents = self.projection(features)
        return latents.reshape(batch_size, self.future_horizon, self.latent_dim)


def gripper_logits_to_command(
    logits: torch.Tensor,
    *,
    open_command: float = 1.0,
    close_command: float = -1.0,
) -> torch.Tensor:
    """Map gripper logits to environment-compatible open/close commands."""

    return torch.where(
        logits >= 0,
        torch.as_tensor(open_command, dtype=logits.dtype, device=logits.device),
        torch.as_tensor(close_command, dtype=logits.dtype, device=logits.device),
    )


class SplitActionGripperHead(nn.Module):
    """Project features to continuous action dims plus categorical gripper logits.

    Shape contract:

    - input `features`: `[B, D]`.
    - output `pred_continuous_actions`: `[B, H, A-1]`.
    - output `pred_gripper_logits`: `[B, H]`.
    - output `pred_actions`: `[B, H, A]`, with the final gripper dimension
      thresholded to environment commands `{-1, +1}`.
    """

    def __init__(self, input_dim: int, action_horizon: int, action_dim: int) -> None:
        super().__init__()
        if action_dim < 2:
            raise ValueError("split gripper head requires action_dim >= 2")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.input_dim = input_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.continuous_dim = action_dim - 1
        self.continuous_projection = nn.Linear(input_dim, action_horizon * self.continuous_dim)
        self.gripper_projection = nn.Linear(input_dim, action_horizon)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return continuous predictions, gripper logits, and env actions."""

        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [B, D], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"features dim {features.shape[1]} does not match {self.input_dim}"
            )
        batch_size = features.shape[0]
        continuous = self.continuous_projection(features).reshape(
            batch_size,
            self.action_horizon,
            self.continuous_dim,
        )
        gripper_logits = self.gripper_projection(features).reshape(
            batch_size,
            self.action_horizon,
        )
        gripper_command = gripper_logits_to_command(gripper_logits).unsqueeze(-1)
        return {
            "pred_continuous_actions": continuous,
            "pred_gripper_logits": gripper_logits,
            "pred_actions": torch.cat([continuous, gripper_command], dim=-1),
        }


__all__ = [
    "ActionChunkHead",
    "FutureLatentChunkHead",
    "SplitActionGripperHead",
    "gripper_logits_to_command",
]
