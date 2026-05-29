"""GRU temporal baseline for offline action prediction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.heads import ActionChunkHead, FutureLatentChunkHead, SplitActionGripperHead


class TemporalGRU(nn.Module):
    """Encode action history with a GRU temporal adapter.

    This is an action-only baseline. It does not implement SNN dynamics,
    future-latent prediction, image encoding, or text encoding.

    Shape contract:

    - input `action_history`: `[B, T, A]`.
    - output features: `[B, hidden_dim]`.
    """

    def __init__(
        self,
        *,
        history_len: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if history_len <= 0:
            raise ValueError("history_len must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.history_len = history_len
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """Return temporal features for `action_history: [B, T, A]`."""

        self._validate_action_history(action_history)
        _, hidden = self.gru(action_history)
        return hidden[-1]

    def _validate_action_history(self, action_history: torch.Tensor) -> None:
        if action_history.ndim != 3:
            raise ValueError(
                "action_history must have shape [B, T, A], "
                f"got {tuple(action_history.shape)}"
            )
        _, history_len, action_dim = action_history.shape
        if history_len != self.history_len:
            raise ValueError(
                f"history_len {history_len} does not match {self.history_len}"
            )
        if action_dim != self.action_dim:
            raise ValueError(f"action_dim {action_dim} does not match {self.action_dim}")


class TemporalGRUActionModel(nn.Module):
    """Action-only GRU baseline.

    Shape contract:

    - input `action_history`: `[B, T, A]`.
    - output `pred_actions`: `[B, H, A]`.
    """

    def __init__(
        self,
        *,
        history_len: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.temporal = TemporalGRU(
            history_len=history_len,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.action_head = ActionChunkHead(hidden_dim, action_horizon, action_dim)

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """Return predicted future action chunk with shape `[B, H, A]`."""

        features = self.temporal(action_history)
        return self.action_head(features)


class TemporalGRUWAMModel(nn.Module):
    """GRU world-action adapter baseline with frozen visual latent input.

    Shape contract:

    - input `action_history`: `[B, T, A]`.
    - input `z_t`: `[B, Z]`, current frozen visual latent only.
    - output `pred_actions`: `[B, action_horizon, A]`.
    - output `pred_future_latents`: `[B, future_horizon, Z]`.

    Future latents are predicted targets; this model does not consume
    `target_future_latents` or future observations as inputs.
    """

    def __init__(
        self,
        *,
        history_len: int,
        action_dim: int,
        action_horizon: int,
        latent_dim: int,
        future_horizon: int,
        hidden_dim: int,
        num_layers: int = 1,
        split_gripper_head: bool = False,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if future_horizon <= 0:
            raise ValueError("future_horizon must be positive")
        self.latent_dim = latent_dim
        self.future_horizon = future_horizon
        self.temporal = TemporalGRU(
            history_len=history_len,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.split_gripper_head = split_gripper_head
        self.action_head = (
            SplitActionGripperHead(hidden_dim, action_horizon, action_dim)
            if split_gripper_head
            else ActionChunkHead(hidden_dim, action_horizon, action_dim)
        )
        self.future_latent_head = FutureLatentChunkHead(
            hidden_dim,
            future_horizon,
            latent_dim,
        )

    def forward(
        self,
        action_history: torch.Tensor,
        z_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return action and future-latent predictions."""

        self._validate_z_t(z_t)
        features = self.temporal(action_history)
        fused = self.fusion(torch.cat([features, z_t], dim=-1))
        action_outputs = self.action_head(fused)
        if isinstance(action_outputs, dict):
            return {
                **action_outputs,
                "pred_future_latents": self.future_latent_head(fused),
            }
        return {
            "pred_actions": action_outputs,
            "pred_future_latents": self.future_latent_head(fused),
        }

    def _validate_z_t(self, z_t: torch.Tensor) -> None:
        if z_t.ndim != 2:
            raise ValueError(f"z_t must have shape [B, Z], got {tuple(z_t.shape)}")
        if z_t.shape[1] != self.latent_dim:
            raise ValueError(f"latent_dim {z_t.shape[1]} does not match {self.latent_dim}")


class LatentProprioTaskGRUActionModel(nn.Module):
    """Minimal BC-GRU baseline with frozen visual latent, proprio, and task id.

    Shape contract:

    - input `action_history`: `[B, T, A]`.
    - input `z_t`: `[B, Z]`, current frozen visual latent only.
    - input `state_t`: `[B, S]`, current robot proprio/state.
    - input `task_id`: `[B]`, integer task ids.
    - output `pred_actions`: `[B, H, A]`.

    This baseline intentionally has no future-latent objective or future
    latent head. It is meant to test whether a simple behavior cloning head
    with the obvious causal inputs can fit actions before making architecture
    claims about future prediction.
    """

    uses_proprio_task = True

    def __init__(
        self,
        *,
        history_len: int,
        action_dim: int,
        action_horizon: int,
        latent_dim: int,
        state_dim: int,
        num_tasks: int,
        hidden_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.num_tasks = num_tasks
        self.temporal = TemporalGRU(
            history_len=history_len,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.latent_encoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = ActionChunkHead(hidden_dim, action_horizon, action_dim)

    def forward(
        self,
        action_history: torch.Tensor,
        z_t: torch.Tensor,
        state_t: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted future action chunk with shape `[B, H, A]`."""

        self._validate_inputs(z_t, state_t, task_id)
        temporal_features = self.temporal(action_history)
        fused = self.fusion(
            torch.cat(
                [
                    temporal_features,
                    self.latent_encoder(z_t),
                    self.state_encoder(state_t),
                    self.task_embedding(task_id),
                ],
                dim=-1,
            )
        )
        return self.action_head(fused)

    def _validate_inputs(
        self,
        z_t: torch.Tensor,
        state_t: torch.Tensor,
        task_id: torch.Tensor,
    ) -> None:
        if z_t.ndim != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(f"z_t must have shape [B, {self.latent_dim}]")
        if state_t.ndim != 2 or state_t.shape[1] != self.state_dim:
            raise ValueError(f"state_t must have shape [B, {self.state_dim}]")
        if task_id.ndim != 1:
            raise ValueError(f"task_id must have shape [B], got {tuple(task_id.shape)}")
        if task_id.numel() and (
            int(task_id.min().item()) < 0 or int(task_id.max().item()) >= self.num_tasks
        ):
            raise ValueError("task_id values must be in [0, num_tasks)")


__all__ = [
    "LatentProprioTaskGRUActionModel",
    "TemporalGRU",
    "TemporalGRUActionModel",
    "TemporalGRUWAMModel",
]
