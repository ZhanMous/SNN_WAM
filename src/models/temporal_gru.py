"""GRU temporal baseline for offline action prediction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.heads import ActionChunkHead, FutureLatentChunkHead


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
        self.action_head = ActionChunkHead(hidden_dim, action_horizon, action_dim)
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
        return {
            "pred_actions": self.action_head(fused),
            "pred_future_latents": self.future_latent_head(fused),
        }

    def _validate_z_t(self, z_t: torch.Tensor) -> None:
        if z_t.ndim != 2:
            raise ValueError(f"z_t must have shape [B, Z], got {tuple(z_t.shape)}")
        if z_t.shape[1] != self.latent_dim:
            raise ValueError(f"latent_dim {z_t.shape[1]} does not match {self.latent_dim}")


__all__ = ["TemporalGRU", "TemporalGRUActionModel", "TemporalGRUWAMModel"]
