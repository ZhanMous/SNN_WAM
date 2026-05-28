"""Offline metric definitions for action prediction."""

from __future__ import annotations

import torch


def action_mse(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return global action mean squared error, lower is better.

    Metric contract:

    - `pred_actions`: `[B, H, A]` or `[N, H, A]`, floating point tensor.
    - `target_actions`: same shape, dtype/device compatible with predictions.
    - `mask`: optional `[B, H]` or `[B, H, 1]` tensor where nonzero entries mark
      valid horizon steps. A valid horizon masks all `A` action dimensions.
    - Reduction: mean over batch/window, horizon, and action dimensions after
      optional masking.
    - Unit: same action units as the tensors provided by the caller. The
      current trainer logs raw LIBERO/mock action units because no action
      normalization is applied yet.
    - Scope: offline model-level metric, not closed-loop success.
    """

    if pred_actions.shape != target_actions.shape:
        raise ValueError(
            "pred_actions and target_actions must have the same shape, "
            f"got {tuple(pred_actions.shape)} and {tuple(target_actions.shape)}"
        )
    if pred_actions.ndim != 3:
        raise ValueError(
            "pred_actions and target_actions must have shape [B, H, A], "
            f"got {tuple(pred_actions.shape)}"
        )
    if not pred_actions.is_floating_point() or not target_actions.is_floating_point():
        raise TypeError("action_mse expects floating point tensors")

    squared_error = (pred_actions - target_actions).pow(2)
    if mask is None:
        return squared_error.mean()

    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    if mask.ndim != 3 or mask.shape[:2] != pred_actions.shape[:2] or mask.shape[2] != 1:
        raise ValueError(
            "mask must have shape [B, H] or [B, H, 1], "
            f"got {tuple(mask.shape)} for predictions {tuple(pred_actions.shape)}"
        )

    mask = mask.to(device=squared_error.device, dtype=squared_error.dtype)
    weighted_error = squared_error * mask
    denominator = mask.sum() * pred_actions.shape[-1]
    if denominator.item() <= 0:
        raise ValueError("mask must contain at least one valid horizon step")
    return weighted_error.sum() / denominator


__all__ = ["action_mse"]
