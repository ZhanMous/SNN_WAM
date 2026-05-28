"""Offline metric definitions for action and future-latent prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def future_latent_cosine_error(
    pred_latents: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return `1 - cosine_similarity` for future latent prediction.

    Metric contract:

    - `pred_latents`: `[B, H, D]`, floating point tensor.
    - `target_latents`: same shape, dtype/device compatible with predictions.
    - `mask`: optional `[B, H]` or `[B, H, 1]` tensor where nonzero entries mark
      valid horizon steps. A valid horizon masks the whole latent vector.
    - Reduction: cosine similarity is computed over latent dimension `D` only.
      `reduction="none"` returns `[B, H]`; `"per_horizon"` returns `[H]`;
      `"mean"` returns a scalar mean over valid batch and horizon entries.
    - Lower is better: perfect prediction is `0`, orthogonal vectors are `1`,
      and opposite vectors are `2` under raw cosine error.
    - Scope: offline model-level metric, not closed-loop success.
    """

    if pred_latents.shape != target_latents.shape:
        raise ValueError(
            "pred_latents and target_latents must have the same shape, "
            f"got {tuple(pred_latents.shape)} and {tuple(target_latents.shape)}"
        )
    if pred_latents.ndim != 3:
        raise ValueError(
            "pred_latents and target_latents must have shape [B, H, D], "
            f"got {tuple(pred_latents.shape)}"
        )
    if not pred_latents.is_floating_point() or not target_latents.is_floating_point():
        raise TypeError("future_latent_cosine_error expects floating point tensors")
    if reduction not in {"mean", "per_horizon", "none"}:
        raise ValueError("reduction must be one of ['mean', 'per_horizon', 'none']")

    error = 1.0 - F.cosine_similarity(
        pred_latents,
        target_latents,
        dim=-1,
        eps=eps,
    )
    if mask is None:
        if reduction == "none":
            return error
        if reduction == "per_horizon":
            return error.mean(dim=0)
        return error.mean()

    mask_2d = _validate_horizon_mask(mask, pred_latents.shape[:2]).to(
        device=error.device,
        dtype=error.dtype,
    )
    weighted = error * mask_2d
    if reduction == "none":
        return weighted
    if reduction == "per_horizon":
        denominator = mask_2d.sum(dim=0)
        if torch.any(denominator <= 0):
            raise ValueError("each horizon step must have at least one valid mask entry")
        return weighted.sum(dim=0) / denominator

    denominator = mask_2d.sum()
    if denominator.item() <= 0:
        raise ValueError("mask must contain at least one valid horizon step")
    return weighted.sum() / denominator


def _validate_horizon_mask(mask: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(shape):
        raise ValueError(
            "mask must have shape [B, H] or [B, H, 1], "
            f"got {tuple(mask.shape)} for batch/horizon {tuple(shape)}"
        )
    return mask


__all__ = ["action_mse", "future_latent_cosine_error"]
