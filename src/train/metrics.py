"""Offline metric definitions for action and future-latent prediction.

Supports both CLS latents ``[B, H, D]`` and patch latents ``[B, H, N, D]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_action_inputs(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> None:
    """Validate that pred and target are compatible [B, H, A] float tensors."""
    if pred.shape != target.shape:
        raise ValueError(
            f"{name}: pred and target must have the same shape, "
            f"got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.ndim != 3:
        raise ValueError(
            f"{name}: pred and target must have shape [B, H, A], "
            f"got {tuple(pred.shape)}"
        )
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError(f"{name} expects floating point tensors")


def _validate_patch_inputs(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
    reduction: str,
) -> None:
    """Validate pred/target shape, dtype, and reduction for latent/patch metrics."""
    if pred.shape != target.shape:
        raise ValueError(
            f"{name}: pred and target must have the same shape, "
            f"got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.ndim != 3:
        raise ValueError(
            f"{name}: pred and target must have shape [B, H, D], "
            f"got {tuple(pred.shape)}"
        )
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError(f"{name} expects floating point tensors")
    if reduction not in {"mean", "per_horizon", "none"}:
        raise ValueError("reduction must be one of ['mean', 'per_horizon', 'none']")


def _apply_masked_reduction(
    error_2d: torch.Tensor,
    mask: torch.Tensor | None,
    shape: torch.Size,
    reduction: str,
) -> torch.Tensor:
    """Apply mask validation and reduction dispatch to a [B, H] error tensor."""
    if mask is None:
        if reduction == "none":
            return error_2d
        if reduction == "per_horizon":
            return error_2d.mean(dim=0)
        return error_2d.mean()

    mask_2d = _validate_horizon_mask(mask, shape).to(
        device=error_2d.device, dtype=error_2d.dtype
    )
    weighted = error_2d * mask_2d
    if reduction == "none":
        return weighted
    if reduction == "per_horizon":
        denom = mask_2d.sum(dim=0)
        if torch.any(denom <= 0):
            raise ValueError("each horizon step must have at least one valid mask entry")
        return weighted.sum(dim=0) / denom
    denom = mask_2d.sum()
    if denom.item() <= 0:
        raise ValueError("mask must contain at least one valid horizon step")
    return weighted.sum() / denom


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
    _validate_action_inputs(pred_actions, target_actions, name="action_mse")

    squared_error = (pred_actions - target_actions).pow(2)
    if mask is None:
        return squared_error.mean()

    # Use _validate_horizon_mask for consistent mask handling, then unsqueeze
    # for 3D broadcasting with action dimensions.
    mask_2d = _validate_horizon_mask(mask, pred_actions.shape[:2]).to(
        device=squared_error.device, dtype=squared_error.dtype
    )
    mask_3d = mask_2d.unsqueeze(-1)
    weighted_error = squared_error * mask_3d
    denominator = mask_3d.sum() * pred_actions.shape[-1]
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

    _validate_patch_inputs(pred_latents, target_latents, name="future_latent_cosine_error", reduction=reduction)

    error = 1.0 - F.cosine_similarity(
        pred_latents,
        target_latents,
        dim=-1,
        eps=eps,
    )
    return _apply_masked_reduction(error, mask, pred_latents.shape[:2], reduction)


def _validate_horizon_mask(mask: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(shape):
        raise ValueError(
            "mask must have shape [B, H] or [B, H, 1], "
            f"got {tuple(mask.shape)} for batch/horizon {tuple(shape)}"
        )
    return mask


def action_mse_per_horizon(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-horizon action MSE with shape `[H]`.

    Metric contract:

    - `pred_actions`: `[B, H, A]` floating point tensor.
    - `target_actions`: same shape.
    - `mask`: optional `[B, H]` or `[B, H, 1]` tensor.
    - Returns: `[H]` tensor with MSE for each horizon step.
    - Lower is better.
    """

    _validate_action_inputs(pred_actions, target_actions, name="action_mse_per_horizon")

    squared_error = (pred_actions - target_actions).pow(2).mean(dim=-1)  # [B, H]

    if mask is None:
        return squared_error.mean(dim=0)  # [H]

    mask_2d = _validate_horizon_mask(mask, pred_actions.shape[:2]).to(
        device=squared_error.device,
        dtype=squared_error.dtype,
    )
    weighted = squared_error * mask_2d
    denominator = mask_2d.sum(dim=0)
    if torch.any(denominator <= 0):
        raise ValueError("each horizon step must have at least one valid mask entry")
    return weighted.sum(dim=0) / denominator


def action_mse_per_dimension(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-dimension action MSE with shape `[A]`.

    Metric contract:

    - `pred_actions`: `[B, H, A]` floating point tensor.
    - `target_actions`: same shape.
    - `mask`: optional `[B, H]` or `[B, H, 1]` tensor.
    - Returns: `[A]` tensor with MSE for each action dimension.
    - Lower is better.
    """

    _validate_action_inputs(pred_actions, target_actions, name="action_mse_per_dimension")

    squared_error = (pred_actions - target_actions).pow(2)  # [B, H, A]

    if mask is None:
        return squared_error.mean(dim=(0, 1))  # [A]

    mask_3d = mask
    if mask_3d.ndim == 2:
        mask_3d = mask_3d.unsqueeze(-1)
    mask_3d = mask_3d.to(device=squared_error.device, dtype=squared_error.dtype)
    weighted = squared_error * mask_3d
    denominator = mask_3d.sum()
    if denominator.item() <= 0:
        raise ValueError("mask must contain at least one valid horizon step")
    return weighted.sum(dim=(0, 1)) / (denominator / pred_actions.shape[-1])


def future_latent_mse(
    pred_latents: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return future latent MSE for prediction quality assessment.

    Metric contract:

    - `pred_latents`: `[B, H, D]` floating point tensor.
    - `target_latents`: same shape.
    - `mask`: optional `[B, H]` or `[B, H, 1]` tensor.
    - Reduction: `"mean"` returns scalar, `"per_horizon"` returns `[H]`, `"none"` returns `[B, H]`.
    - Lower is better.
    """

    _validate_patch_inputs(pred_latents, target_latents, name="future_latent_mse", reduction=reduction)

    mse = (pred_latents - target_latents).pow(2).mean(dim=-1)  # [B, H]

    return _apply_masked_reduction(mse, mask, pred_latents.shape[:2], reduction)


# ---------------------------------------------------------------------------
# Patch latent metrics
# ---------------------------------------------------------------------------


def _validate_patch_shapes(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return patch latents as ``[B, H, N, D]``.

    Accepts either ``[B, H, N, D]`` or ``[B, N, D]`` (no horizon).  In all
    cases the returned tensors are ``[B, H, N, D]``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"{name}: pred and target must have the same shape, "
            f"got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError(f"{name} expects floating point tensors")

    if pred.ndim == 4:
        # Already [B, H, N, D]
        return pred, target
    if pred.ndim == 3:
        # Could be [B, N, D] (no horizon) – treat as [B, 1, N, D]
        return pred.unsqueeze(1), target.unsqueeze(1)
    raise ValueError(
        f"{name}: expected 3 or 4 dims, got {tuple(pred.shape)}"
    )


def patch_mse(
    pred_patch: torch.Tensor,
    target_patch: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean squared error over all patch elements.

    Metric contract:

    - ``pred_patch``: ``[B, H, N, D]`` or ``[B, N, D]`` floating point.
    - ``target_patch``: same shape.
    - ``mask``: optional ``[B, H]`` or ``[B, H, 1]`` (ignored when
      input is ``[B, N, D]``).
    - Reduction: ``"mean"`` returns scalar, ``"per_horizon"`` returns ``[H]``,
      ``"none"`` returns ``[B, H]`` (averaged over N and D).
    - Lower is better.
    """
    pred, target = _validate_patch_shapes(pred_patch, target_patch, name="patch_mse")
    B, H, N, D = pred.shape
    mse_per_element = (pred - target).pow(2)  # [B, H, N, D]
    mse_2d = mse_per_element.mean(dim=(-2, -1))  # [B, H]

    return _apply_masked_reduction(mse_2d, mask, (B, H), reduction)


def patch_cosine_error(
    pred_patch: torch.Tensor,
    target_patch: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-patch cosine error averaged over patches.

    Metric contract:

    - ``pred_patch``: ``[B, H, N, D]`` or ``[B, N, D]``.
    - ``target_patch``: same shape.
    - Cosine similarity is computed over ``D`` for each ``(b, h, n)`` entry.
    - Returns ``1 - cosine_similarity`` averaged over patches and (optionally)
      horizon/batch.
    - Lower is better (0 = perfect, 1 = orthogonal, 2 = opposite).
    """
    pred, target = _validate_patch_shapes(
        pred_patch, target_patch, name="patch_cosine_error"
    )
    B, H, N, D = pred.shape
    cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=eps)
    error = 1.0 - cos_sim  # [B, H, N]
    error_2d = error.mean(dim=-1)  # [B, H]

    return _apply_masked_reduction(error_2d, mask, (B, H), reduction)


def patch_mean_cosine_error(
    pred_patch: torch.Tensor,
    target_patch: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine error of the mean patch vector.

    Instead of computing per-patch cosine error, this first averages over
    patches ``N`` to get a single ``[B, H, D]`` vector per sample, then
    computes ``1 - cosine_similarity`` on those mean vectors.

    This metric captures global spatial direction similarity.

    Metric contract:

    - ``pred_patch``: ``[B, H, N, D]`` or ``[B, N, D]``.
    - ``target_patch``: same shape.
    - Returns scalar (``"mean"``), ``[H]`` (``"per_horizon"``), or
      ``[B, H]`` (``"none"``).
    - Lower is better.
    """
    pred, target = _validate_patch_shapes(
        pred_patch, target_patch, name="patch_mean_cosine_error"
    )
    B, H, N, D = pred.shape
    pred_mean = pred.mean(dim=2)  # [B, H, D]
    target_mean = target.mean(dim=2)  # [B, H, D]
    cos_sim = F.cosine_similarity(pred_mean, target_mean, dim=-1, eps=eps)
    error = 1.0 - cos_sim  # [B, H]

    return _apply_masked_reduction(error, mask, (B, H), reduction)


__all__ = [
    "action_mse",
    "action_mse_per_horizon",
    "action_mse_per_dimension",
    "future_latent_cosine_error",
    "future_latent_mse",
    "patch_cosine_error",
    "patch_mean_cosine_error",
    "patch_mse",
]
