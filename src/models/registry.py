"""Model registry for action-only offline baselines."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.temporal_gru import (
    LatentProprioTaskGRUActionModel,
    TemporalGRUActionModel,
    TemporalGRUWAMModel,
)
from src.models.temporal_mlp import TemporalMLPActionModel


ACTION_MODEL_REGISTRY = {
    "mlp": TemporalMLPActionModel,
    "gru": TemporalGRUActionModel,
}

OFFLINE_MODEL_REGISTRY = {
    **ACTION_MODEL_REGISTRY,
    "bc_gru": LatentProprioTaskGRUActionModel,
    "wam_gru": TemporalGRUWAMModel,
}


def build_action_model(
    config: Mapping[str, Any],
    *,
    action_dim: int,
) -> nn.Module:
    """Build an action-only model with input `[B, T, A]` and output `[B, H, A]`."""

    adapter = str(config["model"]["temporal_adapter"])
    if adapter not in ACTION_MODEL_REGISTRY:
        raise ValueError(
            "train_offline.py currently supports action-only adapters "
            f"{sorted(ACTION_MODEL_REGISTRY)}, got {adapter!r}"
        )
    model_cls = ACTION_MODEL_REGISTRY[adapter]
    return model_cls(
        history_len=int(config["data"]["history_len"]),
        action_dim=action_dim,
        action_horizon=int(config["data"]["action_horizon"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    )


def build_offline_model(
    config: Mapping[str, Any],
    *,
    action_dim: int,
    latent_dim: int | None = None,
    state_dim: int | None = None,
    num_tasks: int | None = None,
) -> nn.Module:
    """Build an offline action or WAM model.

    Action-only models consume `action_history: [B, T, A]`. The WAM-GRU model
    additionally consumes `z_t: [B, Z]` and predicts future latents.
    """

    adapter = str(config["model"]["temporal_adapter"])
    if adapter in ACTION_MODEL_REGISTRY:
        return build_action_model(config, action_dim=action_dim)
    if adapter == "bc_gru":
        if latent_dim is None or latent_dim <= 0:
            raise ValueError("bc_gru requires a positive latent_dim")
        if state_dim is None or state_dim <= 0:
            raise ValueError("bc_gru requires a positive state_dim")
        task_count = int(num_tasks or config["model"].get("num_tasks", 1))
        return LatentProprioTaskGRUActionModel(
            history_len=int(config["data"]["history_len"]),
            action_dim=action_dim,
            action_horizon=int(config["data"]["action_horizon"]),
            latent_dim=latent_dim,
            state_dim=state_dim,
            num_tasks=task_count,
            hidden_dim=int(config["model"]["hidden_dim"]),
        )
    if adapter != "wam_gru":
        raise ValueError(
            "train_offline.py currently supports adapters "
            f"{sorted(OFFLINE_MODEL_REGISTRY)}, got {adapter!r}"
        )
    if latent_dim is None or latent_dim <= 0:
        raise ValueError("wam_gru requires a positive latent_dim")
    return TemporalGRUWAMModel(
        history_len=int(config["data"]["history_len"]),
        action_dim=action_dim,
        action_horizon=int(config["data"]["action_horizon"]),
        latent_dim=latent_dim,
        future_horizon=int(config["data"]["future_horizon"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        split_gripper_head=str(config["model"].get("action_head_type", "mse")) == "split_gripper",
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"parameter_count": total, "trainable_parameter_count": trainable}


__all__ = [
    "ACTION_MODEL_REGISTRY",
    "OFFLINE_MODEL_REGISTRY",
    "build_action_model",
    "build_offline_model",
    "count_parameters",
]
