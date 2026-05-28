"""Model registry for action-only offline baselines."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.temporal_gru import TemporalGRUActionModel
from src.models.temporal_mlp import TemporalMLPActionModel


ACTION_MODEL_REGISTRY = {
    "mlp": TemporalMLPActionModel,
    "gru": TemporalGRUActionModel,
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


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"parameter_count": total, "trainable_parameter_count": trainable}


__all__ = ["ACTION_MODEL_REGISTRY", "build_action_model", "count_parameters"]
