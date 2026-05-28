from __future__ import annotations

import pytest
import torch

from src.models.registry import build_action_model, count_parameters
from src.models.temporal_gru import TemporalGRUActionModel, TemporalGRUWAMModel
from src.train.metrics import action_mse


def test_temporal_gru_action_model_forward_shape() -> None:
    model = TemporalGRUActionModel(
        history_len=4,
        action_dim=7,
        action_horizon=3,
        hidden_dim=16,
    )
    action_history = torch.zeros(2, 4, 7)

    pred_actions = model(action_history)

    assert tuple(pred_actions.shape) == (2, 3, 7)


def test_temporal_gru_rejects_wrong_history_shape() -> None:
    model = TemporalGRUActionModel(
        history_len=4,
        action_dim=7,
        action_horizon=3,
        hidden_dim=16,
    )

    with pytest.raises(ValueError, match=r"\[B, T, A\]"):
        model(torch.zeros(2, 7))

    with pytest.raises(ValueError, match="history_len"):
        model(torch.zeros(2, 5, 7))


def test_temporal_gru_wam_model_forward_shapes() -> None:
    model = TemporalGRUWAMModel(
        history_len=4,
        action_dim=7,
        action_horizon=3,
        latent_dim=8,
        future_horizon=2,
        hidden_dim=16,
    )

    outputs = model(torch.zeros(2, 4, 7), torch.zeros(2, 8))

    assert tuple(outputs["pred_actions"].shape) == (2, 3, 7)
    assert tuple(outputs["pred_future_latents"].shape) == (2, 2, 8)


def test_temporal_gru_wam_rejects_wrong_latent_shape() -> None:
    model = TemporalGRUWAMModel(
        history_len=4,
        action_dim=7,
        action_horizon=3,
        latent_dim=8,
        future_horizon=2,
        hidden_dim=16,
    )

    with pytest.raises(ValueError, match=r"\[B, Z\]"):
        model(torch.zeros(2, 4, 7), torch.zeros(2, 2, 8))


def test_registry_builds_mlp_and_gru_with_same_action_contract() -> None:
    base_config = {
        "data": {"history_len": 4, "action_horizon": 3},
        "model": {"temporal_adapter": "mlp", "hidden_dim": 16},
    }
    action_history = torch.zeros(2, 4, 7)

    mlp = build_action_model(base_config, action_dim=7)
    gru = build_action_model(
        {
            **base_config,
            "model": {"temporal_adapter": "gru", "hidden_dim": 16},
        },
        action_dim=7,
    )

    assert tuple(mlp(action_history).shape) == (2, 3, 7)
    assert tuple(gru(action_history).shape) == (2, 3, 7)
    assert count_parameters(mlp)["parameter_count"] != count_parameters(gru)[
        "parameter_count"
    ]


def test_tiny_batch_gru_training_reduces_action_mse() -> None:
    torch.manual_seed(0)
    model = TemporalGRUActionModel(
        history_len=3,
        action_dim=2,
        action_horizon=2,
        hidden_dim=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    action_history = torch.tensor(
        [
            [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]],
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
        ]
    )
    target_actions = torch.tensor(
        [
            [[3.0, 3.1], [4.0, 4.1]],
            [[4.0, 4.1], [5.0, 5.1]],
        ]
    )

    with torch.no_grad():
        initial_loss = action_mse(model(action_history), target_actions).item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = action_mse(model(action_history), target_actions)
        loss.backward()
        optimizer.step()
    final_loss = action_mse(model(action_history), target_actions).item()

    assert final_loss < initial_loss
