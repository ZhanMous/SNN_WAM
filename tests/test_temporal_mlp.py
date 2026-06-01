from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from src.models.temporal_mlp import TemporalMLPActionModel
from src.train.metrics import action_mse


def test_temporal_mlp_action_model_forward_shape() -> None:
    model = TemporalMLPActionModel(
        history_len=4,
        action_dim=7,
        action_horizon=3,
        hidden_dim=16,
    )
    action_history = torch.zeros(2, 4, 7)

    pred_actions = model(action_history)

    assert tuple(pred_actions.shape) == (2, 3, 7)


def test_temporal_mlp_rejects_wrong_history_shape() -> None:
    model = TemporalMLPActionModel(
        history_len=4,
        action_dim=7,
        action_horizon=3,
        hidden_dim=16,
    )

    with pytest.raises(ValueError, match=r"\[B, T, A\]"):
        model(torch.zeros(2, 7))

    with pytest.raises(ValueError, match="history_len"):
        model(torch.zeros(2, 5, 7))


def test_action_mse_perfect_prediction_is_zero() -> None:
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    assert action_mse(target, target).item() == pytest.approx(0.0)


def test_action_mse_masked_padded_horizon_does_not_affect_result() -> None:
    pred = torch.tensor([[[1.0, 2.0], [1000.0, 1000.0]]])
    target = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])

    assert action_mse(pred, target, mask=mask).item() == pytest.approx(0.0)


def test_tiny_batch_overfit_reduces_action_mse() -> None:
    torch.manual_seed(0)
    model = TemporalMLPActionModel(
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
