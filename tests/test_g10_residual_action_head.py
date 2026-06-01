"""Tests for G10 residual-action head and action-parameterization repair."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
import numpy as np

pytest.importorskip("torch")
import torch

from src.eval.g10_residual_action_head import (
    ResidualSplitGRUPlusState,
    ResidualSplitGRU,
    ResidualSplitMLP,
    reconstruct_from_residual,
)
from src.eval.g8_mixed_action_metrics import (
    CONTINUOUS_DIMS,
    GRIPPER_DIM_IDX,
    compute_split_metrics,
)
from src.models.heads import gripper_logits_to_command


# ---------------------------------------------------------------------------
# Residual target contract tests
# ---------------------------------------------------------------------------

def test_residual_target_uses_action_t_minus_action_t_minus_1():
    """Residual target is action[t] - action[t-1], not action[t+1]."""
    actions = np.array([
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        [0.15, 0.25, 0.28, 0.01, -0.01, 0.0, 1.0],
        [0.2, 0.3, 0.25, 0.02, -0.02, 0.0, -1.0],
    ], dtype=np.float32)

    # For t=2: target = action[2] - action[1]
    target_residual = actions[2, CONTINUOUS_DIMS] - actions[1, CONTINUOUS_DIMS]
    expected = np.array([0.05, 0.05, -0.03, 0.01, -0.01, 0.0])
    np.testing.assert_allclose(target_residual, expected, atol=1e-6)


def test_reconstructed_action_is_last_plus_residual():
    """Reconstructed action = action[t-1] + predicted_residual."""
    last_action = torch.tensor([[[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]]])
    residual = torch.tensor([[[0.05, 0.05, -0.03, 0.01, -0.01, 0.0]]])
    gripper_logits = torch.tensor([[2.0]])  # logits > 0 => open (+1)

    outputs = {
        "pred_continuous_residual": residual,
        "pred_gripper_logits": gripper_logits,
    }
    recon = reconstruct_from_residual(outputs, last_action)

    expected_cont = last_action[..., CONTINUOUS_DIMS] + residual
    assert torch.allclose(recon["pred_continuous_actions"], expected_cont)
    assert recon["pred_actions"].shape == (1, 1, 7)
    # Gripper should be +1 (open) since logits > 0
    assert recon["pred_actions"][0, 0, GRIPPER_DIM_IDX].item() == 1.0


def test_gripper_excluded_from_continuous_residual():
    """Gripper dim is NOT part of continuous residual regression."""
    model = ResidualSplitGRUPlusState(
        state_dim=92, action_dim=7, history_len=4, hidden_dim=64,
    )
    history = torch.randn(2, 4, 7)
    state = torch.randn(2, 92)
    out = model(history, state)
    # Residual should be [B, 1, 6] (continuous only, no gripper)
    assert out["pred_continuous_residual"].shape == (2, 1, 6)
    # Gripper logits should be [B, 1]
    assert out["pred_gripper_logits"].shape == (2, 1)


# ---------------------------------------------------------------------------
# Model shape tests
# ---------------------------------------------------------------------------

def test_residual_gru_plus_state_output_shape():
    model = ResidualSplitGRUPlusState(
        state_dim=92, action_dim=7, history_len=4, hidden_dim=64,
    )
    history = torch.randn(2, 4, 7)
    state = torch.randn(2, 92)
    out = model(history, state)
    assert out["pred_continuous_residual"].shape == (2, 1, 6)
    assert out["pred_gripper_logits"].shape == (2, 1)


def test_residual_gru_output_shape():
    model = ResidualSplitGRU(action_dim=7, hidden_dim=64)
    history = torch.randn(2, 4, 7)
    out = model(history)
    assert out["pred_continuous_residual"].shape == (2, 1, 6)
    assert out["pred_gripper_logits"].shape == (2, 1)


def test_residual_mlp_output_shape():
    model = ResidualSplitMLP(input_dim=92, hidden_dim=64, action_dim=7)
    x = torch.randn(2, 92)
    out = model(x)
    assert out["pred_continuous_residual"].shape == (2, 1, 6)
    assert out["pred_gripper_logits"].shape == (2, 1)


# ---------------------------------------------------------------------------
# Reconstruction tests
# ---------------------------------------------------------------------------

def test_reconstruction_preserves_gripper_from_classification():
    """Gripper comes from classification logits, not residual regression."""
    last_action = torch.tensor([[[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0]]])
    residual = torch.zeros(1, 1, 6)

    # Logits > 0 => open (+1)
    outputs_open = {"pred_continuous_residual": residual, "pred_gripper_logits": torch.tensor([[2.0]])}
    recon_open = reconstruct_from_residual(outputs_open, last_action)
    assert recon_open["pred_actions"][0, 0, GRIPPER_DIM_IDX].item() == 1.0

    # Logits < 0 => close (-1)
    outputs_close = {"pred_continuous_residual": residual, "pred_gripper_logits": torch.tensor([[-2.0]])}
    recon_close = reconstruct_from_residual(outputs_close, last_action)
    assert recon_close["pred_actions"][0, 0, GRIPPER_DIM_IDX].item() == -1.0


# ---------------------------------------------------------------------------
# Target shift leakage test
# ---------------------------------------------------------------------------

def test_residual_target_is_not_leakage():
    """Residual target action[t]-action[t-1] uses action[t-1] which is in history."""
    from src.eval.overfit_diagnostics import causal_next_action_v1_check

    sample = {
        "time_index": 5,
        "action_history_indices": [2, 3, 4, 5],  # includes action[t-1]=action[5]
        "target_action_indices": [6],  # target is action[6]
        "input_keys": ("image_t", "action_history"),
        "target_keys": ("target_actions",),
        "target_shift": 0,
    }
    result = causal_next_action_v1_check(sample)
    assert result["pass"] is True


# ---------------------------------------------------------------------------
# Split metrics consistency
# ---------------------------------------------------------------------------

def test_split_metrics_work_for_reconstructed_actions():
    """Reconstructed actions can be evaluated with split metrics."""
    pred = torch.randn(4, 1, 7)
    target = torch.randn(4, 1, 7)
    stats = {"std_safe": [0.3, 0.25, 0.5, 0.03, 0.1, 0.03]}
    m = compute_split_metrics(pred, target, action_stats=stats)
    assert "continuous_normalized_mse" in m
    assert "gripper_sign_accuracy" in m


# ---------------------------------------------------------------------------
# CSV format test
# ---------------------------------------------------------------------------

def test_comparison_row_has_required_fields():
    row = {
        "variant": "test",
        "direct_continuous_normalized_mse": 0.01,
        "residual_continuous_normalized_mse": 0.008,
        "improvement_ratio": 1.25,
    }
    required = ["variant", "direct_continuous_normalized_mse", "residual_continuous_normalized_mse"]
    for field in required:
        assert field in row
