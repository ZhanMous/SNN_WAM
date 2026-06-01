"""Tests for G8 mixed-action objective and metric repair."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

pytest.importorskip("torch")
import torch

from src.eval.g8_mixed_action_metrics import (
    CONTINUOUS_DIMS,
    GRIPPER_DIM_IDX,
    GRIPPER_ENCODING,
    GRIPPER_THRESHOLD,
    compute_split_metrics,
    build_action_contract,
    SplitLinearAR,
    SplitMLP,
    SplitGRU,
    SplitGRUPlusState,
    resolve_frame_reference,
    get_git_info,
)


# ---------------------------------------------------------------------------
# Action contract tests
# ---------------------------------------------------------------------------

def test_action_contract_identifies_dims():
    """Action contract correctly identifies continuous and gripper dims."""
    actions = np.random.randn(100, 7).astype(np.float32)
    actions[:, GRIPPER_DIM_IDX] = np.sign(actions[:, GRIPPER_DIM_IDX])
    contract = build_action_contract(
        actions_np=actions, trajectory_id="test",
        dataset="test", task_name="test", git_info=get_git_info(),
    )
    assert contract["continuous_dims"] == CONTINUOUS_DIMS
    assert contract["gripper_dim"] == GRIPPER_DIM_IDX
    assert contract["action_dim"] == 7
    assert contract["gripper_encoding"] == "binary_sign"


def test_action_contract_gripper_stats():
    actions = np.zeros((100, 7), dtype=np.float32)
    actions[:60, GRIPPER_DIM_IDX] = 1.0   # open
    actions[60:, GRIPPER_DIM_IDX] = -1.0  # close
    contract = build_action_contract(
        actions_np=actions, trajectory_id="test",
        dataset="test", task_name="test", git_info=get_git_info(),
    )
    assert contract["gripper_stats"]["n_open"] == 60
    assert contract["gripper_stats"]["n_close"] == 40
    assert contract["gripper_stats"]["fraction_open"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Split metrics tests
# ---------------------------------------------------------------------------

def test_continuous_metrics_exclude_gripper():
    """Continuous MSE excludes gripper dim."""
    pred = torch.randn(4, 1, 7)
    target = torch.randn(4, 1, 7)
    metrics = compute_split_metrics(pred, target)

    # Continuous MSE should only use dims 0-5
    cont_se = (pred[..., CONTINUOUS_DIMS] - target[..., CONTINUOUS_DIMS]).pow(2)
    expected_cont_mse = float(cont_se.mean().item())
    assert metrics["continuous_raw_mse"] == pytest.approx(expected_cont_mse, abs=1e-6)

    # Global MSE includes gripper
    global_se = (pred - target).pow(2)
    expected_global = float(global_se.mean().item())
    assert metrics["global_raw_mse"] == pytest.approx(expected_global, abs=1e-6)

    # Global should differ from continuous (gripper contributes)
    assert metrics["global_raw_mse"] != metrics["continuous_raw_mse"]


def test_gripper_metrics_do_not_affect_continuous_mse():
    """Changing gripper prediction does not change continuous MSE."""
    pred1 = torch.randn(4, 1, 7)
    pred2 = pred1.clone()
    pred2[..., GRIPPER_DIM_IDX] = torch.randn(4, 1)  # different gripper
    target = torch.randn(4, 1, 7)

    m1 = compute_split_metrics(pred1, target)
    m2 = compute_split_metrics(pred2, target)
    assert m1["continuous_raw_mse"] == pytest.approx(m2["continuous_raw_mse"])
    assert m1["continuous_normalized_mse"] == pytest.approx(m2["continuous_normalized_mse"])


def test_global_raw_mse_marked_diagnostic():
    """Global raw MSE is diagnostic only, not primary."""
    pred = torch.randn(4, 1, 7)
    target = torch.randn(4, 1, 7)
    metrics = compute_split_metrics(pred, target)
    # The metric exists but is not primary
    assert "global_raw_mse" in metrics
    # old_1e4_gate is engineering gate only
    assert "old_1e4_gate" in metrics


def test_old_1e4_gate_is_engineering_only():
    """Old 1e-4 gate is engineering_overfit_gate_only."""
    pred = torch.zeros(4, 1, 7)
    target = torch.zeros(4, 1, 7)
    metrics = compute_split_metrics(pred, target)
    # Zero error passes 1e-4, but this is engineering gate only
    assert metrics["old_1e4_gate"] is True


def test_baseline_comparison_metrics():
    """Baseline comparison metrics are computed when baselines provided."""
    pred = torch.randn(4, 1, 7)
    target = torch.randn(4, 1, 7)
    last_action = torch.randn(4, 1, 7)
    metrics = compute_split_metrics(pred, target, last_action_pred=last_action)
    assert "beat_last_action_continuous" in metrics
    assert isinstance(metrics["beat_last_action_continuous"], bool)


def test_normalized_mse_uses_action_stats():
    """Normalized MSE divides by std_safe."""
    pred = torch.ones(4, 1, 7)
    target = torch.zeros(4, 1, 7)
    stats = {"std_safe": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
    metrics_norm = compute_split_metrics(pred, target, action_stats=stats)
    metrics_raw = compute_split_metrics(pred, target)
    # With std=1, normalized should equal raw for continuous
    assert metrics_norm["continuous_normalized_mse"] == pytest.approx(
        metrics_raw["continuous_raw_mse"], abs=1e-5)


# ---------------------------------------------------------------------------
# Model shape tests
# ---------------------------------------------------------------------------

def test_split_linear_ar_output_shape():
    model = SplitLinearAR(history_len=4, action_dim=7)
    history = torch.randn(4, 4, 7)
    out = model(history)
    assert out["pred_actions"].shape == (4, 1, 7)
    assert out["pred_continuous_actions"].shape == (4, 1, 6)
    assert out["pred_gripper_logits"].shape == (4, 1)


def test_split_mlp_output_shape():
    model = SplitMLP(input_dim=92, hidden_dim=64, action_dim=7)
    x = torch.randn(4, 92)
    out = model(x)
    assert out["pred_actions"].shape == (4, 1, 7)
    assert out["pred_continuous_actions"].shape == (4, 1, 6)
    assert out["pred_gripper_logits"].shape == (4, 1)


def test_split_gru_output_shape():
    model = SplitGRU(input_dim=7, hidden_dim=64, action_dim=7)
    x = torch.randn(4, 4, 7)
    out = model(x)
    assert out["pred_actions"].shape == (4, 1, 7)


def test_split_gru_plus_state_output_shape():
    model = SplitGRUPlusState(state_dim=92, action_dim=7, history_len=4, hidden_dim=64)
    history = torch.randn(4, 4, 7)
    state = torch.randn(4, 92)
    out = model(history, state)
    assert out["pred_actions"].shape == (4, 1, 7)


# ---------------------------------------------------------------------------
# Target shift leakage test
# ---------------------------------------------------------------------------

def test_target_shift_minus1_remains_leakage():
    """target_shift=-1 must be labeled as leakage_diagnostic_only."""
    from src.eval.overfit_diagnostics import causal_next_action_v1_check
    sample = {
        "time_index": 5,
        "action_history_indices": [2, 3, 4, 5],
        "target_action_indices": [5],  # target_shift=-1 targets actions[t]
        "input_keys": ("image_t", "action_history"),
        "target_keys": ("target_actions",),
        "target_shift": -1,
    }
    result = causal_next_action_v1_check(sample)
    assert result["invariants"]["target_shift_is_zero"]["pass"] is False
    assert result["pass"] is False


# ---------------------------------------------------------------------------
# Frame reference resolution test
# ---------------------------------------------------------------------------

def test_resolve_frame_reference_returns_none_for_missing():
    result = resolve_frame_reference("nonexistent/path.hdf5:data/demo_0:obs/agentview_rgb:0", "/tmp")
    assert result is None


# ---------------------------------------------------------------------------
# Artifact CSV format test
# ---------------------------------------------------------------------------

def test_ladder_row_has_required_fields():
    metrics = {
        "continuous_normalized_mse": 0.01,
        "continuous_raw_mse": 0.02,
        "continuous_raw_mae": 0.1,
        "gripper_sign_accuracy": 0.95,
        "gripper_transition_f1": 0.8,
        "global_raw_mse": 0.05,
        "old_1e4_gate": False,
        "beat_last_action_continuous": True,
        "beat_last_action_gripper_f1": True,
    }
    from src.eval.g8_mixed_action_metrics import _make_ladder_row
    row = _make_ladder_row("test", "mlp", 100, metrics, 50)
    required = [
        "variant", "model_type", "param_count",
        "continuous_normalized_mse", "continuous_raw_mse", "continuous_raw_mae",
        "gripper_sign_accuracy", "gripper_transition_f1",
        "global_raw_mse", "old_1e4_gate",
        "beat_last_action_continuous", "beat_last_action_gripper_f1",
        "best_epoch",
    ]
    for field in required:
        assert field in row, f"Missing required field: {field}"
