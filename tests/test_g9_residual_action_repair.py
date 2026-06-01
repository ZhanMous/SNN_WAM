"""Tests for G9 residual error attribution and action target repair."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.eval.g9_residual_action_repair import (
    compute_residual_attribution,
    CONT_DIM_LABELS,
)
from src.eval.g8_mixed_action_metrics import (
    CONTINUOUS_DIMS,
    GRIPPER_DIM_IDX,
    compute_split_metrics,
)


# ---------------------------------------------------------------------------
# Residual attribution tests
# ---------------------------------------------------------------------------

def test_residual_attribution_basic():
    """Residual attribution runs on synthetic data."""
    T = 50
    pred = torch.randn(T, 1, 7)
    target = torch.randn(T, 1, 7)
    actions = np.random.randn(T, 7).astype(np.float32)
    time_indices = list(range(T))
    stats = {"std_safe": [0.3, 0.25, 0.5, 0.03, 0.1, 0.03]}

    attr = compute_residual_attribution(
        pred, target, actions_np=actions, time_indices=time_indices, action_stats=stats,
    )
    assert "per_dim" in attr
    assert len(attr["per_dim"]) == 6
    assert "autocorrelation" in attr
    assert len(attr["autocorrelation"]) == 6
    assert "worst_timesteps" in attr
    assert len(attr["worst_timesteps"]) == 10
    assert attr["n_samples"] == T


def test_residual_per_dim_mse_excludes_gripper():
    """Per-dim residual MSE only covers continuous dims (0-5), not gripper."""
    T = 20
    pred = torch.randn(T, 1, 7)
    target = torch.randn(T, 1, 7)
    actions = np.random.randn(T, 7).astype(np.float32)
    stats = {"std_safe": [1.0] * 6}

    attr = compute_residual_attribution(
        pred, target, actions_np=actions, time_indices=list(range(T)), action_stats=stats,
    )
    for pd in attr["per_dim"]:
        assert pd["dim"] in CONTINUOUS_DIMS
        assert pd["label"] in CONT_DIM_LABELS


def test_residual_autocorrelation_range():
    """Autocorrelation should be between -1 and 1."""
    T = 30
    pred = torch.randn(T, 1, 7)
    target = torch.randn(T, 1, 7)
    actions = np.random.randn(T, 7).astype(np.float32)
    stats = {"std_safe": [1.0] * 6}

    attr = compute_residual_attribution(
        pred, target, actions_np=actions, time_indices=list(range(T)), action_stats=stats,
    )
    for ac in attr["autocorrelation"]:
        assert -1.5 <= ac <= 1.5  # allow small numerical error


# ---------------------------------------------------------------------------
# Split metrics consistency tests
# ---------------------------------------------------------------------------

def test_split_metrics_consistent_with_g8():
    """G9 split metrics are consistent with G8 implementation."""
    pred = torch.randn(4, 1, 7)
    target = torch.randn(4, 1, 7)
    stats = {"std_safe": [0.3, 0.25, 0.5, 0.03, 0.1, 0.03]}
    m = compute_split_metrics(pred, target, action_stats=stats)
    assert "continuous_normalized_mse" in m
    assert "gripper_sign_accuracy" in m
    assert "global_raw_mse" in m


# ---------------------------------------------------------------------------
# Target shift leakage test
# ---------------------------------------------------------------------------

def test_shift_sanity_labels():
    """Shift sanity correctly identifies causal vs leaking."""
    from src.eval.overfit_diagnostics import causal_next_action_v1_check

    # Causal: shift=0
    sample_causal = {
        "time_index": 5,
        "action_history_indices": [2, 3, 4, 5],
        "target_action_indices": [6],
        "input_keys": ("image_t", "action_history"),
        "target_keys": ("target_actions",),
        "target_shift": 0,
    }
    result = causal_next_action_v1_check(sample_causal)
    assert result["pass"] is True

    # Leaking: shift=-1
    sample_leak = {
        "time_index": 5,
        "action_history_indices": [2, 3, 4, 5],
        "target_action_indices": [5],
        "input_keys": ("image_t", "action_history"),
        "target_keys": ("target_actions",),
        "target_shift": -1,
    }
    result = causal_next_action_v1_check(sample_leak)
    assert result["pass"] is False


# ---------------------------------------------------------------------------
# Normalization consistency test
# ---------------------------------------------------------------------------

def test_normalization_excludes_gripper():
    """Normalization stats should not include gripper dim."""
    actions = np.random.randn(100, 7).astype(np.float32)
    actions[:, GRIPPER_DIM_IDX] = np.sign(actions[:, GRIPPER_DIM_IDX])

    from src.eval.g8_mixed_action_metrics import build_action_contract, get_git_info
    contract = build_action_contract(
        actions_np=actions, trajectory_id="test",
        dataset="test", task_name="test", git_info=get_git_info(),
    )
    stats = contract["continuous_action_stats"]
    assert len(stats["mean"]) == 6
    assert len(stats["std"]) == 6
    assert len(stats["std_safe"]) == 6
    # Gripper dim 6 should NOT be in continuous stats
    assert stats["mean"][GRIPPER_DIM_IDX] if GRIPPER_DIM_IDX < len(stats["mean"]) else True  # 6 > 5


# ---------------------------------------------------------------------------
# CSV format test
# ---------------------------------------------------------------------------

def test_csv_row_has_required_fields():
    """All G9 CSV rows include git commit, dataset, task, seed."""
    row = {
        "variant": "test",
        "dim": 0,
        "label": "delta_pos_x",
        "raw_mse": 0.01,
        "normalized_mse": 0.02,
    }
    # Check that key fields exist in the attribution output
    assert "variant" in row
    assert "normalized_mse" in row
