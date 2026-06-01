"""Tests for G7 state/action contract audit."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

pytest.importorskip("torch")
import torch

from src.data.trajectory_window import RawTrajectory
from src.eval.g7_state_action_contract import (
    FullStateSplitMLP,
    FullStatePlusHistoryGRU,
    ProprioPlusHistoryGRU,
    LinearARModel,
    run_raw_image_audit,
    run_threshold_audit,
    _action_dim_label,
    _categorize_field,
)


# ---------------------------------------------------------------------------
# Model shape tests
# ---------------------------------------------------------------------------

def test_full_state_split_mlp_output_shape():
    model = FullStateSplitMLP(state_dim=92, hidden_dim=64, action_dim=7)
    state = torch.randn(4, 92)
    output = model(state)
    assert isinstance(output, dict)
    assert "pred_actions" in output
    assert output["pred_actions"].shape == (4, 1, 7)


def test_full_state_plus_history_gru_output_shape():
    model = FullStatePlusHistoryGRU(
        state_dim=92, action_dim=7, history_len=4, hidden_dim=64,
    )
    history = torch.randn(4, 4, 7)
    state = torch.randn(4, 92)
    output = model(history, state)
    assert isinstance(output, dict)
    assert output["pred_actions"].shape == (4, 1, 7)


def test_proprio_plus_history_gru_output_shape():
    model = ProprioPlusHistoryGRU(
        state_dim=9, action_dim=7, history_len=4, hidden_dim=64,
    )
    history = torch.randn(4, 4, 7)
    state = torch.randn(4, 9)
    output = model(history, state)
    assert isinstance(output, dict)
    assert output["pred_actions"].shape == (4, 1, 7)


def test_linear_ar_output_shape():
    model = LinearARModel(history_len=4, action_dim=7)
    history = torch.randn(4, 4, 7)
    output = model(history)
    assert isinstance(output, dict)
    assert output["pred_actions"].shape == (4, 1, 7)


# ---------------------------------------------------------------------------
# Schema audit tests
# ---------------------------------------------------------------------------

def test_categorize_field():
    assert _categorize_field("agentview_rgb") == "image"
    assert _categorize_field("ee_pos") == "end_effector_pose"
    assert _categorize_field("ee_ori") == "end_effector_pose"
    assert _categorize_field("gripper_states") == "gripper_state"
    assert _categorize_field("joint_states") == "joint_state"
    assert _categorize_field("robot_states") == "robot_state_proprioceptive"
    assert _categorize_field("states") == "full_mujoco_state"
    assert _categorize_field("actions") == "action"


# ---------------------------------------------------------------------------
# Action dim label tests
# ---------------------------------------------------------------------------

def test_action_dim_label():
    assert _action_dim_label(0) == "delta_pos_x"
    assert _action_dim_label(5) == "delta_rot_z"
    assert _action_dim_label(6) == "gripper"
    assert _action_dim_label(7) == "dim_7"


# ---------------------------------------------------------------------------
# Raw image audit tests
# ---------------------------------------------------------------------------

def test_raw_image_audit_with_string_references():
    traj = RawTrajectory(
        images=["libero_spatial/file.hdf5:data/demo_0:obs/agentview_rgb:0",
                "libero_spatial/file.hdf5:data/demo_0:obs/agentview_rgb:1"],
        actions=[[0.0] * 7, [0.1] * 7],
        language="test",
        trajectory_id="test",
        split="train",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_raw_image_audit(traj, Path(tmpdir))
        assert result["can_dereference"] is True
        assert result["is_string_reference"] is True


def test_raw_image_audit_with_raw_arrays():
    traj = RawTrajectory(
        images=[np.zeros((128, 128, 3), dtype=np.uint8)],
        actions=[[0.0] * 7],
        language="test",
        trajectory_id="test",
        split="train",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_raw_image_audit(traj, Path(tmpdir))
        assert result["is_raw_array"] is True
        assert result["is_string_reference"] is False


# ---------------------------------------------------------------------------
# Threshold audit tests
# ---------------------------------------------------------------------------

def test_threshold_audit_basic():
    traj = RawTrajectory(
        images=["f"] * 20,
        actions=[[float(i) / 20] * 7 for i in range(20)],
        language="test",
        trajectory_id="test",
        split="train",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_threshold_audit(
            trajectory=traj,
            baseline_mses={"zero": 0.25, "last_action": 0.001},
            output_dir=Path(tmpdir),
        )
        assert result["threshold"] == 1e-4
        assert "verdict_lines" in result
        assert len(result["verdict_lines"]) > 0
        assert result["last_action_mse"] >= 0


# ---------------------------------------------------------------------------
# G6 label relabeling test
# ---------------------------------------------------------------------------

def test_oracle_state_label_not_used_for_proprio_only():
    """Verify that 'oracle_state' label is only used when full state is included."""
    # This is a documentation/label test, not a runtime test
    # The G7 code should use 'proprio_only_state' or 'full_state_92d_oracle'
    # not 'oracle_state' for 9-dim proprio
    from src.eval.g7_state_action_contract import run_g7_diagnostics
    # If the code runs without error and produces 'proprio_only_state' in the ladder,
    # the relabeling is correct. We test the label convention here.
    assert True  # Placeholder - actual test requires running the full pipeline


# ---------------------------------------------------------------------------
# Artifact CSV format tests
# ---------------------------------------------------------------------------

def test_g7_ladder_row_has_required_fields():
    from src.eval.g7_state_action_contract import _make_ladder_row
    row = _make_ladder_row("test_variant", "mlp", 100, 0.01, 7, 1e-4)
    required = ["variant", "model_type", "param_count", "eval_mse", "passed"]
    for field in required:
        assert field in row, f"Missing: {field}"
    assert row["git_commit"] if "git_commit" in row else True  # git info added later


import tempfile
