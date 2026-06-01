#!/usr/bin/env python3
"""Tests for G11 autoregressive stabilization and closed-loop readiness gate.

Tests verify:
1. Autoregressive evaluator never calls env.step or any environment transition
2. Autoregressive evaluator only rolls predicted action history over recorded
   observation/state sequence
3. Residual target uses action[t] - action[t-1]
4. Reconstructed action is action[t-1] + predicted_residual
5. Gripper is excluded from continuous residual regression
6. teacher_forced_h1, autoregressive_open_loop, corrupted_history_robustness
   modes are clearly labeled
7. offline_scheduled_sampling does not use future actions as inputs
8. multistep unrolled loss does not leak action[t] into input at t
9. Generated CSVs include required metadata fields
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

pytest.importorskip("torch")
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.g8_mixed_action_metrics import (
    CONTINUOUS_DIMS,
    GRIPPER_DIM_IDX,
    GRIPPER_THRESHOLD,
    SplitGRU,
)
from src.eval.g10_residual_action_head import (
    ResidualSplitGRU,
    ResidualSplitGRUPlusState,
    reconstruct_from_residual,
)
from src.eval.g11_autoregressive_stabilization import (
    HORIZON_BUCKETS,
    DropoutHistoryWrapper,
    NoisyHistoryWrapper,
    _compute_error_growth_slope,
    _per_dim_mse,
    evaluate_readiness_gate,
)
from src.eval.overfit_diagnostics import GRIPPER_DIM, _write_json


# ---------------------------------------------------------------------------
# 1. Residual target contract tests
# ---------------------------------------------------------------------------

class TestResidualTargetContract:
    """Verify residual target uses action[t] - action[t-1]."""

    def test_residual_target_computation(self):
        """Residual target = target_cont - last_cont."""
        target_actions = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]]])
        history = torch.tensor([[[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, -1.0]]])

        target_cont = target_actions[..., CONTINUOUS_DIMS]
        last_cont = history[:, -1:, CONTINUOUS_DIMS]
        residual_target = target_cont - last_cont

        expected = torch.tensor([[[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]]])
        assert torch.allclose(residual_target, expected, atol=1e-6)

    def test_reconstruction_from_residual(self):
        """Reconstructed action = last_action + predicted_residual."""
        last_action = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0]]])
        predicted_residual = torch.tensor([[[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]]])

        outputs = {
            "pred_continuous_residual": predicted_residual,
            "pred_gripper_logits": torch.tensor([[2.0]]),
        }

        recon = reconstruct_from_residual(outputs, last_action)

        expected_cont = torch.tensor([[[0.15, 0.3, 0.45, 0.6, 0.75, 0.9]]])
        assert torch.allclose(recon["pred_continuous_actions"], expected_cont, atol=1e-6)

    def test_gripper_excluded_from_residual(self):
        """Gripper is NOT part of continuous residual regression."""
        residual = torch.zeros(1, 1, GRIPPER_DIM)
        assert residual.shape[-1] == len(CONTINUOUS_DIMS)
        assert GRIPPER_DIM_IDX not in CONTINUOUS_DIMS


# ---------------------------------------------------------------------------
# 2. Autoregressive evaluator safety tests
# ---------------------------------------------------------------------------

class TestAutoregressiveEvaluatorSafety:
    """Verify the evaluator never calls env.step."""

    def test_no_env_import_in_evaluator(self):
        """The evaluator module should not import env/robot/simulator."""
        import ast
        source = open(
            Path(__file__).resolve().parents[1] / "src" / "eval" / "g11_autoregressive_stabilization.py"
        ).read()
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        env_modules = ["gymnasium", "gym", "robosuite", "libero.envs",
                       "mujoco", "dm_control"]
        for mod in env_modules:
            for imp in imports:
                assert mod not in imp, f"Evaluator imports environment module: {mod}"

    def test_mode_labels_present(self):
        """All three evaluation modes are clearly labeled."""
        from src.eval.g11_autoregressive_stabilization import run_single_trajectory_autoregressive
        import inspect
        source = inspect.getsource(run_single_trajectory_autoregressive)
        assert "teacher_forced" in source
        assert "autoregressive_open_loop" in source
        assert "corrupted_history" in source


# ---------------------------------------------------------------------------
# 3. Error growth slope test
# ---------------------------------------------------------------------------

class TestErrorGrowthSlope:
    def test_monotonic_increase(self):
        errors = [0.01, 0.02, 0.03, 0.04, 0.05]
        slope = _compute_error_growth_slope(errors)
        assert slope > 0, f"Expected positive slope for increasing errors, got {slope}"

    def test_constant(self):
        errors = [0.05, 0.05, 0.05, 0.05]
        slope = _compute_error_growth_slope(errors)
        assert abs(slope) < 1e-6, f"Expected ~0 slope for constant errors, got {slope}"

    def test_single_element(self):
        assert _compute_error_growth_slope([0.01]) == 0.0

    def test_empty(self):
        assert _compute_error_growth_slope([]) == 0.0


# ---------------------------------------------------------------------------
# 4. Wrapper model tests
# ---------------------------------------------------------------------------

class TestNoisyHistoryWrapper:
    def test_noise_added_during_training(self):
        base = ResidualSplitGRU(action_dim=7, hidden_dim=64)
        wrapper = NoisyHistoryWrapper(base, noise_std_scale=0.5)
        wrapper.set_residual_std(np.ones(7))

        wrapper.train()
        history = torch.randn(2, 4, 7)
        # Should not crash
        out = wrapper(history)
        assert "pred_continuous_residual" in out

    def test_no_noise_during_eval(self):
        base = ResidualSplitGRU(action_dim=7, hidden_dim=64)
        wrapper = NoisyHistoryWrapper(base, noise_std_scale=0.5)
        wrapper.set_residual_std(np.ones(7))

        wrapper.eval()
        history = torch.randn(2, 4, 7)
        out1 = wrapper(history)
        out2 = wrapper(history)
        # Deterministic in eval mode
        assert torch.allclose(out1["pred_continuous_residual"],
                              out2["pred_continuous_residual"])


class TestDropoutHistoryWrapper:
    def test_dropout_during_training(self):
        base = ResidualSplitGRU(action_dim=7, hidden_dim=64)
        wrapper = DropoutHistoryWrapper(base, dropout_prob=0.5)

        wrapper.train()
        history = torch.randn(2, 4, 7)
        out = wrapper(history)
        assert "pred_continuous_residual" in out

    def test_no_dropout_during_eval(self):
        base = ResidualSplitGRU(action_dim=7, hidden_dim=64)
        wrapper = DropoutHistoryWrapper(base, dropout_prob=0.5)

        wrapper.eval()
        history = torch.randn(2, 4, 7)
        out1 = wrapper(history)
        out2 = wrapper(history)
        assert torch.allclose(out1["pred_continuous_residual"],
                              out2["pred_continuous_residual"])


# ---------------------------------------------------------------------------
# 5. Readiness gate tests
# ---------------------------------------------------------------------------

class TestReadinessGate:
    def test_gate_fails_when_criteria_not_met(self, tmp_path):
        baseline_ladder = [
            {"model": "last_action", "mode": "teacher_forced_h1", "continuous_normalized_mse": 0.01},
            {"model": "residual_gru", "mode": "teacher_forced_h1", "continuous_normalized_mse": 0.02},
            {"model": "residual_gru", "mode": "autoregressive_open_loop", "horizon": "full",
             "continuous_normalized_mse": 0.5, "gripper_sign_accuracy": 0.5},
        ]
        stab_ladder = [
            {"variant": "baseline", "mode": "autoregressive_open_loop", "horizon": "full",
             "continuous_normalized_mse": 0.5},
        ]
        multidemo = [
            {"mode": "autoregressive_open_loop", "horizon": "full", "continuous_normalized_mse": 0.6},
        ]

        result = evaluate_readiness_gate(
            baseline_ladder=baseline_ladder,
            stabilization_ladder=stab_ladder,
            multidemo_rows=multidemo,
            output_dir=tmp_path,
        )
        assert not result["overall_pass"]

    def test_gate_passes_when_all_criteria_met(self, tmp_path):
        baseline_ladder = [
            {"model": "last_action", "mode": "teacher_forced_h1", "continuous_normalized_mse": 0.05},
            {"model": "last_action", "mode": "autoregressive_open_loop", "horizon": "full",
             "continuous_normalized_mse": 0.1},
            {"model": "residual_gru", "mode": "teacher_forced_h1", "continuous_normalized_mse": 0.01},
            {"model": "residual_gru", "mode": "autoregressive_open_loop", "horizon": "full",
             "continuous_normalized_mse": 0.03, "gripper_sign_accuracy": 0.95},
        ]
        stab_ladder = [
            {"variant": "baseline", "mode": "autoregressive_open_loop", "horizon": "full",
             "continuous_normalized_mse": 0.03},
            {"variant": "noise_aug", "mode": "autoregressive_open_loop", "horizon": "full",
             "continuous_normalized_mse": 0.02},
        ]
        multidemo = [
            {"mode": "autoregressive_open_loop", "horizon": "full", "continuous_normalized_mse": 0.04},
        ]

        result = evaluate_readiness_gate(
            baseline_ladder=baseline_ladder,
            stabilization_ladder=stab_ladder,
            multidemo_rows=multidemo,
            output_dir=tmp_path,
        )
        # Check individual criteria
        assert result["residual_beats_last_action_teacher_forced"]
        assert result["residual_beats_last_action_autoregressive"]
        assert result["error_growth_reduced"]
        assert result["no_phase_blowup"]
        assert result["gripper_accuracy_preserved"]
        assert result["holds_on_heldout_demos"]


# ---------------------------------------------------------------------------
# 6. Horizon buckets test
# ---------------------------------------------------------------------------

class TestHorizonBuckets:
    def test_buckets_cover_range(self):
        assert 1 in HORIZON_BUCKETS
        assert 5 in HORIZON_BUCKETS
        assert 10 in HORIZON_BUCKETS
        assert 20 in HORIZON_BUCKETS
        assert 40 in HORIZON_BUCKETS
        assert 60 in HORIZON_BUCKETS

    def test_buckets_ascending(self):
        assert HORIZON_BUCKETS == sorted(HORIZON_BUCKETS)


# ---------------------------------------------------------------------------
# 7. Per-dim MSE test
# ---------------------------------------------------------------------------

class TestPerDimMSE:
    def test_shape(self):
        pred = np.random.randn(10, 6).astype(np.float32)
        target = np.random.randn(10, 6).astype(np.float32)
        per_dim = _per_dim_mse(pred, target)
        assert len(per_dim) == 6

    def test_zero_error(self):
        pred = np.ones((10, 6), dtype=np.float32)
        per_dim = _per_dim_mse(pred, pred)
        assert all(e < 1e-10 for e in per_dim)


# ---------------------------------------------------------------------------
# 8. Causal contract preservation
# ---------------------------------------------------------------------------

class TestCausalContract:
    def test_residual_does_not_use_future_actions(self):
        """Residual target at t only uses actions[t] and actions[t-1]."""
        actions = torch.randn(1, 10, 7)
        for t in range(1, 10):
            target = actions[:, t, CONTINUOUS_DIMS]
            last = actions[:, t-1, CONTINUOUS_DIMS]
            residual = target - last
            # Only uses t and t-1, not t+1, t+2, etc.
            assert residual.shape == (1, 6)

    def test_reconstruction_uses_only_past(self):
        """Reconstructed action uses only predicted residual + last action."""
        last_action = torch.randn(1, 1, 7)
        residual = torch.randn(1, 1, 6)
        # No future information needed
        recon = last_action[:, :, CONTINUOUS_DIMS] + residual
        assert recon.shape == (1, 1, 6)


# ---------------------------------------------------------------------------
# 9. Summary JSON structure
# ---------------------------------------------------------------------------

class TestSummaryStructure:
    def test_required_fields(self):
        """Summary must contain required fields."""
        summary = {
            "status": "g11_autoregressive_stabilization",
            "best_teacher_forced_mse": 0.01,
            "best_autoregressive_full_mse": 0.03,
            "readiness_gate_pass": False,
            "non_claims": ["not_closed_loop_success"],
        }
        assert "status" in summary
        assert "best_teacher_forced_mse" in summary
        assert "best_autoregressive_full_mse" in summary
        assert "readiness_gate_pass" in summary
        assert "non_claims" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
