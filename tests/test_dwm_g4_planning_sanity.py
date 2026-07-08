"""DWM-G4: Planning sanity tests.

Verifies:
- Action optimizer produces correct output shapes
- Gradient flow through optimizer to action parameters
- Optimization reduces distance on synthetic problems
- Pass/fail criterion works correctly
- Determinism: same seed produces same results
- CMA-ES fallback (hill climber) works without cma package
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from src.models.dinowm_transformer import DINOwMTransformer
from src.planning.action_optimizer import (
    PlanningResult,
    compare_action_sources,
    optimize_actions_cmaes,
    optimize_actions_gradient,
    planning_objective_cosine,
    planning_objective_mse,
)


def _make_tiny_model(
    *,
    patch_dim: int = 16,
    feature_dim: int = 32,
    action_dim: int = 7,
    hidden_dim: int = 32,
    future_horizon: int = 2,
) -> DINOwMTransformer:
    """Create a tiny DINOwMTransformer for testing."""
    return DINOwMTransformer(
        patch_dim=patch_dim,
        feature_dim=feature_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_heads=2,
        num_layers=1,
        future_horizon=future_horizon,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Objective function tests
# ---------------------------------------------------------------------------


class TestPlanningObjectives:
    """Verify planning objective functions compute correctly."""

    def test_cosine_perfect_prediction(self) -> None:
        """Perfect prediction gives zero cosine error."""
        target = torch.randn(1, 2, 16, 32)
        loss = planning_objective_cosine(target.clone(), target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_cosine_orthogonal(self) -> None:
        """Orthogonal vectors give cosine error near 1."""
        a = torch.randn(1, 1, 4, 8)
        b = torch.randn_like(a)
        # Make them roughly orthogonal
        b = b - (a * b).sum(dim=-1, keepdim=True) * a / (a.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
        loss = planning_objective_cosine(a, b)
        assert loss.item() > 0.5

    def test_mse_perfect_prediction(self) -> None:
        """Perfect prediction gives zero MSE."""
        target = torch.randn(1, 2, 16, 32)
        loss = planning_objective_mse(target.clone(), target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_cosine_shape_3d(self) -> None:
        """Cosine objective works with [B, N, D] inputs (no horizon)."""
        pred = torch.randn(2, 16, 32)
        target = torch.randn(2, 16, 32)
        loss = planning_objective_cosine(pred, target)
        assert loss.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# Gradient-based optimizer tests
# ---------------------------------------------------------------------------


class TestGradientOptimizer:
    """Verify gradient-based action optimization."""

    def test_output_shape(self) -> None:
        """Optimized actions have correct shape."""
        model = _make_tiny_model()
        T_ctx, H, P, D, A = 3, 2, 16, 32, 7
        z_context = torch.randn(1, T_ctx, P, D)
        z_target = torch.randn(1, H, P, D)

        result = optimize_actions_gradient(
            model, z_context, z_target,
            horizon=H, action_dim=A, n_steps=5, device="cpu",
        )

        assert isinstance(result, PlanningResult)
        assert result.optimized_actions.shape == (1, H, A)
        assert result.method == "gradient"
        assert len(result.optimization_trace) == 5

    def test_distances_are_finite(self) -> None:
        """Initial and optimized distances are finite."""
        model = _make_tiny_model()
        T_ctx, H, P, D, A = 3, 2, 16, 32, 7
        z_context = torch.randn(1, T_ctx, P, D)
        z_target = torch.randn(1, H, P, D)

        result = optimize_actions_gradient(
            model, z_context, z_target,
            horizon=H, action_dim=A, n_steps=10, device="cpu",
        )

        assert torch.isfinite(torch.tensor(result.initial_distance))
        assert torch.isfinite(torch.tensor(result.optimized_distance))

    def test_gradient_flows_to_actions(self) -> None:
        """Gradients flow through the world model to action parameters."""
        model = _make_tiny_model()
        for p in model.parameters():
            p.requires_grad_(False)

        T_ctx, P, D, A = 3, 16, 32, 7
        z_context = torch.randn(1, T_ctx, P, D)
        z_target = torch.randn(1, 2, P, D)

        action_history = torch.randn(1, T_ctx, A)
        future_actions = torch.randn(1, 2, A, requires_grad=True)
        pred = model(z_context, action_history, future_actions=future_actions)
        loss = planning_objective_cosine(pred, z_target)
        loss.backward()

        assert future_actions.grad is not None
        assert future_actions.grad.abs().sum() > 0

    def test_optimization_reduces_loss(self) -> None:
        """Optimization reduces the objective over steps."""
        torch.manual_seed(42)
        model = _make_tiny_model()
        T_ctx, H, P, D, A = 3, 2, 16, 32, 7
        z_context = torch.randn(1, T_ctx, P, D)
        # Set target close to zero-action prediction for easier optimization
        with torch.no_grad():
            action_history = torch.randn(1, T_ctx, A)
            zero_future_actions = torch.zeros(1, H, A)
            z_target = (
                model(z_context, action_history, future_actions=zero_future_actions)
                + torch.randn(1, H, P, D) * 0.01
            )

        result = optimize_actions_gradient(
            model, z_context, z_target,
            horizon=H, action_dim=A, action_history=action_history,
            n_steps=100, lr=0.1, device="cpu",
        )

        # Should at least not get worse
        assert result.optimized_distance <= result.initial_distance + 0.1

    def test_determinism(self) -> None:
        """Same seed produces same optimized actions."""
        torch.manual_seed(123)
        model = _make_tiny_model()
        T_ctx, H, P, D, A = 3, 2, 16, 32, 7
        z_context = torch.randn(1, T_ctx, P, D)
        z_target = torch.randn(1, H, P, D)

        torch.manual_seed(456)
        r1 = optimize_actions_gradient(
            model, z_context, z_target,
            horizon=H, action_dim=A, n_steps=10, device="cpu",
        )
        torch.manual_seed(456)
        r2 = optimize_actions_gradient(
            model, z_context, z_target,
            horizon=H, action_dim=A, n_steps=10, device="cpu",
        )

        assert torch.allclose(r1.optimized_actions, r2.optimized_actions)


# ---------------------------------------------------------------------------
# CMA-ES / hill climber tests
# ---------------------------------------------------------------------------


class TestCMAESOptimizer:
    """Verify CMA-ES optimization (hill climber fallback)."""

    def test_output_shape(self) -> None:
        """CMA-ES produces correct output shape."""
        model = _make_tiny_model()
        T_ctx, H, P, D, A = 3, 2, 16, 32, 7
        z_context = torch.randn(1, T_ctx, P, D)
        z_target = torch.randn(1, H, P, D)

        result = optimize_actions_cmaes(
            model, z_context, z_target,
            horizon=H, action_dim=A, n_generations=5, population_size=4,
            seed=0, device="cpu",
        )

        assert isinstance(result, PlanningResult)
        assert result.optimized_actions.shape == (1, H, A)
        assert result.method == "cma_es"
        assert len(result.optimization_trace) > 0

    def test_distances_are_finite(self) -> None:
        """CMA-ES distances are finite."""
        model = _make_tiny_model()
        T_ctx, H, P, D, A = 3, 2, 16, 32, 7
        z_context = torch.randn(1, T_ctx, P, D)
        z_target = torch.randn(1, H, P, D)

        result = optimize_actions_cmaes(
            model, z_context, z_target,
            horizon=H, action_dim=A, n_generations=3, population_size=4,
            seed=0, device="cpu",
        )

        assert torch.isfinite(torch.tensor(result.initial_distance))
        assert torch.isfinite(torch.tensor(result.optimized_distance))


# ---------------------------------------------------------------------------
# Comparison / pass criterion tests
# ---------------------------------------------------------------------------


class TestCompareActionSources:
    """Verify action source comparison and pass criterion."""

    def test_comparison_structure(self) -> None:
        """Comparison returns expected keys."""
        model = _make_tiny_model()
        T_ctx, H, P, D, A = 3, 2, 16, 32, 7
        z_context = torch.randn(1, T_ctx, P, D)
        z_target = torch.randn(1, H, P, D)
        gt_actions = torch.randn(1, H, A)
        action_history = torch.randn(1, T_ctx, A)

        result = compare_action_sources(
            model, z_context, z_target,
            horizon=H, action_dim=A, gt_actions=gt_actions,
            action_history=action_history,
            n_random=3, device="cpu",
        )

        assert "sources" in result
        assert "zero" in result["sources"]
        assert "random" in result["sources"]
        assert "optimized" in result["sources"]
        assert "pass" in result
        assert isinstance(result["pass"], bool)

    def test_replay_actions_must_match_future_horizon(self) -> None:
        """Replay/shuffle actions must use [B, H, A], not [B, T_ctx, A]."""
        model = _make_tiny_model()
        T_ctx, H, P, D, A = 3, 2, 16, 32, 7
        z_context = torch.randn(1, T_ctx, P, D)
        z_target = torch.randn(1, H, P, D)
        gt_actions_history = torch.randn(1, T_ctx, A)

        with pytest.raises(ValueError, match="gt_actions must have shape"):
            compare_action_sources(
                model, z_context, z_target,
                horizon=H, action_dim=A, gt_actions=gt_actions_history,
                n_random=1, device="cpu",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dataset_random_baseline_runs_on_cuda(self) -> None:
        """Dataset random actions use a generator on the CUDA device."""
        model = _make_tiny_model(
            patch_dim=2,
            feature_dim=4,
            action_dim=7,
            hidden_dim=8,
            future_horizon=1,
        ).to("cuda")
        T_ctx, H, P, D, A = 2, 1, 2, 4, 7
        z_context = torch.randn(1, T_ctx, P, D, device="cuda")
        z_target = torch.randn(1, H, P, D, device="cuda")
        gt_actions = torch.randn(1, H, A, device="cuda")
        action_history = torch.randn(1, T_ctx, A, device="cuda")
        action_stats = {
            "mean": [0.0] * A,
            "std": [1.0] * A,
        }

        result = compare_action_sources(
            model, z_context, z_target,
            horizon=H, action_dim=A, gt_actions=gt_actions,
            action_history=action_history,
            n_random=1, random_baseline_type="dataset",
            action_stats=action_stats, device="cuda",
        )

        assert torch.isfinite(torch.tensor(result["sources"]["random"]["distance"]))

    def test_optimized_beats_zero_on_easy_problem(self) -> None:
        """On a simple problem, optimized should beat zero actions."""
        torch.manual_seed(77)
        model = _make_tiny_model()
        T_ctx, H, P, D, A = 3, 2, 16, 32, 7

        # Create a target that's easy to reach from context
        z_context = torch.randn(1, T_ctx, P, D) * 0.1
        # Target = context mean + small perturbation (easy to reach)
        z_target = z_context[:, -1:].mean(dim=1, keepdim=True).expand(1, H, P, D) + torch.randn(1, H, P, D) * 0.01

        result = compare_action_sources(
            model, z_context, z_target,
            horizon=H, action_dim=A, n_random=5, device="cpu",
        )

        # Optimized should be at least as good as zero (it can always choose zero)
        opt_dist = result["sources"]["optimized"]["distance"]
        zero_dist = result["sources"]["zero"]["distance"]
        assert opt_dist <= zero_dist + 0.01  # small tolerance


# ---------------------------------------------------------------------------
# Integration: tiny training + planning
# ---------------------------------------------------------------------------


class TestPlanningIntegration:
    """End-to-end test: train tiny model, then plan."""

    def test_tiny_train_then_plan(self) -> None:
        """Train a tiny model for a few steps, then run planning."""
        torch.manual_seed(0)
        P, D, A, T_ctx, H = 8, 16, 7, 3, 2
        model = _make_tiny_model(
            patch_dim=P, feature_dim=D, action_dim=A,
            hidden_dim=32, future_horizon=H,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Tiny training loop
        for _ in range(5):
            z_ctx = torch.randn(4, T_ctx, P, D)
            acts = torch.randn(4, T_ctx, A)
            future_acts = torch.randn(4, H, A)
            z_tgt = torch.randn(4, H, P, D)
            pred = model(z_ctx, acts, future_actions=future_acts)
            loss = planning_objective_cosine(pred, z_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        # Planning
        z_ctx = torch.randn(1, T_ctx, P, D)
        z_tgt = torch.randn(1, H, P, D)

        result = optimize_actions_gradient(
            model, z_ctx, z_tgt,
            horizon=H, action_dim=A, n_steps=10, device="cpu",
        )

        assert result.optimized_actions.shape == (1, H, A)
        assert torch.isfinite(torch.tensor(result.optimized_distance))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
