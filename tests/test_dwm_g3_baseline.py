"""DWM-G3: DINO-WM baseline tests for model and metrics.

Verifies:
- DINOwMTransformer produces correct output shapes
- Model can be trained on a tiny batch
- Patch latent metrics compute correctly
- One-step and multi-step predictions work
- Model parameters are trainable
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from scripts.eval_persistence_baseline import eval_persistence
from scripts.train_dinowm_baseline import run_one_split
from src.models.dinowm_transformer import DINOwMTransformer, build_dinowm_model
from src.train.metrics import patch_mse, patch_cosine_error


# ---------------------------------------------------------------------------
# DWM-G3 Gate 3: Model shape tests
# ---------------------------------------------------------------------------


class TestDWMG3ModelShapes:
    """Verify DINOwMTransformer produces correct shapes."""

    def test_forward_shape(self) -> None:
        """Forward pass produces correct output shape."""
        B, T, P, D, A, H = 2, 3, 256, 384, 7, 2

        model = DINOwMTransformer(
            patch_dim=P,
            feature_dim=D,
            action_dim=A,
            hidden_dim=128,
            num_heads=4,
            num_layers=1,
            future_horizon=H,
        )

        patch_latents = torch.randn(B, T, P, D)
        actions = torch.randn(B, T, A)
        future_actions = torch.randn(B, H, A)

        output = model(patch_latents, actions, future_actions=future_actions)

        assert output.shape == (B, H, P, D), (
            f"Expected shape ({B}, {H}, {P}, {D}), got {output.shape}"
        )

    def test_predict_one_step_shape(self) -> None:
        """predict_one_step produces correct output shape."""
        B, T, P, D, A = 2, 3, 256, 384, 7

        model = DINOwMTransformer(
            patch_dim=P,
            feature_dim=D,
            action_dim=A,
            hidden_dim=128,
            num_heads=4,
            num_layers=1,
            future_horizon=2,
        )

        patch_latents = torch.randn(B, T, P, D)
        actions = torch.randn(B, T, A)
        future_actions = torch.randn(B, 1, A)

        output = model.predict_one_step(patch_latents, actions, future_actions=future_actions)

        assert output.shape == (B, 1, P, D), (
            f"Expected shape ({B}, 1, {P}, {D}), got {output.shape}"
        )

    def test_single_timestep(self) -> None:
        """Model works with single timestep input."""
        B, T, P, D, A, H = 1, 1, 64, 128, 7, 2

        model = DINOwMTransformer(
            patch_dim=P,
            feature_dim=D,
            action_dim=A,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            future_horizon=H,
        )

        patch_latents = torch.randn(B, T, P, D)
        actions = torch.randn(B, T, A)
        future_actions = torch.randn(B, H, A)

        output = model(patch_latents, actions, future_actions=future_actions)

        assert output.shape == (B, H, P, D)

    def test_output_is_finite(self) -> None:
        """Model output is finite (no NaN or Inf)."""
        B, T, P, D, A, H = 2, 3, 64, 128, 7, 2

        model = DINOwMTransformer(
            patch_dim=P,
            feature_dim=D,
            action_dim=A,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            future_horizon=H,
        )

        patch_latents = torch.randn(B, T, P, D)
        actions = torch.randn(B, T, A)
        future_actions = torch.randn(B, H, A)

        output = model(patch_latents, actions, future_actions=future_actions)

        assert torch.isfinite(output).all(), "Model output contains NaN or Inf"

    def test_future_actions_shape(self) -> None:
        """Forward pass explicitly accepts future_actions [B, H, A]."""
        B, T, P, D, A, H = 2, 3, 32, 64, 7, 2
        model = DINOwMTransformer(
            patch_dim=P,
            feature_dim=D,
            action_dim=A,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            future_horizon=H,
            dropout=0.0,
        )

        z_context = torch.randn(B, T, P, D)
        action_history = torch.randn(B, T, A)
        future_actions = torch.randn(B, H, A)

        output = model(z_context, action_history, future_actions=future_actions)
        assert output.shape == (B, H, P, D)

    def test_future_actions_affect_prediction(self) -> None:
        """Changing future candidate actions changes model predictions."""
        torch.manual_seed(0)
        B, T, P, D, A, H = 2, 3, 16, 32, 7, 2
        model = DINOwMTransformer(
            patch_dim=P,
            feature_dim=D,
            action_dim=A,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            future_horizon=H,
            dropout=0.0,
        )
        model.eval()

        z_context = torch.randn(B, T, P, D)
        action_history = torch.randn(B, T, A)
        future_a = torch.zeros(B, H, A)
        future_b = torch.ones(B, H, A)

        with torch.no_grad():
            pred_a = model(z_context, action_history, future_actions=future_a)
            pred_b = model(z_context, action_history, future_actions=future_b)

        assert not torch.allclose(pred_a, pred_b)


# ---------------------------------------------------------------------------
# DWM-G3 Gate 3: Model training tests
# ---------------------------------------------------------------------------


class TestDWMG3ModelTraining:
    """Verify model can be trained on tiny batch."""

    def test_tiny_batch_training(self) -> None:
        """Model training reduces loss on tiny batch."""
        B, T, P, D, A, H = 2, 3, 32, 64, 7, 2

        model = DINOwMTransformer(
            patch_dim=P,
            feature_dim=D,
            action_dim=A,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            future_horizon=H,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed target
        patch_latents = torch.randn(B, T, P, D)
        actions = torch.randn(B, T, A)
        future_actions = torch.randn(B, H, A)
        z_target = torch.randn(B, H, P, D)

        # Train for a few steps
        initial_loss = None
        for step in range(10):
            z_pred = model(patch_latents, actions, future_actions=future_actions)
            loss = torch.nn.functional.mse_loss(z_pred, z_target)

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss, (
            f"Training did not reduce loss: {initial_loss} -> {final_loss}"
        )

    def test_gradient_flow(self) -> None:
        """Gradients flow through the model."""
        B, T, P, D, A, H = 2, 3, 32, 64, 7, 2

        model = DINOwMTransformer(
            patch_dim=P,
            feature_dim=D,
            action_dim=A,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            future_horizon=H,
        )

        patch_latents = torch.randn(B, T, P, D)
        actions = torch.randn(B, T, A)
        future_actions = torch.randn(B, H, A)
        z_target = torch.randn(B, H, P, D)

        z_pred = model(patch_latents, actions, future_actions=future_actions)
        loss = torch.nn.functional.mse_loss(z_pred, z_target)
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name}"
                )


# ---------------------------------------------------------------------------
# DWM-G3 Gate 3: Metric tests
# ---------------------------------------------------------------------------


class TestDWMG3Metrics:
    """Verify patch latent metrics compute correctly."""

    def test_patch_mse_perfect_prediction(self) -> None:
        """Patch MSE is zero for perfect prediction."""
        target = torch.randn(2, 3, 8, 16)
        result = patch_mse(target, target)
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_patch_mse_known_error(self) -> None:
        """Patch MSE computes correct error."""
        pred = torch.zeros(1, 1, 4, 8)
        target = torch.ones(1, 1, 4, 8)
        result = patch_mse(pred, target)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_patch_cosine_error_perfect(self) -> None:
        """Patch cosine error is zero for perfect prediction."""
        target = torch.randn(2, 3, 8, 16)
        result = patch_cosine_error(target, target)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_patch_cosine_error_orthogonal(self) -> None:
        """Patch cosine error is 1.0 for orthogonal vectors."""
        pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        result = patch_cosine_error(pred, target)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_metrics_on_model_output(self) -> None:
        """Metrics compute correctly on model output."""
        B, T, P, D, A, H = 2, 3, 32, 64, 7, 2

        model = DINOwMTransformer(
            patch_dim=P,
            feature_dim=D,
            action_dim=A,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            future_horizon=H,
        )

        patch_latents = torch.randn(B, T, P, D)
        actions = torch.randn(B, T, A)
        future_actions = torch.randn(B, H, A)
        z_target = torch.randn(B, H, P, D)

        z_pred = model(patch_latents, actions, future_actions=future_actions)

        mse = patch_mse(z_pred, z_target)
        cosine = patch_cosine_error(z_pred, z_target)

        assert mse.item() >= 0, "MSE should be non-negative"
        assert cosine.item() >= 0, "Cosine error should be non-negative"
        assert cosine.item() <= 2.0, "Cosine error should be <= 2.0 (range: 0=perfect, 2=opposite)"

    def test_run_one_split_overall_metrics_average_horizon(self) -> None:
        """Overall [B, H] patch metrics are averaged over batch and horizon."""

        class ZeroWorldModel(torch.nn.Module):
            def forward(
                self,
                z_context: torch.Tensor,
                actions: torch.Tensor,
                future_actions: torch.Tensor,
            ) -> torch.Tensor:
                B = z_context.shape[0]
                return torch.zeros(B, 2, 1, 2, device=z_context.device)

        batch = {
            "z_context": torch.zeros(3, 3, 1, 2),
            "actions": torch.zeros(3, 3, 7),
            "future_actions": torch.zeros(3, 2, 7),
            "z_target": torch.ones(3, 2, 1, 2),
        }
        metrics = run_one_split(
            ZeroWorldModel(),
            [batch],
            device=torch.device("cpu"),
            optimizer=None,
            lambda_patch_cosine=1.0,
            lambda_action=0.0,
            grad_clip_norm=None,
            max_steps=None,
        )

        assert metrics["samples"] == 3
        assert metrics["patch_mse"] == pytest.approx(1.0)
        assert metrics["patch_cosine_error"] == pytest.approx(1.0)
        assert metrics["patch_mean_cosine_error"] == pytest.approx(1.0)
        assert metrics["patch_mse_by_horizon"] == pytest.approx([1.0, 1.0])
        assert metrics["patch_cosine_error_by_horizon"] == pytest.approx([1.0, 1.0])

    def test_persistence_metrics_average_horizon(self) -> None:
        """Persistence overall metrics reduce [B, H] to one value per sample."""
        batch = {
            "z_context": torch.zeros(3, 3, 1, 2),
            "actions": torch.zeros(3, 3, 7),
            "z_target": torch.ones(3, 2, 1, 2),
        }
        metrics = eval_persistence([batch], eval_horizon=2)

        assert metrics["n_samples"] == 3
        assert metrics["patch_mse"] == pytest.approx(1.0)
        assert metrics["patch_cosine_error"] == pytest.approx(1.0)
        assert metrics["patch_mean_cosine_error"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DWM-G3 Gate 3: Build model tests
# ---------------------------------------------------------------------------


class TestDWMG3BuildModel:
    """Verify build_dinowm_model factory works."""

    def test_build_model(self) -> None:
        """build_dinowm_model creates model from config."""
        config = {
            "patch_dim": 256,
            "feature_dim": 384,
            "action_dim": 7,
            "hidden_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "future_horizon": 2,
            "dropout": 0.1,
        }

        model = build_dinowm_model(config)

        assert isinstance(model, DINOwMTransformer)
        assert model.patch_dim == 256
        assert model.feature_dim == 384
        assert model.action_dim == 7
        assert model.hidden_dim == 256
        assert model.future_horizon == 2

    def test_build_model_defaults(self) -> None:
        """build_dinowm_model uses sensible defaults."""
        config = {}

        model = build_dinowm_model(config)

        assert isinstance(model, DINOwMTransformer)
        assert model.patch_dim == 256
        assert model.feature_dim == 384
        assert model.action_dim == 7
