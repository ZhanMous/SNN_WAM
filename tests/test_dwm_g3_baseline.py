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

        output = model(patch_latents, actions)

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

        output = model.predict_one_step(patch_latents, actions)

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

        output = model(patch_latents, actions)

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

        output = model(patch_latents, actions)

        assert torch.isfinite(output).all(), "Model output contains NaN or Inf"


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
        z_target = torch.randn(B, H, P, D)

        # Train for a few steps
        initial_loss = None
        for step in range(10):
            z_pred = model(patch_latents, actions)
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
        z_target = torch.randn(B, H, P, D)

        z_pred = model(patch_latents, actions)
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
        z_target = torch.randn(B, H, P, D)

        z_pred = model(patch_latents, actions)

        mse = patch_mse(z_pred, z_target)
        cosine = patch_cosine_error(z_pred, z_target)

        assert mse.item() >= 0, "MSE should be non-negative"
        assert cosine.item() >= 0, "Cosine error should be non-negative"
        assert cosine.item() <= 2.0, "Cosine error should be <= 2.0 (range: 0=perfect, 2=opposite)"


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
