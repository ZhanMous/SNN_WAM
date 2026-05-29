"""Synthetic tests for offline metric definitions.

Every metric is a scientific claim in code. These tests verify:
- Perfect prediction gives zero error.
- Known vector cases give expected values.
- Masked/padded horizons do not affect results.
- Shape and dtype contracts are enforced.
- Reduction dimensions are correct.
- Direction: lower is better for all metrics.
"""

from __future__ import annotations

import pytest
import torch

from src.train.metrics import (
    action_mse,
    action_mse_per_dimension,
    action_mse_per_horizon,
    future_latent_cosine_error,
    future_latent_mse,
)


# ---------------------------------------------------------------------------
# action_mse tests
# ---------------------------------------------------------------------------


def test_action_mse_perfect_prediction_is_zero() -> None:
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    assert action_mse(target, target).item() == pytest.approx(0.0)


def test_action_mse_known_value() -> None:
    pred = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    # MSE = (1 + 4 + 9 + 16) / 4 = 7.5
    assert action_mse(pred, target).item() == pytest.approx(7.5)


def test_action_mse_masked_padding_does_not_affect_result() -> None:
    pred = torch.tensor([[[1.0, 2.0], [99.0, 99.0]]])
    target = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])

    # With mask, only first horizon step counts -> MSE = 0.0
    assert action_mse(pred, target, mask=mask).item() == pytest.approx(0.0)
    # Without mask, second horizon contributes error
    assert action_mse(pred, target).item() > 0.0


def test_action_mse_mask_3d_shape() -> None:
    pred = torch.tensor([[[1.0, 2.0], [99.0, 99.0]]])
    target = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    mask = torch.tensor([[[1.0], [0.0]]])  # [B, H, 1]

    assert action_mse(pred, target, mask=mask).item() == pytest.approx(0.0)


def test_action_mse_batch_reduction() -> None:
    # Two batches with different errors -> mean over both
    pred = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
    target = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])
    # Batch 0: MSE = (4+0)/2 = 2.0; Batch 1: MSE = (0+16)/2 = 8.0
    # Global MSE = (4+16)/4 = 5.0
    assert action_mse(pred, target).item() == pytest.approx(5.0)


def test_action_mse_rejects_2d_shape() -> None:
    with pytest.raises(ValueError, match=r"\[B, H, A\]"):
        action_mse(torch.zeros(2, 3), torch.zeros(2, 3))


def test_action_mse_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        action_mse(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))


def test_action_mse_rejects_non_floating_point() -> None:
    with pytest.raises(TypeError, match="floating point"):
        action_mse(torch.zeros(1, 2, 3, dtype=torch.long), torch.zeros(1, 2, 3))


def test_action_mse_rejects_empty_mask() -> None:
    pred = torch.zeros(1, 2, 3)
    target = torch.zeros(1, 2, 3)
    mask = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="at least one valid"):
        action_mse(pred, target, mask=mask)


def test_action_mse_rejects_wrong_mask_shape() -> None:
    pred = torch.zeros(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    mask = torch.zeros(2, 4)  # wrong: H=4 but pred H=3
    with pytest.raises(ValueError, match="mask must have shape"):
        action_mse(pred, target, mask=mask)


def test_action_mse_direction_lower_is_better() -> None:
    target = torch.tensor([[[1.0, 2.0, 3.0]]])
    close_pred = torch.tensor([[[1.1, 2.1, 3.1]]])
    far_pred = torch.tensor([[[5.0, 6.0, 7.0]]])
    assert action_mse(close_pred, target) < action_mse(far_pred, target)


# ---------------------------------------------------------------------------
# future_latent_cosine_error tests
# ---------------------------------------------------------------------------


def test_future_latent_cosine_error_perfect_prediction_is_zero() -> None:
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert future_latent_cosine_error(target, target).item() == pytest.approx(0.0)


def test_future_latent_cosine_error_known_vector_cases() -> None:
    pred = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    per_horizon = future_latent_cosine_error(
        pred,
        target,
        reduction="per_horizon",
    )

    assert per_horizon.tolist() == pytest.approx([1.0, 2.0])
    assert future_latent_cosine_error(pred, target).item() == pytest.approx(1.5)


def test_future_latent_cosine_error_orthogonal_vectors_give_one() -> None:
    pred = torch.tensor([[[1.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0, 0.0]]])
    assert future_latent_cosine_error(pred, target).item() == pytest.approx(1.0)


def test_future_latent_cosine_error_opposite_vectors_give_two() -> None:
    pred = torch.tensor([[[1.0, 0.0, 0.0]]])
    target = torch.tensor([[[-1.0, 0.0, 0.0]]])
    assert future_latent_cosine_error(pred, target).item() == pytest.approx(2.0)


def test_future_latent_cosine_error_masked_padding_does_not_affect_result() -> None:
    pred = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])

    assert future_latent_cosine_error(pred, target, mask=mask).item() == pytest.approx(
        0.0
    )


def test_future_latent_cosine_error_reduction_none_returns_b_h() -> None:
    pred = torch.randn(3, 4, 8)
    target = torch.randn(3, 4, 8)
    result = future_latent_cosine_error(pred, target, reduction="none")
    assert result.shape == (3, 4)


def test_future_latent_cosine_error_reduction_per_horizon_returns_h() -> None:
    pred = torch.randn(3, 4, 8)
    target = torch.randn(3, 4, 8)
    result = future_latent_cosine_error(pred, target, reduction="per_horizon")
    assert result.shape == (4,)


def test_future_latent_cosine_error_batch_reduction() -> None:
    # Two batches: batch 0 is perfect, batch 1 is orthogonal
    pred = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    # Batch 0: error=0, Batch 1: error=1 -> mean=0.5
    assert future_latent_cosine_error(pred, target).item() == pytest.approx(0.5)


def test_future_latent_cosine_error_mask_with_reduction_none() -> None:
    pred = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    result = future_latent_cosine_error(pred, target, mask=mask, reduction="none")
    assert result.shape == (1, 2)
    assert result[0, 0].item() == pytest.approx(0.0)
    # Masked position should be 0 (weighted by mask=0)
    assert result[0, 1].item() == pytest.approx(0.0)


def test_future_latent_cosine_error_mask_with_reduction_per_horizon() -> None:
    # Two batches, two horizon steps. Mask ensures both horizons have >= 1 valid.
    pred = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = future_latent_cosine_error(
        pred, target, mask=mask, reduction="per_horizon"
    )
    assert result.shape == (2,)
    # Horizon 0: only batch 0 is valid, error=0.0
    assert result[0].item() == pytest.approx(0.0)
    # Horizon 1: only batch 1 is valid, error=2.0 (opposite)
    assert result[1].item() == pytest.approx(2.0)


def test_future_latent_cosine_error_per_horizon_fully_masked_raises() -> None:
    pred = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="at least one valid"):
        future_latent_cosine_error(
            pred, target, mask=mask, reduction="per_horizon"
        )


def test_future_latent_cosine_error_rejects_2d_shape() -> None:
    with pytest.raises(ValueError, match=r"\[B, H, D\]"):
        future_latent_cosine_error(torch.zeros(2, 3), torch.zeros(2, 3))


def test_future_latent_cosine_error_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        future_latent_cosine_error(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))


def test_future_latent_cosine_error_rejects_non_floating_point() -> None:
    with pytest.raises(TypeError, match="floating point"):
        future_latent_cosine_error(
            torch.zeros(1, 2, 3, dtype=torch.long),
            torch.zeros(1, 2, 3),
        )


def test_future_latent_cosine_error_rejects_invalid_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        future_latent_cosine_error(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), reduction="sum"
        )


def test_future_latent_cosine_error_direction_lower_is_better() -> None:
    target = torch.tensor([[[1.0, 0.0, 0.0]]])
    close_pred = torch.tensor([[[0.9, 0.1, 0.0]]])
    far_pred = torch.tensor([[[-1.0, 0.0, 0.0]]])
    assert future_latent_cosine_error(close_pred, target) < future_latent_cosine_error(
        far_pred, target
    )


# ---------------------------------------------------------------------------
# action_mse_per_horizon tests
# ---------------------------------------------------------------------------


def test_action_mse_per_horizon_perfect_prediction_is_zero() -> None:
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    result = action_mse_per_horizon(target, target)
    assert result.shape == (2,)
    assert result.tolist() == pytest.approx([0.0, 0.0])


def test_action_mse_per_horizon_known_value() -> None:
    pred = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    target = torch.tensor([[[2.0, 4.0], [0.0, 0.0]]])
    # Horizon 0: MSE = (4+16)/2 = 10.0; Horizon 1: MSE = 0.0
    result = action_mse_per_horizon(pred, target)
    assert result.shape == (2,)
    assert result[0].item() == pytest.approx(10.0)
    assert result[1].item() == pytest.approx(0.0)


def test_action_mse_per_horizon_masked_padding() -> None:
    # Two batches, mask ensures horizon 1 has exactly 1 valid entry
    pred = torch.tensor([[[1.0, 2.0], [99.0, 99.0]], [[1.0, 2.0], [99.0, 99.0]]])
    target = torch.tensor([[[1.0, 2.0], [0.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    result = action_mse_per_horizon(pred, target, mask=mask)
    # Horizon 0: both valid, perfect -> 0.0
    assert result[0].item() == pytest.approx(0.0)
    # Horizon 1: only batch 1 valid, error = (99^2 + 99^2)/2 = 9801
    assert result[1].item() == pytest.approx(9801.0)


def test_action_mse_per_horizon_batch_averaging() -> None:
    # Two batches, two horizons, two action dims
    # H0: batch0 SE=[4,0] mean=2; batch1 SE=[0,16] mean=8 -> H0 mean=(2+8)/2=5
    # H1: both zero
    pred = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    target = torch.tensor([[[2.0, 0.0], [0.0, 0.0]], [[0.0, 4.0], [0.0, 0.0]]])
    result = action_mse_per_horizon(pred, target)
    assert result[0].item() == pytest.approx(5.0)
    assert result[1].item() == pytest.approx(0.0)


def test_action_mse_per_horizon_direction_lower_is_better() -> None:
    target = torch.tensor([[[1.0, 2.0, 3.0]]])
    close_pred = torch.tensor([[[1.1, 2.1, 3.1]]])
    far_pred = torch.tensor([[[5.0, 6.0, 7.0]]])
    assert action_mse_per_horizon(close_pred, target)[0] < action_mse_per_horizon(
        far_pred, target
    )[0]


def test_action_mse_per_horizon_rejects_2d() -> None:
    with pytest.raises(ValueError, match=r"\[B, H, A\]"):
        action_mse_per_horizon(torch.zeros(2, 3), torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# action_mse_per_dimension tests
# ---------------------------------------------------------------------------


def test_action_mse_per_dimension_perfect_prediction_is_zero() -> None:
    target = torch.tensor([[[1.0, 2.0, 3.0]]])
    result = action_mse_per_dimension(target, target)
    assert result.shape == (3,)
    assert result.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_action_mse_per_dimension_known_value() -> None:
    pred = torch.tensor([[[0.0, 0.0, 0.0]]])
    target = torch.tensor([[[2.0, 4.0, 6.0]]])
    # Dim 0: MSE=4, Dim 1: MSE=16, Dim 2: MSE=36
    result = action_mse_per_dimension(pred, target)
    assert result.shape == (3,)
    assert result[0].item() == pytest.approx(4.0)
    assert result[1].item() == pytest.approx(16.0)
    assert result[2].item() == pytest.approx(36.0)


def test_action_mse_per_dimension_masked() -> None:
    pred = torch.tensor([[[1.0, 2.0], [99.0, 99.0]]])
    target = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    result = action_mse_per_dimension(pred, target, mask=mask)
    assert result.shape == (2,)
    assert result.tolist() == pytest.approx([0.0, 0.0])


def test_action_mse_per_dimension_batch_aggregation() -> None:
    # Two batches, two dimensions
    pred = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
    target = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])
    # All elements: errors are [4, 0, 0, 16] -> dim 0 mean=(4+0)/2=2, dim 1 mean=(0+16)/2=8
    result = action_mse_per_dimension(pred, target)
    assert result[0].item() == pytest.approx(2.0)
    assert result[1].item() == pytest.approx(8.0)


def test_action_mse_per_dimension_direction_lower_is_better() -> None:
    target = torch.tensor([[[1.0, 2.0, 3.0]]])
    close_pred = torch.tensor([[[1.1, 2.1, 3.1]]])
    far_pred = torch.tensor([[[5.0, 6.0, 7.0]]])
    assert all(
        action_mse_per_dimension(close_pred, target)[i]
        < action_mse_per_dimension(far_pred, target)[i]
        for i in range(3)
    )


def test_action_mse_per_dimension_rejects_2d() -> None:
    with pytest.raises(ValueError, match=r"\[B, H, A\]"):
        action_mse_per_dimension(torch.zeros(2, 3), torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# future_latent_mse tests
# ---------------------------------------------------------------------------


def test_future_latent_mse_perfect_prediction_is_zero() -> None:
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert future_latent_mse(target, target).item() == pytest.approx(0.0)


def test_future_latent_mse_known_value() -> None:
    pred = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[3.0, 4.0]]])
    # MSE = (9 + 16) / 2 = 12.5
    assert future_latent_mse(pred, target).item() == pytest.approx(12.5)


def test_future_latent_mse_reduction_none_returns_b_h() -> None:
    pred = torch.randn(3, 4, 8)
    target = torch.randn(3, 4, 8)
    result = future_latent_mse(pred, target, reduction="none")
    assert result.shape == (3, 4)


def test_future_latent_mse_reduction_per_horizon_returns_h() -> None:
    pred = torch.randn(3, 4, 8)
    target = torch.randn(3, 4, 8)
    result = future_latent_mse(pred, target, reduction="per_horizon")
    assert result.shape == (4,)


def test_future_latent_mse_masked_padding_does_not_affect_result() -> None:
    pred = torch.tensor([[[1.0, 0.0], [99.0, 99.0]]])
    target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    assert future_latent_mse(pred, target, mask=mask).item() == pytest.approx(0.0)
    # Without mask, second horizon contributes error
    assert future_latent_mse(pred, target).item() > 0.0


def test_future_latent_mse_mask_with_reduction_per_horizon() -> None:
    pred = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = future_latent_mse(pred, target, mask=mask, reduction="per_horizon")
    assert result.shape == (2,)
    # Horizon 0: only batch 0 valid, perfect -> 0.0
    assert result[0].item() == pytest.approx(0.0)
    # Horizon 1: only batch 1 valid, pred=[1,0] target=[0,1] -> MSE=(1+1)/2=1.0
    assert result[1].item() == pytest.approx(1.0)


def test_future_latent_mse_batch_reduction() -> None:
    pred = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
    target = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])
    # MSE reduces over D first: batch0 = (4+0)/2=2, batch1 = (0+16)/2=8
    # Then mean over B: (2+8)/2 = 5
    assert future_latent_mse(pred, target).item() == pytest.approx(5.0)


def test_future_latent_mse_rejects_2d() -> None:
    with pytest.raises(ValueError, match=r"\[B, H, D\]"):
        future_latent_mse(torch.zeros(2, 3), torch.zeros(2, 3))


def test_future_latent_mse_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        future_latent_mse(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))


def test_future_latent_mse_rejects_non_floating_point() -> None:
    with pytest.raises(TypeError, match="floating point"):
        future_latent_mse(torch.zeros(1, 2, 3, dtype=torch.long), torch.zeros(1, 2, 3))


def test_future_latent_mse_rejects_invalid_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        future_latent_mse(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), reduction="sum")


def test_future_latent_mse_direction_lower_is_better() -> None:
    target = torch.tensor([[[1.0, 0.0, 0.0]]])
    close_pred = torch.tensor([[[0.9, 0.1, 0.0]]])
    far_pred = torch.tensor([[[-1.0, 0.0, 0.0]]])
    assert future_latent_mse(close_pred, target) < future_latent_mse(far_pred, target)
