"""Tests for G0 spatial patch latent pipeline.

Verifies:
- DINOv2PatchEncoder mock returns [B, N, D]
- return_cls=True returns patch tokens plus CLS token
- PatchLatentMetadata save/load roundtrip
- Dataset loads [T, N, D] patch latents without flattening
- Patch metrics compute correctly on [B, H, N, D]
- CLS latent tests still pass (no regression)
- Old configs still load
- G0 patch smoke config validates
- pool_patch_latents correctness
- collate_action_batch handles patch tensors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")
import torch
import yaml

from src.data.trajectory_window import (
    RawTrajectory,
    TrajectoryWindowDataset,
    make_mock_trajectory_dataset,
)
from src.models.encoders import (
    DEFAULT_DINOV2_REVISION,
    DINOv2PatchEncoder,
    PatchLatentMetadata,
)
from src.train.metrics import (
    patch_cosine_error,
    patch_mean_cosine_error,
    patch_mse,
)
from src.train.train_offline import (
    collate_action_batch,
    pool_patch_latents,
)


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# PatchLatentMetadata tests
# ---------------------------------------------------------------------------


class TestPatchLatentMetadata:
    def test_roundtrip_dict(self) -> None:
        meta = PatchLatentMetadata(
            encoder_name="dinov2-small",
            encoder_type="dinov2_patch",
            image_size=224,
            patch_size=14,
            num_patches=256,
            feature_dim=384,
            include_cls=False,
            dtype="float16",
            normalization="dino_internal",
            source_dataset="libero_spatial",
            revision="abc123",
        )
        d = meta.to_dict()
        restored = PatchLatentMetadata.from_dict(d)
        assert restored == meta

    def test_from_dict_extra_keys_ignored(self) -> None:
        d = {
            "encoder_name": "x",
            "encoder_type": "dinov2_patch",
            "image_size": 224,
            "patch_size": 14,
            "num_patches": 256,
            "feature_dim": 384,
            "include_cls": True,
            "dtype": "float16",
            "normalization": "dino_internal",
            "source_dataset": "",
            "git_commit": "",
            "revision": "a" * 40,
            "unknown_key": "ignored",
        }
        meta = PatchLatentMetadata.from_dict(d)
        assert meta.include_cls is True
        assert meta.revision == "a" * 40

    def test_frozen_dataclass(self) -> None:
        meta = PatchLatentMetadata(
            encoder_name="x",
            encoder_type="dinov2_patch",
            image_size=224,
            patch_size=14,
            num_patches=256,
            feature_dim=384,
            include_cls=False,
            dtype="float16",
            normalization="dino_internal",
        )
        with pytest.raises(AttributeError):
            meta.feature_dim = 512  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DINOv2PatchEncoder tests (mocked, no real model download)
# ---------------------------------------------------------------------------


class TestDINOv2PatchEncoder:
    def test_metadata_fields(self) -> None:
        enc = DINOv2PatchEncoder(
            model_id="facebook/dinov2-small",
            image_size=224,
            patch_size=14,
        )
        m = enc.metadata()
        assert m["encoder_type"] == "dinov2_patch"
        assert m["num_patches"] == 256
        assert m["model_id"] == "facebook/dinov2-small"
        assert m["frozen"] is True
        assert m["trainable_parameters"] == 0

    def test_patch_latent_metadata(self) -> None:
        enc = DINOv2PatchEncoder(
            model_id="facebook/dinov2-small",
            image_size=224,
            patch_size=14,
            return_cls=True,
        )
        meta = enc.patch_latent_metadata(source_dataset="test")
        assert isinstance(meta, PatchLatentMetadata)
        assert meta.num_patches == 256
        assert meta.include_cls is True
        assert meta.source_dataset == "test"

    def test_feature_dim_raises_before_forward(self) -> None:
        enc = DINOv2PatchEncoder(model_id="facebook/dinov2-small")
        with pytest.raises(RuntimeError, match="feature_dim is not yet known"):
            _ = enc.feature_dim

    def test_forward_rejects_non_tensor(self) -> None:
        enc = DINOv2PatchEncoder(model_id="facebook/dinov2-small")
        with pytest.raises(TypeError, match="tensor inputs"):
            enc("not_a_tensor")

    def test_forward_mock_returns_patch_shape(self) -> None:
        """Test with a mock model that returns realistic shapes."""
        enc = DINOv2PatchEncoder(
            model_id="facebook/dinov2-small",
            image_size=224,
            patch_size=14,
            return_cls=False,
        )
        mock_model = MagicMock()
        mock_processor = MagicMock()
        # Remove forward_features so _extract_patch_tokens uses standard forward
        del mock_model.forward_features

        B, N, D = 2, 256, 384
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(B, N + 1, D)
        mock_model.return_value = mock_output
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        def mock_processor_fn(**kwargs):
            imgs = kwargs.get("images", [])
            return {"pixel_values": torch.randn(len(imgs), 3, 224, 224)}

        mock_processor.side_effect = mock_processor_fn

        enc._model = mock_model
        enc._processor = mock_processor
        enc._feature_dim = D

        images = torch.randn(B, 3, 224, 224)
        result = enc(images)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (B, N, D)

    def test_forward_mock_return_cls(self) -> None:
        """Test return_cls=True returns dict with patch_tokens and cls_token."""
        enc = DINOv2PatchEncoder(
            model_id="facebook/dinov2-small",
            image_size=224,
            patch_size=14,
            return_cls=True,
        )
        mock_model = MagicMock()
        mock_processor = MagicMock()
        del mock_model.forward_features

        B, N, D = 2, 256, 384
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(B, N + 1, D)
        mock_model.return_value = mock_output
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        def mock_processor_fn(**kwargs):
            imgs = kwargs.get("images", [])
            return {"pixel_values": torch.randn(len(imgs), 3, 224, 224)}

        mock_processor.side_effect = mock_processor_fn

        enc._model = mock_model
        enc._processor = mock_processor
        enc._feature_dim = D

        images = torch.randn(B, 3, 224, 224)
        result = enc(images)
        assert isinstance(result, dict)
        assert "patch_tokens" in result
        assert "cls_token" in result
        assert result["patch_tokens"].shape == (B, N, D)
        assert result["cls_token"].shape == (B, D)


# ---------------------------------------------------------------------------
# Patch latent metrics tests
# ---------------------------------------------------------------------------


class TestPatchMSE:
    def test_perfect_prediction(self) -> None:
        target = torch.randn(2, 3, 8, 16)
        result = patch_mse(target, target)
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_known_error(self) -> None:
        pred = torch.zeros(1, 1, 4, 8)
        target = torch.ones(1, 1, 4, 8)
        result = patch_mse(pred, target)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_reduction_none(self) -> None:
        pred = torch.randn(2, 3, 4, 8)
        target = torch.randn(2, 3, 4, 8)
        result = patch_mse(pred, target, reduction="none")
        assert result.shape == (2, 3)

    def test_reduction_per_horizon(self) -> None:
        pred = torch.randn(4, 5, 8, 16)
        target = torch.randn(4, 5, 8, 16)
        result = patch_mse(pred, target, reduction="per_horizon")
        assert result.shape == (5,)

    def test_mask(self) -> None:
        pred = torch.randn(2, 3, 4, 8)
        target = torch.randn(2, 3, 4, 8)
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        result = patch_mse(pred, target, mask=mask, reduction="mean")
        assert result.ndim == 0

    def test_no_horizon(self) -> None:
        """Input [B, N, D] is treated as [B, 1, N, D]."""
        pred = torch.randn(2, 4, 8)
        target = torch.randn(2, 4, 8)
        result = patch_mse(pred, target, reduction="none")
        assert result.shape == (2, 1)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            patch_mse(torch.randn(2, 3, 4, 8), torch.randn(2, 3, 5, 8))

    def test_dtype_check(self) -> None:
        with pytest.raises(TypeError, match="floating point"):
            patch_mse(
                torch.randint(0, 10, (1, 1, 4, 8)),
                torch.randint(0, 10, (1, 1, 4, 8)),
            )


class TestPatchCosineError:
    def test_perfect_prediction(self) -> None:
        target = torch.randn(2, 3, 8, 16)
        result = patch_cosine_error(target, target)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal(self) -> None:
        pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        result = patch_cosine_error(pred, target)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_reduction_none(self) -> None:
        pred = torch.randn(2, 3, 4, 8)
        target = torch.randn(2, 3, 4, 8)
        result = patch_cosine_error(pred, target, reduction="none")
        assert result.shape == (2, 3)

    def test_reduction_per_horizon(self) -> None:
        pred = torch.randn(4, 5, 8, 16)
        target = torch.randn(4, 5, 8, 16)
        result = patch_cosine_error(pred, target, reduction="per_horizon")
        assert result.shape == (5,)


class TestPatchMeanCosineError:
    def test_perfect_prediction(self) -> None:
        target = torch.randn(2, 3, 8, 16)
        result = patch_mean_cosine_error(target, target)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_reduction_none(self) -> None:
        pred = torch.randn(2, 3, 4, 8)
        target = torch.randn(2, 3, 4, 8)
        result = patch_mean_cosine_error(pred, target, reduction="none")
        assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# Dataset patch latent tests
# ---------------------------------------------------------------------------


class TestPatchLatentDataset:
    def test_mock_patch_latent_shapes(self) -> None:
        ds = make_mock_trajectory_dataset(
            length=12,
            history_len=3,
            action_horizon=3,
            future_horizon=2,
            latent_dim=4,
            num_patches=6,
            include_current_patch_latents=True,
            include_future_patch_latents=True,
        )
        assert len(ds) > 0
        sample = ds[0]
        # z_t_patch should be [N, D]
        assert "z_t_patch" in sample
        z_patch = torch.as_tensor(sample["z_t_patch"])
        assert z_patch.shape == (6, 4)
        # target_future_patch_latents should be [H, N, D]
        assert "target_future_patch_latents" in sample
        fp = torch.as_tensor(sample["target_future_patch_latents"])
        assert fp.shape == (2, 6, 4)

    def test_no_flatten(self) -> None:
        """Patch latents are not silently flattened."""
        ds = make_mock_trajectory_dataset(
            length=12,
            history_len=3,
            action_horizon=3,
            future_horizon=2,
            latent_dim=8,
            num_patches=16,
            include_current_patch_latents=True,
            include_future_patch_latents=True,
        )
        sample = ds[0]
        z = torch.as_tensor(sample["z_t_patch"])
        assert z.ndim == 2  # [N, D], not flattened to [N*D]
        fp = torch.as_tensor(sample["target_future_patch_latents"])
        assert fp.ndim == 3  # [H, N, D], not flattened

    def test_cls_still_works(self) -> None:
        """CLS latent path remains backward-compatible."""
        ds = make_mock_trajectory_dataset(
            length=12,
            history_len=3,
            action_horizon=3,
            future_horizon=2,
            latent_dim=4,
            include_current_latent=True,
            include_future_latents=True,
        )
        sample = ds[0]
        z = torch.as_tensor(sample["z_t"])
        assert z.shape == (4,)
        assert "z_t_patch" not in sample or sample["z_t_patch"] is None

    def test_both_together(self) -> None:
        """CLS and patch latents can coexist."""
        ds = make_mock_trajectory_dataset(
            length=12,
            history_len=3,
            action_horizon=3,
            future_horizon=2,
            latent_dim=4,
            num_patches=6,
            include_current_latent=True,
            include_future_latents=True,
            include_current_patch_latents=True,
            include_future_patch_latents=True,
        )
        sample = ds[0]
        assert sample["z_t"] is not None
        assert sample["z_t_patch"] is not None
        z_cls = torch.as_tensor(sample["z_t"])
        z_patch = torch.as_tensor(sample["z_t_patch"])
        assert z_cls.shape == (4,)
        assert z_patch.shape == (6, 4)

    def test_missing_patch_latents_raises(self) -> None:
        traj = RawTrajectory(
            images=list(range(10)),
            actions=list(range(10)),
            language="test",
        )
        with pytest.raises(ValueError, match="patch_latents are required"):
            TrajectoryWindowDataset(
                [traj],
                history_len=3,
                action_horizon=3,
                future_horizon=2,
                include_current_patch_latents=True,
            )

    def test_raw_trajectory_validates_patch_length(self) -> None:
        traj = RawTrajectory(
            images=list(range(10)),
            actions=list(range(10)),
            language="test",
            patch_latents=list(range(5)),  # wrong length
        )
        with pytest.raises(ValueError, match="patch_latents length"):
            traj.validate()


# ---------------------------------------------------------------------------
# Collate and pooling tests
# ---------------------------------------------------------------------------


class TestCollatePatchLatents:
    def test_collate_includes_z_t_patch(self) -> None:
        samples = [
            {
                "action_history": [[1.0, 2.0, 3.0]] * 4,
                "target_actions": [[4.0, 5.0, 6.0]] * 4,
                "z_t_patch": [[1.0] * 8] * 6,
                "trajectory_id": "a",
                "time_index": 0,
                "language": "test",
                "task_name": "t",
            }
        ]
        batch = collate_action_batch(samples)
        assert "z_t_patch" in batch
        assert batch["z_t_patch"].shape == (1, 6, 8)

    def test_collate_includes_future_patch(self) -> None:
        samples = [
            {
                "action_history": [[1.0, 2.0]] * 4,
                "target_actions": [[3.0, 4.0]] * 4,
                "z_t_patch": [[1.0] * 4] * 6,
                "target_future_patch_latents": [[[1.0] * 4] * 6] * 2,
                "trajectory_id": "a",
                "time_index": 0,
                "language": "test",
                "task_name": "t",
            }
        ]
        batch = collate_action_batch(samples)
        assert "target_future_patch_latents" in batch
        assert batch["target_future_patch_latents"].shape == (1, 2, 6, 4)


class TestPoolPatchLatents:
    def test_3d_pooling(self) -> None:
        x = torch.randn(4, 8, 16)
        result = pool_patch_latents(x)
        assert result.shape == (4, 16)
        # Should be mean over dim 1
        expected = x.mean(dim=1)
        assert torch.allclose(result, expected)

    def test_4d_pooling(self) -> None:
        x = torch.randn(4, 3, 8, 16)
        result = pool_patch_latents(x)
        assert result.shape == (4, 3, 16)

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            pool_patch_latents(torch.randn(4, 16))


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestG0ConfigValidation:
    def test_g0_smoke_config_loads(self) -> None:
        config_path = ROOT / "configs" / "smoke" / "g0_patch_latent_smoke.yaml"
        if not config_path.exists():
            pytest.skip("G0 smoke config not found")
        config = yaml.safe_load(config_path.read_text())
        assert config["data"]["latent_type"] == "patch"
        assert config["model"]["temporal_adapter"] == "wam_gru"

    def test_existing_wam_gru_config_still_loads(self) -> None:
        config_path = ROOT / "configs" / "smoke" / "libero_spatial_wam_gru.yaml"
        config = yaml.safe_load(config_path.read_text())
        assert config["model"]["temporal_adapter"] == "wam_gru"


# ---------------------------------------------------------------------------
# Backward compatibility: existing CLS encoder unchanged
# ---------------------------------------------------------------------------


class TestCLSBackwardCompatibility:
    def test_dinov2_encoder_unchanged(self) -> None:
        from src.models.encoders import DINOv2VisualEncoder

        enc = DINOv2VisualEncoder(revision=None, latent_dim=384)
        assert enc.latent_dim == 384
        assert enc.output_token == "cls"
        m = enc.metadata()
        assert m["encoder_id"] == "dinov2_dinov2-small"
        assert m["frozen"] is True

    def test_build_frozen_visual_encoder_rejects_patch(self) -> None:
        from src.models.encoders import build_frozen_visual_encoder

        config = {
            "visual_encoder": "dinov2_patch_small",
            "visual_latent_dim": 384,
            "model_id": "facebook/dinov2-small",
            "patch_size": 14,
            "image_size": 224,
        }
        with pytest.raises(ValueError, match="DINOv2PatchEncoder cannot be built"):
            build_frozen_visual_encoder(config)

    def test_smoke_encoder_unchanged(self) -> None:
        from src.models.encoders import SmokeTimeIndexVisualEncoder

        enc = SmokeTimeIndexVisualEncoder(latent_dim=8)
        images = torch.tensor([5.0])
        result = enc(images)
        assert result.shape == (1, 8)
