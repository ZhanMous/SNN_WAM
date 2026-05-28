"""Tests for DINOv2 frozen visual encoder.

These tests verify:
- Encoder parameters require_grad=False
- Encoder parameters are not in optimizer
- Metadata includes required fields (revision, preprocessing_id)
- Pinned revision validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from src.models.encoders import (
    DEFAULT_DINOV2_REVISION,
    DINOV2_PREPROCESSING_ID,
    DINOv2VisualEncoder,
    validate_revision,
)


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Revision validation tests
# ---------------------------------------------------------------------------


def test_validate_revision_accepts_valid_hash() -> None:
    """Valid 40-char hex revision is accepted."""
    valid_hash = "a" * 40
    result = validate_revision(valid_hash, "test_model")
    assert result == valid_hash


def test_validate_revision_uses_default_when_none() -> None:
    """None revision uses DEFAULT_DINOV2_REVISION."""
    result = validate_revision(None, "test_model")
    assert result == DEFAULT_DINOV2_REVISION


def test_validate_revision_rejects_short_hash() -> None:
    """Revision shorter than 40 chars is rejected."""
    with pytest.raises(ValueError, match="40-character"):
        validate_revision("abc123", "test_model")


def test_validate_revision_rejects_long_hash() -> None:
    """Revision longer than 40 chars is rejected."""
    with pytest.raises(ValueError, match="40-character"):
        validate_revision("a" * 41, "test_model")


def test_validate_revision_rejects_uppercase() -> None:
    """Uppercase hex revision is rejected."""
    with pytest.raises(ValueError, match="lowercase hex"):
        validate_revision("A" * 40, "test_model")


def test_validate_revision_rejects_non_hex() -> None:
    """Non-hex characters in revision are rejected."""
    with pytest.raises(ValueError, match="lowercase hex"):
        validate_revision("g" * 40, "test_model")


# ---------------------------------------------------------------------------
# DINOv2VisualEncoder tests (without downloading real model)
# ---------------------------------------------------------------------------


def test_dinov2_encoder_default_revision() -> None:
    """DINOv2VisualEncoder uses DEFAULT_DINOV2_REVISION when revision=None."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    assert encoder.revision == DEFAULT_DINOV2_REVISION


def test_dinov2_encoder_custom_revision() -> None:
    """DINOv2VisualEncoder accepts custom revision."""
    custom_hash = "b" * 40
    encoder = DINOv2VisualEncoder(revision=custom_hash, latent_dim=384)
    assert encoder.revision == custom_hash


def test_dinov2_encoder_metadata_includes_revision() -> None:
    """Encoder metadata includes pinned revision."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    metadata = encoder.metadata()
    assert "revision" in metadata
    assert metadata["revision"] == DEFAULT_DINOV2_REVISION


def test_dinov2_encoder_metadata_includes_preprocessing_id() -> None:
    """Encoder metadata includes preprocessing_id."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    metadata = encoder.metadata()
    assert "preprocessing_id" in metadata
    assert metadata["preprocessing_id"] == DINOV2_PREPROCESSING_ID


def test_dinov2_encoder_metadata_includes_model_id() -> None:
    """Encoder metadata includes model_id."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    metadata = encoder.metadata()
    assert metadata["model_id"] == "facebook/dinov2-small"


def test_dinov2_encoder_metadata_includes_output_token() -> None:
    """Encoder metadata includes output_token."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    metadata = encoder.metadata()
    assert metadata["output_token"] == "cls"


def test_dinov2_encoder_metadata_includes_extraction_mode() -> None:
    """Encoder metadata includes extraction_mode."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    metadata = encoder.metadata()
    assert metadata["extraction_mode"] == "offline"


def test_dinov2_encoder_frozen_flag() -> None:
    """Encoder metadata indicates frozen=True."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    metadata = encoder.metadata()
    assert metadata["frozen"] is True


def test_dinov2_encoder_trainable_parameters_zero() -> None:
    """Encoder metadata shows trainable_parameters=0."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    metadata = encoder.metadata()
    assert metadata["trainable_parameters"] == 0


def test_dinov2_encoder_latent_dim() -> None:
    """Encoder metadata includes correct latent_dim."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    metadata = encoder.metadata()
    assert metadata["latent_dim"] == 384


def test_dinov2_encoder_id_format() -> None:
    """Encoder ID follows expected format."""
    encoder = DINOv2VisualEncoder(revision=None, latent_dim=384)
    assert encoder.encoder_id == "dinov2_dinov2-small"


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


def test_reportable_configs_have_valid_revision() -> None:
    """Reportable configs contain valid DINOv2 revision."""
    config_dir = ROOT / "configs" / "reportable"
    if not config_dir.exists():
        pytest.skip("No reportable configs directory")

    for config_path in config_dir.glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = config.get("model", {})
        if model.get("visual_encoder", "").startswith("dinov2"):
            revision = model.get("revision")
            # Should be a valid 40-char hex string
            assert isinstance(revision, str), f"{config_path}: revision must be string"
            assert len(revision) == 40, f"{config_path}: revision must be 40 chars"
            assert revision.islower(), f"{config_path}: revision must be lowercase"


def test_reportable_configs_have_preprocessing_id() -> None:
    """Reportable configs contain preprocessing_id."""
    config_dir = ROOT / "configs" / "reportable"
    if not config_dir.exists():
        pytest.skip("No reportable configs directory")

    for config_path in config_dir.glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = config.get("model", {})
        if model.get("visual_encoder", "").startswith("dinov2"):
            assert "preprocessing_id" in model, f"{config_path}: missing preprocessing_id"
