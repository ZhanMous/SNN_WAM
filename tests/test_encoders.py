from __future__ import annotations

import pytest

pytest.importorskip("torch")
from src.models.encoders import (
    RealVisualEncoderPlaceholder,
    SmokeTimeIndexVisualEncoder,
    encode_sequence,
)


def test_smoke_time_index_encoder_is_frozen_and_deterministic() -> None:
    encoder = SmokeTimeIndexVisualEncoder(latent_dim=3)

    latents = encode_sequence(
        encoder,
        ["mock_train_0:frame:2", "mock_train_0:frame:5"],
    )

    assert latents[0] == pytest.approx([2.0, 2.01, 2.02])
    assert latents[1] == pytest.approx([5.0, 5.01, 5.02])
    assert encoder.metadata()["frozen"] is True
    assert encoder.metadata()["trainable_parameters"] == 0


def test_real_visual_encoder_placeholder_fails_closed() -> None:
    encoder = RealVisualEncoderPlaceholder(encoder_id="frozen_clip", latent_dim=8)

    with pytest.raises(NotImplementedError, match="placeholder"):
        encoder.encode(["frame_0"])
