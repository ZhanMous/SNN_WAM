"""Frozen visual encoder interfaces for Phase-1 latent targets.

Phase 1 uses frozen encoders only. The smoke encoder below is deterministic
and lightweight; real visual backbones must be supplied through the adapter or
a later explicit implementation without enabling gradient updates.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Mapping

import torch
from torch import nn


class FrozenVisualEncoder(nn.Module, ABC):
    """Base interface for frozen image-to-latent encoders.

    Shape contract:

    - input: encoder-specific image/reference batch.
    - output: `[B, latent_dim]` floating point tensor.

    Implementations must keep all parameters frozen. The trainer uses current
    latents as causal inputs and future latents as targets only.
    """

    def __init__(self, *, latent_dim: int, encoder_id: str) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not encoder_id:
            raise ValueError("encoder_id must be non-empty")
        self.latent_dim = latent_dim
        self.encoder_id = encoder_id

    def forward(self, images: Any) -> torch.Tensor:
        """Return frozen visual latents with shape `[B, latent_dim]`."""

        return self.encode(images)

    @abstractmethod
    def encode(self, images: Any) -> torch.Tensor:
        """Encode a batch of image observations or mock frame references."""

    def freeze(self) -> None:
        """Disable gradients and switch to eval mode."""

        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata for run artifacts."""

        return {
            "encoder_id": self.encoder_id,
            "latent_dim": self.latent_dim,
            "frozen": True,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }


class SmokeTimeIndexVisualEncoder(FrozenVisualEncoder):
    """Deterministic frozen encoder for synthetic smoke tests.

    The encoder extracts a time index from a numeric mock image, scalar time
    value, or frame-reference string such as `mock_train_0:frame:5`. It returns
    a simple latent vector `[t, t + offset, ...]` with shape `[B, latent_dim]`.
    This avoids generating pixels while still testing latent alignment.
    """

    def __init__(self, *, latent_dim: int = 8, offset: float = 0.01) -> None:
        super().__init__(latent_dim=latent_dim, encoder_id="smoke_time_index")
        self.offset = float(offset)
        self.freeze()

    def encode(self, images: Any) -> torch.Tensor:
        """Encode mock image/reference values into `[B, latent_dim]` latents."""

        time_values = self._time_values(images)
        offsets = torch.arange(self.latent_dim, dtype=torch.float32) * self.offset
        return time_values.unsqueeze(-1) + offsets.unsqueeze(0)

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata["offset"] = self.offset
        metadata["input"] = "mock_numeric_image_or_frame_reference"
        return metadata

    def _time_values(self, images: Any) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            tensor = images.detach().to(dtype=torch.float32)
            if tensor.ndim == 0:
                return tensor.reshape(1)
            if tensor.ndim == 1:
                return tensor
            if tensor.ndim >= 4:
                return tensor.reshape(tensor.shape[0], -1).mean(dim=1)
            return tensor.reshape(1, -1).mean(dim=1)

        if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
            return torch.tensor([extract_time_index(images)], dtype=torch.float32)

        return torch.tensor(
            [extract_time_index(item) for item in images],
            dtype=torch.float32,
        )


class FrozenVisualEncoderAdapter(FrozenVisualEncoder):
    """Adapter for externally supplied frozen visual encoder modules.

    The wrapped module is forced to eval mode and all parameters are frozen.
    This class does not download or instantiate large backbones; callers must
    provide the module and preprocessing function explicitly.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        latent_dim: int,
        encoder_id: str,
        preprocess: Callable[[Any], torch.Tensor] | None = None,
    ) -> None:
        super().__init__(latent_dim=latent_dim, encoder_id=encoder_id)
        self.module = module
        self.preprocess = preprocess
        self.freeze()

    def encode(self, images: Any) -> torch.Tensor:
        """Encode a tensor batch and return `[B, latent_dim]` detached latents."""

        inputs = self.preprocess(images) if self.preprocess is not None else images
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("FrozenVisualEncoderAdapter expects tensor inputs")
        if inputs.ndim < 2:
            raise ValueError(
                "adapter inputs must include batch and feature/image dimensions"
            )

        with torch.no_grad():
            latents = self.module(inputs)
        if not isinstance(latents, torch.Tensor):
            raise TypeError("wrapped encoder must return a torch.Tensor")
        if latents.ndim > 2:
            latents = latents.flatten(start_dim=1)
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(
                "wrapped encoder must return shape [B, latent_dim], "
                f"got {tuple(latents.shape)} for latent_dim={self.latent_dim}"
            )
        return latents.detach()


class RealVisualEncoderPlaceholder(FrozenVisualEncoder):
    """Fail-closed placeholder for real frozen visual backbones."""

    def __init__(self, *, encoder_id: str, latent_dim: int) -> None:
        super().__init__(latent_dim=latent_dim, encoder_id=encoder_id)
        self.freeze()

    def encode(self, images: Any) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.encoder_id} is a placeholder. Provide a frozen module via "
            "FrozenVisualEncoderAdapter or precompute latents with recorded metadata."
        )


def build_frozen_visual_encoder(config: Mapping[str, Any]) -> FrozenVisualEncoder:
    """Build a frozen visual encoder from the model config."""

    encoder_name = str(config.get("visual_encoder", "stub"))
    latent_dim = int(config.get("visual_latent_dim", 8))
    if encoder_name in {"smoke_time_index", "frozen_smoke", "stub"}:
        return SmokeTimeIndexVisualEncoder(latent_dim=latent_dim)
    if encoder_name in {"frozen_resnet", "frozen_clip", "clip", "resnet"}:
        return RealVisualEncoderPlaceholder(
            encoder_id=encoder_name,
            latent_dim=latent_dim,
        )
    raise ValueError(
        "unknown visual_encoder. Expected one of "
        "['smoke_time_index', 'frozen_smoke', 'stub', 'frozen_resnet', "
        "'frozen_clip', 'clip', 'resnet'], "
        f"got {encoder_name!r}"
    )


def encode_sequence(
    encoder: FrozenVisualEncoder,
    images: Sequence[Any],
) -> list[list[float]]:
    """Encode a trajectory image/reference sequence as `[T, latent_dim]` lists."""

    encoder.freeze()
    with torch.no_grad():
        latents = encoder.encode(images)
    if latents.ndim != 2 or latents.shape[0] != len(images):
        raise ValueError(
            "encoded latents must have shape [T, latent_dim], "
            f"got {tuple(latents.shape)} for T={len(images)}"
        )
    return latents.detach().cpu().to(dtype=torch.float32).tolist()


def extract_time_index(value: Any) -> float:
    """Extract a deterministic time index from a mock value."""

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        matches = re.findall(r"-?\d+(?:\.\d+)?", value)
        if not matches:
            raise ValueError(f"cannot extract time index from {value!r}")
        return float(matches[-1])
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            value = value.to(dtype=torch.float32)
        return float(value.detach().mean().item())
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError("cannot extract time index from an empty value")
    return float(tensor.mean().item())


__all__ = [
    "FrozenVisualEncoder",
    "FrozenVisualEncoderAdapter",
    "RealVisualEncoderPlaceholder",
    "SmokeTimeIndexVisualEncoder",
    "build_frozen_visual_encoder",
    "encode_sequence",
    "extract_time_index",
]
