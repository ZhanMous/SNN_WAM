"""Frozen visual encoder interfaces for Phase-1 latent targets.

Phase 1 uses frozen encoders only. The smoke encoder below is deterministic
and lightweight; real visual backbones must be supplied through the adapter or
a later explicit implementation without enabling gradient updates.

Supports two latent modes:

- CLS: ``[B, latent_dim]`` global token embedding.
- Patch: ``[B, N, D]`` spatial patch token embeddings.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

import torch
from torch import nn

# Pinned revision for facebook/dinov2-small (latest stable as of 2026-05-28)
# This ensures reproducible latent extraction across runs.
DEFAULT_DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"

# Preprocessing ID for manifest metadata
DINOV2_PREPROCESSING_ID = "AutoImageProcessor.from_pretrained"


def validate_revision(revision: str | None, model_id: str) -> str:
    """Validate and return a revision hash.

    Args:
        revision: Revision hash to validate, or None to use default.
        model_id: HuggingFace model ID for error messages.

    Returns:
        Validated 40-character hex revision string.

    Raises:
        ValueError: If revision is invalid format.
    """
    if revision is None:
        revision = DEFAULT_DINOV2_REVISION
    if not isinstance(revision, str) or not re.match(r"^[0-9a-f]{40}$", revision):
        raise ValueError(
            f"Invalid revision for {model_id}: {revision!r}. "
            "Must be a 40-character lowercase hex string."
        )
    return revision


def _load_dinov2_model(
    model_id: str,
    revision: str,
    *,
    class_name: str,
) -> tuple[Any, Any]:
    """Lazy-load a DINOv2 model and processor with local-first fallback.

    Returns ``(processor, model)`` tuple. The caller is responsible for
    freezing parameters and setting eval mode.
    """
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            f"transformers library is required for {class_name}. "
            "Install with: pip install transformers"
        ) from exc

    try:
        processor = AutoImageProcessor.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
    except OSError:
        processor = AutoImageProcessor.from_pretrained(
            model_id, revision=revision
        )
        model = AutoModel.from_pretrained(
            model_id, revision=revision
        )
    return processor, model


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


class DINOv2VisualEncoder(FrozenVisualEncoder):
    """DINOv2 frozen visual encoder for real LIBERO experiments.

    Uses the DINOv2 ViT-S/14 model from HuggingFace. The encoder is frozen
    and produces CLS token embeddings with shape [B, 384].

    This encoder requires the transformers library and DINOv2 model weights.
    It is intended for offline latent extraction, not real-time training.

    The revision is pinned to ensure reproducible latent extraction.
    If revision=None, uses DEFAULT_DINOV2_REVISION.
    """

    def __init__(
        self,
        *,
        model_id: str = "facebook/dinov2-small",
        revision: str | None = None,
        latent_dim: int = 384,
        output_token: str = "cls",
    ) -> None:
        # Validate and set revision before calling super().__init__
        revision = validate_revision(revision, model_id)

        super().__init__(latent_dim=latent_dim, encoder_id=f"dinov2_{model_id.split('/')[-1]}")
        self.model_id = model_id
        self.revision = revision
        self.output_token = output_token

        # Lazy loading - only load when actually used
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Lazy load the DINOv2 model and processor."""
        if self._model is not None:
            return

        self._processor, self._model = _load_dinov2_model(
            self.model_id, self.revision, class_name="DINOv2VisualEncoder"
        )
        self.freeze()

    def encode(self, images: Any) -> torch.Tensor:
        """Encode images using DINOv2 and return CLS embeddings."""
        self._load_model()

        if not isinstance(images, torch.Tensor):
            raise TypeError("DINOv2VisualEncoder expects tensor inputs")
        target_device = images.device
        if self._model is not None:
            self._model.to(target_device)

        # images should be [B, C, H, W] or [B, H, W, C]
        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Convert to PIL images for the processor
        from torchvision.transforms.functional import to_pil_image

        pil_images = [to_pil_image(img) for img in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(next(self._model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Extract CLS token or mean of patch tokens
        if self.output_token == "cls":
            latents = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            latents = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # Mean of patches

        return latents.detach()

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update({
            "model_id": self.model_id,
            "revision": self.revision,
            "output_token": self.output_token,
            "extraction_mode": "offline",
            "preprocessing_id": DINOV2_PREPROCESSING_ID,
        })
        return metadata


@dataclass(frozen=True)
class PatchLatentMetadata:
    """Metadata for patch latent cache files and datasets.

    This is a structured record of the encoder configuration used to produce
    patch latent tensors. It travels alongside cached ``.pt`` / ``.npz`` files
    and is stored in dataset metadata so that downstream consumers can reason
    about tensor shapes without inspecting the encoder.
    """

    encoder_name: str
    encoder_type: str
    image_size: int
    patch_size: int
    num_patches: int
    feature_dim: int
    include_cls: bool
    dtype: str
    normalization: str
    source_dataset: str = ""
    git_commit: str = ""
    revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PatchLatentMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DINOv2PatchEncoder(nn.Module):
    """DINOv2 frozen visual encoder that returns spatial patch tokens.

    Unlike :class:`DINOv2VisualEncoder` which returns only the CLS token
    with shape ``[B, D]``, this encoder returns all patch tokens with shape
    ``[B, N, D]`` where ``N`` is the number of spatial patches and ``D`` is
    the feature dimension.

    For DINOv2 ViT-S/14 on 224x224 images, ``N = (224/14)^2 = 256`` and
    ``D = 384``.

    Shape contract:

    - input: ``[B, C, H, W]`` or ``[B, H, W, C]`` float images.
    - output: ``[B, N, D]`` patch tokens (detached, no grad).
    - optional CLS: when ``return_cls=True`` the forward method returns a
      dict with ``"patch_tokens": [B, N, D]`` and ``"cls_token": [B, D]``.

    The encoder is frozen by default. No gradient updates are applied.

    Implementation note:
        This wrapper loads the HuggingFace DINOv2 model and uses
        ``forward_features()`` when available, otherwise falls back to
        ``__call__()`` and extracts ``last_hidden_state[:, 1:, :]`` for
        patch tokens. If neither path works, a clear error is raised.
    """

    def __init__(
        self,
        *,
        model_id: str = "facebook/dinov2-small",
        revision: str | None = None,
        image_size: int = 224,
        patch_size: int = 14,
        return_cls: bool = False,
    ) -> None:
        super().__init__()
        revision = validate_revision(revision, model_id)
        self.model_id = model_id
        self.revision = revision
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.return_cls = return_cls
        self._feature_dim: int | None = None
        self._model = None
        self._processor = None

    @property
    def feature_dim(self) -> int:
        """Feature dimension of patch tokens. Populated after first forward."""
        if self._feature_dim is None:
            raise RuntimeError(
                "feature_dim is not yet known; run a forward pass first "
                "or call _infer_feature_dim()"
            )
        return self._feature_dim

    def _load_model(self) -> None:
        """Lazy-load the DINOv2 model and image processor."""
        if self._model is not None:
            return

        self._processor, self._model = _load_dinov2_model(
            self.model_id, self.revision, class_name="DINOv2PatchEncoder"
        )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._infer_feature_dim()

    def _infer_feature_dim(self) -> None:
        """Infer the feature dimension from the loaded model."""
        if self._model is None:
            return
        try:
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            dummy = dummy.to(next(self._model.parameters()).device)
            with torch.no_grad():
                features = self._extract_patch_tokens(dummy)
            self._feature_dim = int(features.shape[-1])
        except Exception:
            # Fallback: infer from config if available
            cfg = getattr(self._model, "config", None)
            if cfg is not None and hasattr(cfg, "hidden_size"):
                self._feature_dim = cfg.hidden_size
            else:
                self._feature_dim = 384  # DINOv2-S default

    def _extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from raw images.

        Returns ``[B, N, D]`` tensor of patch tokens (excluding CLS).
        """
        if self._model is None:
            raise RuntimeError("model not loaded; call _load_model() first")

        # Try forward_features first (available in many DINOv2 implementations)
        if hasattr(self._model, "forward_features"):
            outputs = self._model.forward_features(images)
            if isinstance(outputs, dict) and "patch_tokens" in outputs:
                return outputs["patch_tokens"]
            if hasattr(outputs, "last_hidden_state"):
                # CLS is index 0, patches are 1:
                return outputs.last_hidden_state[:, 1:, :]

        # Fallback to standard forward
        outputs = self._model(images)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 1:, :]

        raise RuntimeError(
            "DINOv2PatchEncoder: unable to extract patch tokens from model "
            f"{self.model_id}. forward_features() and last_hidden_state are "
            "both unavailable."
        )

    def _extract_patch_tokens_from_outputs(self, outputs: Any) -> torch.Tensor:
        """Extract patch tokens from model outputs (post-forward)."""
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 1:, :]
        if isinstance(outputs, dict) and "patch_tokens" in outputs:
            return outputs["patch_tokens"]
        raise RuntimeError(
            f"DINOv2PatchEncoder: cannot extract patch tokens from outputs "
            f"of model {self.model_id}"
        )

    def forward(
        self, images: Any
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Encode images and return patch tokens.

        Args:
            images: ``[B, C, H, W]`` or ``[B, H, W, C]`` float tensor.

        Returns:
            When ``return_cls=False``: ``[B, N, D]`` patch token tensor.
            When ``return_cls=True``: dict with ``"patch_tokens": [B, N, D]``
            and ``"cls_token": [B, D]``.
        """
        self._load_model()
        if not isinstance(images, torch.Tensor):
            raise TypeError("DINOv2PatchEncoder expects tensor inputs")

        target_device = next(self._model.parameters()).device
        self._model.to(target_device)
        images = images.to(target_device)

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(
                f"images must have 3 or 4 dims, got {tuple(images.shape)}"
            )

        # Convert to PIL for the processor
        from torchvision.transforms.functional import to_pil_image

        pil_images = [to_pil_image(img.cpu()) for img in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(inputs["pixel_values"])

        # Extract both patch tokens and CLS from a single forward pass.
        if hasattr(outputs, "last_hidden_state"):
            cls_token = outputs.last_hidden_state[:, 0, :].detach()
            patch_tokens = outputs.last_hidden_state[:, 1:, :].detach()
        else:
            # Fallback: extract patches via forward_features path
            patch_tokens = self._extract_patch_tokens_from_outputs(outputs).detach()
            cls_token = None

        if not self.return_cls:
            return patch_tokens

        if cls_token is None:
            raise RuntimeError("CLS token unavailable from model outputs")
        return {
            "patch_tokens": patch_tokens,
            "cls_token": cls_token,
        }

    def metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata for cache files."""
        return {
            "encoder_name": self.model_id.split("/")[-1],
            "encoder_type": "dinov2_patch",
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_patches": self.num_patches,
            "feature_dim": self._feature_dim,
            "return_cls": self.return_cls,
            "model_id": self.model_id,
            "revision": self.revision,
            "frozen": True,
            "trainable_parameters": 0,
        }

    def patch_latent_metadata(
        self,
        *,
        source_dataset: str = "",
        git_commit: str = "",
    ) -> PatchLatentMetadata:
        """Return a structured PatchLatentMetadata record."""
        return PatchLatentMetadata(
            encoder_name=self.model_id.split("/")[-1],
            encoder_type="dinov2_patch",
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            feature_dim=self._feature_dim or 384,
            include_cls=self.return_cls,
            dtype="float16",
            normalization="dino_internal",
            source_dataset=source_dataset,
            git_commit=git_commit,
            revision=self.revision,
        )

    def freeze(self) -> None:
        """Disable gradients and switch to eval mode."""
        self.eval()
        if self._model is not None:
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)


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
    if encoder_name.startswith("dinov2_patch"):
        model_id = config.get("model_id", "facebook/dinov2-small")
        revision = config.get("revision")
        image_size = int(config.get("image_size", 224))
        patch_size = int(config.get("patch_size", 14))
        return_cls = bool(config.get("return_cls", False))
        # DINOv2PatchEncoder is not a FrozenVisualEncoder subclass, so we
        # do not route it through build_frozen_visual_encoder.  Callers who
        # want patch latents should instantiate DINOv2PatchEncoder directly.
        raise ValueError(
            "DINOv2PatchEncoder cannot be built via build_frozen_visual_encoder. "
            "Instantiate DINOv2PatchEncoder directly."
        )
    if encoder_name.startswith("dinov2_"):
        model_id = config.get("model_id", "facebook/dinov2-small")
        revision = config.get("revision")
        output_token = config.get("output_token", "cls")
        return DINOv2VisualEncoder(
            model_id=model_id,
            revision=revision,
            latent_dim=latent_dim,
            output_token=output_token,
        )
    raise ValueError(
        "unknown visual_encoder. Expected one of "
        "['smoke_time_index', 'frozen_smoke', 'stub', 'frozen_resnet', "
        "'frozen_clip', 'clip', 'resnet', 'dinov2_*', 'dinov2_patch_*'], "
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
    "DEFAULT_DINOV2_REVISION",
    "DINOV2_PREPROCESSING_ID",
    "DINOv2PatchEncoder",
    "FrozenVisualEncoder",
    "FrozenVisualEncoderAdapter",
    "PatchLatentMetadata",
    "RealVisualEncoderPlaceholder",
    "SmokeTimeIndexVisualEncoder",
    "build_frozen_visual_encoder",
    "encode_sequence",
    "extract_time_index",
    "validate_revision",
]
