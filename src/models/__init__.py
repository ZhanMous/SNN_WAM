"""Model components for SNN-WAM baselines."""

from src.models.encoders import (
    FrozenVisualEncoder,
    FrozenVisualEncoderAdapter,
    RealVisualEncoderPlaceholder,
    SmokeTimeIndexVisualEncoder,
    build_frozen_visual_encoder,
    encode_sequence,
)
from src.models.heads import ActionChunkHead, FutureLatentChunkHead
from src.models.registry import build_action_model, build_offline_model, count_parameters
from src.models.temporal_gru import TemporalGRU, TemporalGRUActionModel, TemporalGRUWAMModel
from src.models.temporal_mlp import TemporalMLP, TemporalMLPActionModel

__all__ = [
    "ActionChunkHead",
    "FutureLatentChunkHead",
    "FrozenVisualEncoder",
    "FrozenVisualEncoderAdapter",
    "RealVisualEncoderPlaceholder",
    "SmokeTimeIndexVisualEncoder",
    "TemporalGRU",
    "TemporalGRUActionModel",
    "TemporalGRUWAMModel",
    "TemporalMLP",
    "TemporalMLPActionModel",
    "build_action_model",
    "build_frozen_visual_encoder",
    "build_offline_model",
    "count_parameters",
    "encode_sequence",
]
