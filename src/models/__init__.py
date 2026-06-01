"""Model components for SNN-WAM baselines."""

from src.models.encoders import (
    DINOv2PatchEncoder,
    FrozenVisualEncoder,
    FrozenVisualEncoderAdapter,
    PatchLatentMetadata,
    RealVisualEncoderPlaceholder,
    SmokeTimeIndexVisualEncoder,
    build_frozen_visual_encoder,
    encode_sequence,
)
from src.models.heads import (
    ActionChunkHead,
    FutureLatentChunkHead,
    SplitActionGripperHead,
    gripper_logits_to_command,
)
from src.models.registry import build_action_model, build_offline_model, count_parameters
from src.models.temporal_gru import (
    LatentProprioTaskGRUActionModel,
    TemporalGRU,
    TemporalGRUActionModel,
    TemporalGRUWAMModel,
)
from src.models.temporal_mlp import TemporalMLP, TemporalMLPActionModel

__all__ = [
    "ActionChunkHead",
    "DINOv2PatchEncoder",
    "FutureLatentChunkHead",
    "FrozenVisualEncoder",
    "FrozenVisualEncoderAdapter",
    "LatentProprioTaskGRUActionModel",
    "PatchLatentMetadata",
    "RealVisualEncoderPlaceholder",
    "SmokeTimeIndexVisualEncoder",
    "SplitActionGripperHead",
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
    "gripper_logits_to_command",
]
