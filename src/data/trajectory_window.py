"""Causal trajectory-window dataset for LIBERO-style demonstrations.

Raw trajectory arrays are expected to use time as the first axis:

- images: `[T, H, W, C]`
- actions: `[T, action_dim]`
- optional states: `[T, state_dim]`
- optional visual_latents: `[T, latent_dim]`
- optional frame references: `[T]`

Each returned sample is one unbatched time window. A downstream collate
function would add a batch axis, producing shapes such as
`image_t: [B, H, W, C]`, `action_history: [B, history_len, action_dim]`,
and `target_actions: [B, action_horizon, action_dim]`.

The default LIBERO processed-HDF5 convention is `action_to_current_obs`:
raw `actions[t]` is treated as the action that led to `images[t]`, so the
next action target after observing `images[t]` starts at `actions[t+1]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ACTION_TO_CURRENT_OBS = "action_to_current_obs"
ACTION_FROM_CURRENT_OBS = "action_from_current_obs"
VALID_ACTION_CONVENTIONS = {ACTION_TO_CURRENT_OBS, ACTION_FROM_CURRENT_OBS}
FORBIDDEN_INPUT_KEY_TOKENS = ("target", "future")


@dataclass(frozen=True)
class RawTrajectory:
    """One LIBERO-like trajectory with explicit `[T, ...]` arrays.

    Args:
        images: Raw image observations with shape `[T, H, W, C]`.
        actions: Robot actions with shape `[T, action_dim]`.
        language: Language instruction string for the trajectory.
        states: Optional state/proprio array with shape `[T, state_dim]`.
        visual_latents: Optional frozen visual latents with shape
            `[T, latent_dim]`.
        frame_refs: Optional frame references with shape `[T]`, such as
            HDF5 dataset paths or integer frame ids.
        trajectory_id: Stable id used only for sample metadata.
        split: Split label such as `train`, `val`, or `test`.
    """

    images: Sequence[Any]
    actions: Sequence[Any]
    language: str
    states: Sequence[Any] | None = None
    visual_latents: Sequence[Any] | None = None
    frame_refs: Sequence[Any] | None = None
    trajectory_id: str = "trajectory_0"
    split: str = "unspecified"

    @property
    def length(self) -> int:
        """Trajectory length `T` shared by `[T, ...]` fields."""

        return len(self.images)

    def validate(self) -> None:
        """Validate that all time-varying fields share the same `T` length."""

        if self.length <= 0:
            raise ValueError("images must contain at least one time step")
        if len(self.actions) != self.length:
            raise ValueError(
                f"actions length {len(self.actions)} does not match images length {self.length}"
            )
        if self.states is not None and len(self.states) != self.length:
            raise ValueError(
                f"states length {len(self.states)} does not match images length {self.length}"
            )
        if self.visual_latents is not None and len(self.visual_latents) != self.length:
            raise ValueError(
                "visual_latents length "
                f"{len(self.visual_latents)} does not match images length {self.length}"
            )
        if self.frame_refs is not None and len(self.frame_refs) != self.length:
            raise ValueError(
                f"frame_refs length {len(self.frame_refs)} does not match images length {self.length}"
            )


class TrajectoryWindowDataset:
    """Deterministic causal windows over LIBERO-style trajectories.

    For a current time index `t`, returned sample fields have these unbatched
    shapes under the default `action_to_current_obs` convention:

    - `image_t`: `[H, W, C]`, exactly `images[t]`.
    - `language`: `str`, trajectory-level instruction.
    - `action_history`: `[history_len, action_dim]`, exactly
      `actions[t-history_len+1:t+1]`; this includes the last executed action
      that led to `images[t]` and excludes future action targets.
    - `optional_state_t`: `[state_dim]` when states are present, otherwise
      `None`; this is exactly `states[t]`.
    - `z_t`: `[latent_dim]` when `include_current_latent=True`; this is
      exactly `visual_latents[t]` and is a causal current-observation input.
    - `target_actions`: `[action_horizon, action_dim]`, exactly
      `actions[t+1:t+1+action_horizon]`; action targets start after the
      current observation.
    - `target_future_latents`: `[future_horizon, latent_dim]` when
      `future_horizon > 0` and `include_future_latents=True`; these are
      targets only and start at `visual_latents[t+1]`.
    - `target_future_images`: `[future_horizon, H, W, C]` when
      `future_horizon > 0` and `include_future_images=True`; these are targets
      only and start at `images[t+1]`.
    - `target_future_frame_refs`: `[future_horizon]` when
      `future_horizon > 0` and `include_future_frame_refs=True`; these are
      targets only and start at frame `t+1`.

    No padding is performed in v1. Valid windows require enough past actions,
    target actions, and future image/reference targets, so edge windows that
    would need padding are excluded from the index.
    """

    def __init__(
        self,
        trajectories: Iterable[RawTrajectory | Mapping[str, Any]],
        *,
        history_len: int,
        action_horizon: int,
        future_horizon: int = 0,
        include_current_latent: bool = False,
        include_future_latents: bool = False,
        include_future_images: bool = False,
        include_future_frame_refs: bool = False,
        split: str | None = None,
        action_convention: str = ACTION_TO_CURRENT_OBS,
    ) -> None:
        if history_len <= 0:
            raise ValueError("history_len must be positive")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if future_horizon < 0:
            raise ValueError("future_horizon must be non-negative")
        if include_future_latents and future_horizon <= 0:
            raise ValueError("include_future_latents requires future_horizon > 0")
        if action_convention not in VALID_ACTION_CONVENTIONS:
            raise ValueError(
                f"action_convention must be one of {sorted(VALID_ACTION_CONVENTIONS)}"
            )

        self.history_len = history_len
        self.action_horizon = action_horizon
        self.future_horizon = future_horizon
        self.include_current_latent = include_current_latent
        self.include_future_latents = include_future_latents
        self.include_future_images = include_future_images
        self.include_future_frame_refs = include_future_frame_refs
        self.split = split
        self.action_convention = action_convention
        self.trajectories = [
            trajectory
            for trajectory in (
                coerce_trajectory(item, index) for index, item in enumerate(trajectories)
            )
            if split is None or trajectory.split == split
        ]
        for trajectory in self.trajectories:
            trajectory.validate()
            if (include_current_latent or include_future_latents) and (
                trajectory.visual_latents is None
            ):
                raise ValueError(
                    "visual_latents are required when latent fields are requested"
                )

        self._index: list[tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(self.trajectories):
            for t in valid_time_indices(
                trajectory.length,
                history_len=history_len,
                action_horizon=action_horizon,
                future_horizon=future_horizon,
                action_convention=action_convention,
            ):
                self._index.append((trajectory_index, t))

    def __len__(self) -> int:
        """Number of valid unpadded windows across all trajectories."""

        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one deterministic unbatched sample.

        The `input_keys` tuple contains only causal inputs. It never includes
        future images, future frame references, target actions, rewards, dones,
        success labels, or simulator outcome fields.
        """

        trajectory_index, t = self._index[index]
        trajectory = self.trajectories[trajectory_index]

        history_start, history_stop, target_start, action_end = action_index_ranges(
            t,
            history_len=self.history_len,
            action_horizon=self.action_horizon,
            action_convention=self.action_convention,
        )
        future_start = t + 1
        future_end = future_start + self.future_horizon

        input_keys = ["image_t", "language", "action_history"]
        optional_state_t = None
        if trajectory.states is not None:
            optional_state_t = trajectory.states[t]
            input_keys.append("optional_state_t")
        z_t = None
        if self.include_current_latent:
            if trajectory.visual_latents is None:
                raise RuntimeError("visual_latents unexpectedly missing")
            z_t = trajectory.visual_latents[t]
            input_keys.append("z_t")

        sample: dict[str, Any] = {
            "trajectory_index": trajectory_index,
            "trajectory_id": trajectory.trajectory_id,
            "split": trajectory.split,
            "time_index": t,
            "image_t": trajectory.images[t],
            "language": trajectory.language,
            "action_history": slice_sequence(trajectory.actions, history_start, history_stop),
            "optional_state_t": optional_state_t,
            "z_t": z_t,
            "target_actions": slice_sequence(trajectory.actions, target_start, action_end),
            "action_history_indices": list(range(history_start, history_stop)),
            "target_action_indices": list(range(target_start, action_end)),
            "input_keys": tuple(input_keys),
            "target_keys": ("target_actions",),
            "action_convention": self.action_convention,
        }

        target_keys = list(sample["target_keys"])
        if self.future_horizon > 0:
            sample["target_future_indices"] = list(range(future_start, future_end))
            target_keys.append("target_future_indices")
            if self.include_future_latents:
                if trajectory.visual_latents is None:
                    raise RuntimeError("visual_latents unexpectedly missing")
                sample["target_future_latents"] = slice_sequence(
                    trajectory.visual_latents,
                    future_start,
                    future_end,
                )
                target_keys.append("target_future_latents")
            if self.include_future_images:
                sample["target_future_images"] = slice_sequence(
                    trajectory.images,
                    future_start,
                    future_end,
                )
                target_keys.append("target_future_images")
            if self.include_future_frame_refs:
                refs = (
                    trajectory.frame_refs
                    if trajectory.frame_refs is not None
                    else list(range(trajectory.length))
                )
                sample["target_future_frame_refs"] = slice_sequence(
                    refs,
                    future_start,
                    future_end,
                )
                target_keys.append("target_future_frame_refs")

        sample["target_keys"] = tuple(target_keys)
        validate_causal_input_keys(sample["input_keys"])
        return sample

    def time_index_for_dataset_index(self, index: int) -> int:
        """Return current time `t` for a dataset index."""

        return self._index[index][1]

    def dataset_index_for_time(self, trajectory_index: int, time_index: int) -> int:
        """Return dataset index for a `(trajectory_index, t)` pair."""

        try:
            return self._index.index((trajectory_index, time_index))
        except ValueError as exc:
            raise KeyError(
                f"no valid window for trajectory_index={trajectory_index}, time_index={time_index}"
            ) from exc


def coerce_trajectory(item: RawTrajectory | Mapping[str, Any], index: int) -> RawTrajectory:
    """Convert a mapping with `[T, ...]` fields into `RawTrajectory`."""

    if isinstance(item, RawTrajectory):
        return item
    return RawTrajectory(
        images=item["images"],
        actions=item["actions"],
        language=item["language"],
        states=item.get("states"),
        visual_latents=item.get("visual_latents"),
        frame_refs=item.get("frame_refs"),
        trajectory_id=item.get("trajectory_id", f"trajectory_{index}"),
        split=item.get("split", "unspecified"),
    )


def valid_time_indices(
    length: int,
    *,
    history_len: int,
    action_horizon: int,
    future_horizon: int,
    action_convention: str = ACTION_TO_CURRENT_OBS,
) -> range:
    """Return valid current times for unpadded windows.

    Valid `t` values satisfy:

    - action history has shape `[history_len, action_dim]`.
    - target actions have shape `[action_horizon, action_dim]`.
    - If `future_horizon > 0`, `images[t+1:t+1+future_horizon]`
      has shape `[future_horizon, H, W, C]`.
    """

    if action_convention not in VALID_ACTION_CONVENTIONS:
        raise ValueError(
            f"action_convention must be one of {sorted(VALID_ACTION_CONVENTIONS)}"
        )
    earliest = history_len - 1 if action_convention == ACTION_TO_CURRENT_OBS else history_len
    latest_exclusive = length - action_horizon
    if action_convention == ACTION_FROM_CURRENT_OBS:
        latest_exclusive = length - action_horizon + 1
    if future_horizon > 0:
        latest_exclusive = min(latest_exclusive, length - future_horizon)
    latest_exclusive = max(earliest, latest_exclusive)
    return range(earliest, latest_exclusive)


def action_index_ranges(
    t: int,
    *,
    history_len: int,
    action_horizon: int,
    action_convention: str = ACTION_TO_CURRENT_OBS,
) -> tuple[int, int, int, int]:
    """Return action index ranges for history and targets.

    Returns `(history_start, history_stop, target_start, target_stop)`.
    For LIBERO processed HDF5 with `action_to_current_obs`, `actions[t]`
    is past context for `images[t]`, and target actions start at `t+1`.
    """

    if action_convention == ACTION_TO_CURRENT_OBS:
        history_stop = t + 1
        history_start = history_stop - history_len
        target_start = t + 1
    elif action_convention == ACTION_FROM_CURRENT_OBS:
        history_stop = t
        history_start = t - history_len
        target_start = t
    else:
        raise ValueError(
            f"action_convention must be one of {sorted(VALID_ACTION_CONVENTIONS)}"
        )
    return history_start, history_stop, target_start, target_start + action_horizon


def slice_sequence(values: Sequence[Any], start: int, stop: int) -> Any:
    """Slice a `[T, ...]` sequence without changing inner dimensions."""

    return values[start:stop]


def validate_causal_input_keys(input_keys: Sequence[str]) -> None:
    """Reject target or future fields from model inputs."""

    forbidden = [
        key
        for key in input_keys
        if any(token in key.lower() for token in FORBIDDEN_INPUT_KEY_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"future/target fields are forbidden as inputs: {forbidden}")


def make_mock_trajectory_dataset(
    *,
    length: int = 10,
    history_len: int = 4,
    action_horizon: int = 4,
    future_horizon: int = 4,
    image_shape: tuple[int, int, int] = (2, 2, 1),
    action_dim: int = 3,
    state_dim: int | None = 2,
    include_current_latent: bool = False,
    include_future_latents: bool = False,
    latent_dim: int = 4,
    include_future_images: bool = True,
    include_future_frame_refs: bool = True,
    split: str = "train",
) -> TrajectoryWindowDataset:
    """Create a deterministic synthetic dataset whose values encode time.

    Mock arrays use these shapes:

    - images: `[T, H, W, C]`, every scalar in `images[t]` equals `t`.
    - actions: `[T, action_dim]`, every scalar in `actions[t]` equals `t`.
    - states: `[T, state_dim]` when `state_dim` is not `None`, every scalar
      in `states[t]` equals `t`.
    - visual_latents: `[T, latent_dim]` when requested, with each scalar in
      `visual_latents[t]` equal to `t` plus a small dimension offset.
    - frame references: `[T]`, with integer reference `t`.
    """

    images = [filled_image(t, image_shape) for t in range(length)]
    actions = [[t for _ in range(action_dim)] for t in range(length)]
    states = None
    if state_dim is not None:
        states = [[t for _ in range(state_dim)] for t in range(length)]
    visual_latents = None
    if include_current_latent or include_future_latents:
        visual_latents = [
            [float(t) + 0.01 * float(dim) for dim in range(latent_dim)]
            for t in range(length)
        ]

    trajectory = RawTrajectory(
        images=images,
        actions=actions,
        states=states,
        visual_latents=visual_latents,
        frame_refs=list(range(length)),
        language="mock instruction",
        trajectory_id="mock_trajectory_0",
        split=split,
    )
    return TrajectoryWindowDataset(
        [trajectory],
        history_len=history_len,
        action_horizon=action_horizon,
        future_horizon=future_horizon,
        include_current_latent=include_current_latent,
        include_future_latents=include_future_latents,
        include_future_images=include_future_images,
        include_future_frame_refs=include_future_frame_refs,
        split=split,
    )


def filled_image(value: int, shape: tuple[int, int, int]) -> list[list[list[int]]]:
    """Return one `[H, W, C]` image whose scalars all equal `value`."""

    height, width, channels = shape
    return [
        [[value for _ in range(channels)] for _ in range(width)]
        for _ in range(height)
    ]


__all__ = [
    "RawTrajectory",
    "TrajectoryWindowDataset",
    "ACTION_FROM_CURRENT_OBS",
    "ACTION_TO_CURRENT_OBS",
    "action_index_ranges",
    "make_mock_trajectory_dataset",
    "validate_causal_input_keys",
    "valid_time_indices",
]
