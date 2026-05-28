"""Split and normalization helpers for dataset policy tests.

These helpers operate on in-memory `RawTrajectory` objects. They do not load
LIBERO files, run training, or create model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

from src.data.trajectory_window import RawTrajectory


@dataclass(frozen=True)
class FieldStats:
    """Per-dimension standardization statistics for `[N, D]` rows."""

    mean: list[float]
    std: list[float]
    count: int
    source_split: str


def select_trajectories_by_split(
    trajectories: Iterable[RawTrajectory],
    split: str,
) -> list[RawTrajectory]:
    """Return only trajectories whose split label equals `split`."""

    return [trajectory for trajectory in trajectories if trajectory.split == split]


def fit_train_only_standardization_stats(
    trajectories: Iterable[RawTrajectory],
    *,
    split: str = "train",
    fields: Sequence[str] = ("actions", "states"),
) -> dict[str, FieldStats]:
    """Fit mean/std only on trajectories from the requested train split.

    Args:
        trajectories: Raw trajectories with `[T, ...]` arrays.
        split: Split label to fit on. Phase-1 policy uses `train`.
        fields: Supported fields are `actions` and `states`.

    Returns:
        A mapping from field name to `FieldStats`.
    """

    selected = select_trajectories_by_split(trajectories, split)
    if not selected:
        raise ValueError(f"no trajectories found for split={split!r}")

    stats: dict[str, FieldStats] = {}
    for field in fields:
        rows: list[list[float]] = []
        for trajectory in selected:
            values = getattr(trajectory, field)
            if values is None:
                continue
            rows.extend(as_float_rows(values))
        if not rows:
            continue
        stats[field] = compute_field_stats(rows, split)
    return stats


def as_float_rows(values: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert a `[T, D]` sequence to float rows without flattening time."""

    rows = [[float(item) for item in row] for row in values]
    if not rows:
        return rows
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all rows must have the same feature dimension")
    return rows


def compute_field_stats(rows: list[list[float]], split: str) -> FieldStats:
    """Compute per-dimension mean/std for `[N, D]` rows.

    Uses population variance (divides by N, not N-1), which is standard for
    ML normalization statistics where the rows are the full train population.
    """

    count = len(rows)
    width = len(rows[0])
    mean = [sum(row[dim] for row in rows) / count for dim in range(width)]
    variances = [
        sum((row[dim] - mean[dim]) ** 2 for row in rows) / count
        for dim in range(width)
    ]
    std = [sqrt(value) if value > 0 else 1.0 for value in variances]
    return FieldStats(mean=mean, std=std, count=count, source_split=split)


__all__ = [
    "FieldStats",
    "fit_train_only_standardization_stats",
    "select_trajectories_by_split",
]
