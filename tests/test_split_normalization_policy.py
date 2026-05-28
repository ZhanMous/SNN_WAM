from __future__ import annotations

from src.data.split_normalization import (
    fit_train_only_standardization_stats,
    select_trajectories_by_split,
)
from src.data.trajectory_window import RawTrajectory, TrajectoryWindowDataset


def make_policy_trajectory(
    trajectory_id: str,
    split: str,
    base: int,
) -> RawTrajectory:
    return RawTrajectory(
        images=[[[[base + t]]] for t in range(6)],
        actions=[[base + t, base + t + 10] for t in range(6)],
        states=[[base + t + 100] for t in range(6)],
        frame_refs=[f"{trajectory_id}/{t}" for t in range(6)],
        language=f"{split} trajectory",
        trajectory_id=trajectory_id,
        split=split,
    )


def test_train_only_normalization_stats_ignore_val_and_test_data() -> None:
    train = make_policy_trajectory("train_0", "train", 0)
    val = make_policy_trajectory("val_0", "val", 1000)
    test = make_policy_trajectory("test_0", "test", 2000)

    stats = fit_train_only_standardization_stats([train, val, test])

    assert stats["actions"].source_split == "train"
    assert stats["actions"].count == 6
    assert stats["actions"].mean == [2.5, 12.5]
    assert stats["states"].mean == [102.5]


def test_select_trajectories_by_split_returns_only_requested_split() -> None:
    train = make_policy_trajectory("train_0", "train", 0)
    val = make_policy_trajectory("val_0", "val", 1000)

    selected = select_trajectories_by_split([train, val], "train")

    assert [trajectory.trajectory_id for trajectory in selected] == ["train_0"]


def test_trajectory_window_split_filter_never_crosses_split_boundaries() -> None:
    train = make_policy_trajectory("train_0", "train", 0)
    val = make_policy_trajectory("val_0", "val", 1000)

    dataset = TrajectoryWindowDataset(
        [train, val],
        split="train",
        history_len=2,
        action_horizon=2,
        future_horizon=1,
        include_future_frame_refs=True,
    )

    assert len(dataset) > 0
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["split"] == "train"
        assert sample["trajectory_id"] == "train_0"
        assert all(action[0] < 1000 for action in sample["action_history"])
        assert all(action[0] < 1000 for action in sample["target_actions"])
        assert all(ref.startswith("train_0/") for ref in sample["target_future_frame_refs"])
