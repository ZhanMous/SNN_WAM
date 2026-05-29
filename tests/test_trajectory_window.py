from __future__ import annotations

import pytest

from src.data.trajectory_window import (
    RawTrajectory,
    TrajectoryWindowDataset,
    make_mock_trajectory_dataset,
    validate_causal_input_keys,
    valid_time_indices,
)


def nested_shape(value: object) -> list[int]:
    shape: list[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        current = current[0] if current else None
    return shape


def image_time_value(image: list[list[list[int]]]) -> int:
    return image[0][0][0]


def row_time_values(rows: list[list[int]]) -> list[int]:
    return [row[0] for row in rows]


def latent_time_values(rows: list[list[float]]) -> list[int]:
    return [int(row[0]) for row in rows]


def test_mock_dataset_import_smoke() -> None:
    dataset = make_mock_trajectory_dataset()
    assert len(dataset) == 3


def test_valid_time_indices_drop_edges_without_padding() -> None:
    assert list(
        valid_time_indices(
            length=10,
            history_len=4,
            action_horizon=4,
            future_horizon=4,
        )
    ) == [3, 4, 5]


def test_time_indexed_mock_sample_shapes_and_causal_alignment() -> None:
    dataset = make_mock_trajectory_dataset(
        length=10,
        history_len=4,
        action_horizon=4,
        future_horizon=4,
        image_shape=(2, 2, 1),
        action_dim=3,
        state_dim=2,
    )
    sample = dataset[dataset.dataset_index_for_time(0, 5)]

    assert sample["time_index"] == 5
    assert nested_shape(sample["image_t"]) == [2, 2, 1]
    assert nested_shape(sample["action_history"]) == [4, 3]
    assert nested_shape(sample["optional_state_t"]) == [2]
    assert nested_shape(sample["target_actions"]) == [4, 3]
    assert nested_shape(sample["target_future_images"]) == [4, 2, 2, 1]
    assert nested_shape(sample["target_future_frame_refs"]) == [4]

    assert image_time_value(sample["image_t"]) == 5
    assert row_time_values(sample["action_history"]) == [2, 3, 4, 5]
    assert 6 not in row_time_values(sample["action_history"])
    assert sample["action_history_indices"] == [2, 3, 4, 5]

    assert row_time_values(sample["target_actions"]) == [6, 7, 8, 9]
    assert sample["target_action_indices"] == [6, 7, 8, 9]
    assert [image_time_value(image) for image in sample["target_future_images"]] == [
        6,
        7,
        8,
        9,
    ]
    assert sample["target_future_indices"] == [6, 7, 8, 9]
    assert sample["target_future_frame_refs"] == [6, 7, 8, 9]


def test_future_targets_are_not_input_keys() -> None:
    dataset = make_mock_trajectory_dataset()
    sample = dataset[dataset.dataset_index_for_time(0, 5)]

    input_keys = set(sample["input_keys"])
    target_keys = set(sample["target_keys"])
    assert input_keys == {"image_t", "language", "action_history", "optional_state_t"}
    assert target_keys == {
        "target_actions",
        "target_future_indices",
        "target_future_images",
        "target_future_frame_refs",
    }
    assert input_keys.isdisjoint(target_keys)
    assert not any("target" in key or "future" in key for key in input_keys)
    assert "target_future_images" in sample
    assert "target_future_images" not in input_keys
    assert "target_actions" not in input_keys
    assert "rewards" not in sample
    assert "dones" not in sample


def test_future_latent_targets_align_after_current_time_and_do_not_leak() -> None:
    dataset = make_mock_trajectory_dataset(
        length=10,
        history_len=4,
        action_horizon=4,
        future_horizon=4,
        include_current_latent=True,
        include_future_latents=True,
        latent_dim=3,
        include_future_images=False,
        include_future_frame_refs=False,
    )
    sample = dataset[dataset.dataset_index_for_time(0, 5)]

    assert sample["time_index"] == 5
    assert sample["z_t"] == pytest.approx([5.0, 5.01, 5.02])
    assert nested_shape(sample["target_future_latents"]) == [4, 3]
    assert latent_time_values(sample["target_future_latents"]) == [6, 7, 8, 9]
    assert sample["target_future_indices"] == [6, 7, 8, 9]

    assert "z_t" in sample["input_keys"]
    assert "target_future_latents" not in sample["input_keys"]
    assert "target_future_latents" in sample["target_keys"]
    assert not any("future" in key or "target" in key for key in sample["input_keys"])


def test_future_latent_placeholders_are_forbidden_as_inputs() -> None:
    with pytest.raises(ValueError, match="future/target fields"):
        validate_causal_input_keys(("image_t", "target_future_latents"))

    with pytest.raises(ValueError, match="future/target fields"):
        validate_causal_input_keys(("image_t", "future_latent_placeholder"))


def test_optional_state_t_is_current_time_not_future_state() -> None:
    dataset = make_mock_trajectory_dataset(
        length=10,
        history_len=4,
        action_horizon=3,
        future_horizon=2,
        state_dim=2,
    )
    sample = dataset[dataset.dataset_index_for_time(0, 5)]

    assert sample["optional_state_t"] == [5, 5]
    assert sample["optional_state_t"][0] == sample["time_index"]
    assert 6 not in sample["optional_state_t"]


def test_sample_exposes_causal_state_and_task_conditioning() -> None:
    trajectory = RawTrajectory(
        images=["obs0", "obs1", "obs2", "obs3", "obs4", "obs5"],
        actions=[[float(t)] for t in range(6)],
        states=[[float(t), float(t) + 0.5] for t in range(6)],
        visual_latents=[[float(t)] for t in range(6)],
        language="pick object",
        task_id=3,
        task_name="pick_object",
        trajectory_id="traj_0",
        split="train",
    )
    dataset = TrajectoryWindowDataset(
        [trajectory],
        history_len=2,
        action_horizon=2,
        future_horizon=1,
        include_current_latent=True,
        include_future_latents=True,
        split="train",
    )

    sample = dataset[0]

    assert sample["optional_state_t"] == [1.0, 1.5]
    assert sample["task_id"] == 3
    assert sample["task_name"] == "pick_object"
    assert "optional_state_t" in sample["input_keys"]
    assert "task_id" in sample["input_keys"]


def test_optional_state_is_none_and_not_an_input_key_when_absent() -> None:
    dataset = make_mock_trajectory_dataset(state_dim=None)
    sample = dataset[0]

    assert sample["optional_state_t"] is None
    assert "optional_state_t" not in sample["input_keys"]


def test_dataset_returns_deterministic_samples_in_mock_mode() -> None:
    dataset_a = make_mock_trajectory_dataset()
    dataset_b = make_mock_trajectory_dataset()

    assert dataset_a[0] == dataset_a[0]
    assert dataset_a[0] == dataset_b[0]
    assert dataset_a[1] == dataset_b[1]


def test_multiple_trajectories_do_not_share_history_across_boundaries() -> None:
    trajectory_a = RawTrajectory(
        images=[[[[100 + t]] for _ in range(1)] for t in range(8)],
        actions=[[100 + t] for t in range(8)],
        states=[[100 + t] for t in range(8)],
        frame_refs=[f"a/{t}" for t in range(8)],
        language="trajectory a",
        trajectory_id="a",
        split="train",
    )
    trajectory_b = RawTrajectory(
        images=[[[[200 + t]] for _ in range(1)] for t in range(8)],
        actions=[[200 + t] for t in range(8)],
        states=[[200 + t] for t in range(8)],
        frame_refs=[f"b/{t}" for t in range(8)],
        language="trajectory b",
        trajectory_id="b",
        split="val",
    )
    dataset = TrajectoryWindowDataset(
        [trajectory_a, trajectory_b],
        history_len=3,
        action_horizon=2,
        future_horizon=2,
        include_future_frame_refs=True,
    )
    sample = dataset[dataset.dataset_index_for_time(1, 3)]

    assert sample["trajectory_id"] == "b"
    assert sample["split"] == "val"
    assert sample["action_history"] == [[201], [202], [203]]
    assert sample["optional_state_t"] == [203]
    assert sample["target_actions"] == [[204], [205]]
    assert sample["target_future_frame_refs"] == ["b/4", "b/5"]
    assert all(action[0] >= 200 for action in sample["action_history"])


def test_trajectory_length_validation_rejects_misaligned_time_axes() -> None:
    with pytest.raises(ValueError, match="actions length"):
        TrajectoryWindowDataset(
            [
                RawTrajectory(
                    images=[[[[0]]], [[[1]]]],
                    actions=[[0]],
                    language="bad trajectory",
                )
            ],
            history_len=1,
            action_horizon=1,
        )
