from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from src.data.trajectory_window import RawTrajectory
from src.eval.overfit_diagnostics import (
    ShiftedTargetWindowDataset,
    _repair_metrics_from_tensors,
)


def _make_repair_config() -> dict:
    return {
        "data": {
            "history_len": 3,
            "future_horizon": 1,
        },
        "model": {"temporal_adapter": "wam_gru"},
        "training": {"lambda_future": 0.0},
    }


def _make_repair_trajectory() -> RawTrajectory:
    actions = [[float(t), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if t % 2 == 0 else -1.0] for t in range(8)]
    visual_latents = [[float(t), float(t + 1)] for t in range(8)]
    return RawTrajectory(
        images=[f"frame_{t}" for t in range(8)],
        actions=actions,
        visual_latents=visual_latents,
        language="test instruction",
        trajectory_id="traj_0",
        split="train",
    )


def test_shifted_target_window_dataset_uses_explicit_h1_target_shift() -> None:
    trajectory = _make_repair_trajectory()
    config = _make_repair_config()

    rows = {}
    for shift in [-1, 0, 2]:
        dataset = ShiftedTargetWindowDataset(
            [trajectory],
            split="train",
            config=config,
            action_horizon=1,
            target_shift=shift,
        )
        rows[shift] = next(
            sample for sample in (dataset[i] for i in range(len(dataset)))
            if sample["time_index"] == 2
        )

    assert rows[-1]["action_history_indices"] == [0, 1, 2]
    assert rows[-1]["target_action_indices"] == [2]
    assert rows[0]["target_action_indices"] == [3]
    assert rows[2]["target_action_indices"] == [5]
    assert rows[0]["target_keys"] == ("target_actions",)
    assert rows[0]["target_future_indices"] == [3]


def test_repair_metrics_report_continuous_and_gripper_separately() -> None:
    target = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]])
    pred = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]])

    metrics = _repair_metrics_from_tensors(pred, target)

    assert metrics["continuous_mse"] == pytest.approx(1.0 / 6.0)
    assert metrics["gripper_mse"] == 4.0
    assert metrics["gripper_sign_accuracy"] == 0.0
