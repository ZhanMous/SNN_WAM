from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src.eval.open_loop_diagnostics import (
    last_action_like,
    mean_action_like,
    run_open_loop_diagnostics,
)
from src.train.train_offline import run_training


ROOT = Path(__file__).resolve().parents[1]


def test_action_baseline_shapes_and_values() -> None:
    target = torch.zeros(2, 3, 2)
    mean = torch.tensor([1.5, -2.0])
    history = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 2.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )

    mean_pred = mean_action_like(target, mean)
    last_pred = last_action_like(target, history)

    assert tuple(mean_pred.shape) == (2, 3, 2)
    assert mean_pred[0, 0].tolist() == pytest.approx([1.5, -2.0])
    assert torch.allclose(last_pred[0], torch.tensor([[1.0, 2.0]] * 3))
    assert torch.allclose(last_pred[1], torch.tensor([[5.0, 6.0]] * 3))


def test_open_loop_diagnostics_writes_metrics_and_trace(tmp_path: Path) -> None:
    run_dir = run_training(
        ROOT / "configs/smoke/libero_spatial_wam_gru.yaml",
        dry_run=True,
        max_steps=1,
        output_dir=tmp_path / "runs",
        run_id="open_loop_source",
        command=["python3", "src/train/train_offline.py", "--dry_run"],
    )

    output_dir = run_open_loop_diagnostics(
        run_dir=run_dir,
        split="val",
        max_batches=1,
        trace_limit=8,
        device_name="cpu",
        command=["python3", "-m", "src.eval.open_loop_diagnostics"],
    )

    metrics_path = output_dir / "open_loop_metrics.csv"
    trace_path = output_dir / "action_trace_diagnostics.csv"
    summary_path = output_dir / "summary.json"

    assert metrics_path.exists()
    assert trace_path.exists()
    assert summary_path.exists()
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_commit.txt").exists()
    assert (output_dir / "notes.md").exists()
    assert (output_dir / "diagnostic_summary.md").exists()

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["baseline"] for row in rows} == {
        "model",
        "zero_action",
        "random_action_train_gaussian",
        "mean_action_train",
        "last_action",
    }
    for row in rows:
        assert float(row["action_mse"]) >= 0.0
        assert len(json.loads(row["action_mse_by_horizon"])) == 4
        assert len(json.loads(row["action_mse_by_dimension"])) == 7

    with trace_path.open(newline="", encoding="utf-8") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert trace_rows
    assert set(trace_rows[0]) >= {
        "expert_action",
        "pred_action",
        "expert_action_norm",
        "pred_action_norm",
        "expert_gripper",
        "pred_gripper",
        "cosine_similarity",
        "per_dimension_error",
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "teacher_forced_open_loop_diagnostic"
    assert "not_closed_loop_success" in summary["non_claims"]
