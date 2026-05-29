#!/usr/bin/env python3
"""Single-demonstration teacher-forced overfit diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.trajectory_window import TrajectoryWindowDataset  # noqa: E402
from src.eval.open_loop_diagnostics import (  # noqa: E402
    TRACE_FIELDNAMES,
    build_trace_rows,
    maybe_denormalize,
)
from src.models.registry import build_offline_model, count_parameters  # noqa: E402
from src.train.train_offline import (  # noqa: E402
    METRIC_FIELDNAMES,
    apply_action_transform,
    build_action_transform,
    capture_environment_json,
    checkpoint_payload,
    collate_action_batch,
    format_metric_row,
    forward_offline_model,
    has_current_latent,
    has_future_latent_targets,
    infer_action_dim,
    infer_current_latent_dim,
    infer_latent_dim,
    infer_state_dim,
    infer_task_count,
    load_real_libero_trajectories,
    run_one_split,
    uses_current_latent,
    requires_future_latents,
    validate_training_scope,
    write_json,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.experiment_io import capture_environment, capture_git_commit  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/diagnostics/single_demo_overfit"))
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--trajectory_id", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "any"], default="train")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--loss_threshold", type=float, default=1e-4)
    parser.add_argument("--trace_limit", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = run_single_demo_overfit(
        config_path=args.config,
        output_root=args.output_dir,
        run_id=args.run_id,
        trajectory_id=args.trajectory_id,
        source_split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        loss_threshold=args.loss_threshold,
        trace_limit=args.trace_limit,
        device_name=args.device,
        seed=args.seed,
        command=[sys.executable, "-m", "src.eval.single_demo_overfit", *(argv or sys.argv[1:])],
    )
    print(f"single_demo_overfit_dir={output_dir}")
    return 0


def run_single_demo_overfit(
    *,
    config_path: Path,
    output_root: Path,
    run_id: str | None = None,
    trajectory_id: str | None = None,
    source_split: str = "train",
    epochs: int = 300,
    batch_size: int = 64,
    lr: float | None = None,
    max_steps: int | None = None,
    loss_threshold: float = 1e-4,
    trace_limit: int = 512,
    device_name: str = "cpu",
    seed: int = 0,
    command: Sequence[str] | None = None,
) -> Path:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if trace_limit < 0:
        raise ValueError("trace_limit must be non-negative")

    config = load_config(config_path)
    validate_training_scope(config)
    seed_everything(seed)

    trajectories, source_metadata = load_real_libero_trajectories(config)
    selected = select_trajectory(
        trajectories,
        trajectory_id=trajectory_id,
        source_split=source_split,
    )
    train_trajectory = replace(selected, split="train")
    val_trajectory = replace(selected, split="val")
    diagnostic_trajectories = [train_trajectory, val_trajectory]

    action_transform, normalization_stats = build_action_transform(
        diagnostic_trajectories,
        config,
    )
    if action_transform is not None:
        diagnostic_trajectories = apply_action_transform(
            diagnostic_trajectories,
            action_transform,
        )

    train_dataset = build_single_demo_dataset(
        diagnostic_trajectories,
        config,
        split="train",
    )
    val_dataset = build_single_demo_dataset(
        diagnostic_trajectories,
        config,
        split="val",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_action_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_action_batch,
    )

    sample = train_dataset[0]
    action_dim = infer_action_dim(sample)
    latent_dim = None
    if has_future_latent_targets(sample):
        latent_dim = infer_latent_dim(sample)
    elif has_current_latent(sample):
        latent_dim = infer_current_latent_dim(sample)
    split_metadata = {
        "method": "single_demo_duplicate_train_val",
        "source_split_metadata": source_metadata,
        "source_trajectory_id": selected.trajectory_id,
        "train": [train_trajectory.trajectory_id],
        "val": [val_trajectory.trajectory_id],
        "test": [],
        "evaluation_split": "same_demo_teacher_forced",
        "task_id": selected.task_id,
        "task_name": selected.task_name,
        "task_id_map": source_metadata.get("task_id_map", {}),
    }
    model = build_offline_model(
        config,
        action_dim=action_dim,
        latent_dim=latent_dim,
        state_dim=infer_state_dim(sample),
        num_tasks=infer_task_count(config, split_metadata),
    )
    parameter_counts = count_parameters(model)
    device = torch.device(device_name)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr if lr is not None else config["training"]["lr"]),
    )
    lambda_action = float(config["training"]["lambda_action"])
    lambda_future = float(config["training"]["lambda_future"])
    grad_clip_norm = config["training"].get("grad_clip_norm")

    run_id = run_id or default_run_id(config)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.csv"
    rows: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = -1
    last_checkpoint: dict[str, Any] | None = None
    passed = False

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        for epoch in range(epochs):
            train_metrics = run_one_split(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                lambda_action=lambda_action,
                lambda_future=lambda_future,
                grad_clip_norm=grad_clip_norm,
                max_steps=max_steps,
                action_transform=action_transform,
            )
            val_metrics = run_one_split(
                model,
                val_loader,
                device=device,
                optimizer=None,
                lambda_action=lambda_action,
                lambda_future=lambda_future,
                grad_clip_norm=None,
                max_steps=max_steps,
                action_transform=action_transform,
            )
            train_row = format_metric_row(
                epoch,
                "train_single_demo",
                train_metrics,
                parameter_counts=parameter_counts,
            )
            val_row = format_metric_row(
                epoch,
                "eval_same_demo_teacher_forced",
                val_metrics,
                parameter_counts=parameter_counts,
            )
            writer.writerow(train_row)
            writer.writerow(val_row)
            handle.flush()
            rows.extend([train_row, val_row])

            current_metric = float(val_metrics["action_mse"])
            last_checkpoint = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                best_metric=min(best_metric, current_metric),
                best_epoch=best_epoch,
                metrics={"train": train_metrics, "val": val_metrics},
            )
            torch.save(last_checkpoint, output_dir / "checkpoint.pt")
            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch
                last_checkpoint["best_metric"] = best_metric
                last_checkpoint["best_epoch"] = best_epoch
                torch.save(last_checkpoint, output_dir / "best.pt")

            if float(train_metrics["action_mse"]) <= loss_threshold:
                passed = True
                break

    if last_checkpoint is None:
        raise RuntimeError("single-demo overfit produced no checkpoint")
    if not (output_dir / "best.pt").exists():
        torch.save(last_checkpoint, output_dir / "best.pt")

    trace_path = output_dir / "action_trace_diagnostics.csv"
    write_final_trace(
        model=model,
        loader=val_loader,
        device=device,
        action_transform=action_transform,
        run_id=run_id,
        model_name=str(config["model"].get("name", config["model"]["temporal_adapter"])),
        trace_limit=trace_limit,
        output_path=trace_path,
    )
    write_repro_files(
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        command=command,
        split_metadata=split_metadata,
        normalization_stats=normalization_stats,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        loss_threshold=loss_threshold,
        best_metric=best_metric,
        best_epoch=best_epoch,
        passed=passed,
        metrics_path=metrics_path,
        trace_path=trace_path,
        rows=rows,
    )
    return output_dir


def select_trajectory(
    trajectories: Sequence[Any],
    *,
    trajectory_id: str | None,
    source_split: str,
) -> Any:
    candidates = list(trajectories)
    if trajectory_id is not None:
        for trajectory in candidates:
            if trajectory.trajectory_id == trajectory_id:
                return trajectory
        raise ValueError(f"trajectory_id not found: {trajectory_id}")
    if source_split != "any":
        candidates = [trajectory for trajectory in candidates if trajectory.split == source_split]
    if not candidates:
        raise ValueError(f"no trajectories available for split={source_split!r}")
    return sorted(candidates, key=lambda item: item.trajectory_id)[0]


def build_single_demo_dataset(
    trajectories: Sequence[Any],
    config: Mapping[str, Any],
    *,
    split: str,
) -> TrajectoryWindowDataset:
    return TrajectoryWindowDataset(
        trajectories,
        split=split,
        history_len=int(config["data"]["history_len"]),
        action_horizon=int(config["data"]["action_horizon"]),
        future_horizon=int(config["data"]["future_horizon"]),
        include_current_latent=uses_current_latent(config),
        include_future_latents=requires_future_latents(config),
    )


def write_final_trace(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_transform: Any,
    run_id: str,
    model_name: str,
    trace_limit: int,
    output_path: Path,
) -> None:
    rows: list[dict[str, str]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(rows) >= trace_limit:
                break
            outputs = forward_offline_model(model, batch, device=device)
            pred_actions = outputs["pred_actions"] if isinstance(outputs, Mapping) else outputs
            metric_pred = maybe_denormalize(pred_actions.detach(), action_transform)
            metric_target = maybe_denormalize(batch["target_actions"], action_transform)
            rows.extend(
                build_trace_rows(
                    run_id=run_id,
                    model_name=model_name,
                    split="eval_same_demo_teacher_forced",
                    batch=batch,
                    pred_actions=metric_pred.detach().cpu(),
                    target_actions=metric_target.detach().cpu(),
                    remaining=trace_limit - len(rows),
                )
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_repro_files(
    *,
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    command: Sequence[str] | None,
    split_metadata: Mapping[str, Any],
    normalization_stats: Mapping[str, Any],
    seed: int,
    epochs: int,
    batch_size: int,
    loss_threshold: float,
    best_metric: float,
    best_epoch: int,
    passed: bool,
    metrics_path: Path,
    trace_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    shutil.copyfile(config_path, output_dir / "config.yaml")
    write_json(output_dir / "split.json", split_metadata)
    write_json(output_dir / "normalization_stats.json", normalization_stats)
    write_json(output_dir / "environment.json", capture_environment_json())
    (output_dir / "environment.txt").write_text(capture_environment(), encoding="utf-8")
    (output_dir / "git_commit.txt").write_text(capture_git_commit(), encoding="utf-8")
    (output_dir / "seeds.txt").write_text(f"{seed}\n", encoding="utf-8")
    (output_dir / "command.txt").write_text(
        (" ".join(command) if command else " ".join(sys.argv)) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "single_demo_overfit_pass" if passed else "single_demo_overfit_fail",
        "config": str(config_path),
        "model": str(config["model"].get("name", config["model"]["temporal_adapter"])),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "loss_threshold": loss_threshold,
        "best_eval_same_demo_action_mse": best_metric,
        "best_epoch": best_epoch,
        "passed_threshold": passed,
        "metrics_csv": str(metrics_path),
        "trace_csv": str(trace_path),
        "evaluation_split": "same_demo_teacher_forced",
        "closed_loop_same_initial_condition": "not_run_demo_to_init_state_mapping_not_available",
        "is_reportable": False,
        "non_claims": [
            "not_closed_loop_success",
            "not_generalization_evidence",
            "not_future_latent_benefit_evidence",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "notes.md").write_text(build_notes(summary, rows), encoding="utf-8")
    (output_dir / "diagnostic_summary.md").write_text(
        build_notes(summary, rows),
        encoding="utf-8",
    )


def build_notes(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    final_rows = rows[-2:] if len(rows) >= 2 else rows
    lines = [
        "# Single-Demo Overfit Diagnostic",
        "",
        f"Status: {summary['status']}",
        f"Best same-demo action MSE: {summary['best_eval_same_demo_action_mse']}",
        f"Loss threshold: {summary['loss_threshold']}",
        f"Closed-loop same initial condition: {summary['closed_loop_same_initial_condition']}",
        "",
        "Final rows:",
    ]
    for row in final_rows:
        lines.append(
            f"- {row['split']}: action_mse={row['action_mse']}, "
            f"action_loss={row['action_loss']}"
        )
    lines.extend(
        [
            "",
            "Interpretation boundaries:",
            "- This trains and evaluates on the same demonstration under teacher forcing.",
            "- Passing only validates capacity and basic training mechanics.",
            "- It does not measure closed-loop success or generalization.",
        ]
    )
    return "\n".join(lines) + "\n"


def default_run_id(config: Mapping[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = str(config["experiment"]["name"])
    return f"{timestamp}_{name}_single_demo_overfit"


if __name__ == "__main__":
    raise SystemExit(main())
