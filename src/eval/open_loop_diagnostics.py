#!/usr/bin/env python3
"""Teacher-forced open-loop action diagnostics for offline checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.registry import build_offline_model  # noqa: E402
from src.train.eval_offline import load_checkpoint  # noqa: E402
from src.train.metrics import action_mse_per_dimension, action_mse_per_horizon  # noqa: E402
from src.train.train_offline import (  # noqa: E402
    ActionTransform,
    build_datasets,
    collate_action_batch,
    forward_offline_model,
    has_current_latent,
    has_future_latent_targets,
    infer_action_dim,
    infer_current_latent_dim,
    infer_latent_dim,
    infer_state_dim,
    infer_task_count,
    validate_training_scope,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


METRIC_FIELDNAMES = [
    "run_id",
    "model",
    "checkpoint",
    "split",
    "baseline",
    "action_mse",
    "action_mse_by_horizon",
    "action_mse_by_dimension",
    "samples",
    "windows",
    "action_units",
]

TRACE_FIELDNAMES = [
    "run_id",
    "model",
    "split",
    "trajectory_id",
    "time_index",
    "target_action_index",
    "horizon",
    "expert_action",
    "pred_action",
    "expert_action_norm",
    "pred_action_norm",
    "expert_gripper",
    "pred_gripper",
    "cosine_similarity",
    "per_dimension_error",
    "squared_error",
]


@dataclass
class BaselineAccumulator:
    squared_error_sum: torch.Tensor | None = None
    batch_count: int = 0
    sample_count: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        error = (pred.detach().cpu() - target.detach().cpu()).pow(2)
        if self.squared_error_sum is None:
            self.squared_error_sum = error.sum(dim=0)
        else:
            self.squared_error_sum += error.sum(dim=0)
        self.batch_count += 1
        self.sample_count += int(target.shape[0])

    def to_metrics(self) -> dict[str, Any]:
        if self.squared_error_sum is None or self.sample_count <= 0:
            raise ValueError("baseline received no samples")
        mean_error = self.squared_error_sum / self.sample_count  # [H, A]
        return {
            "action_mse": float(mean_error.mean().item()),
            "action_mse_by_horizon": mean_error.mean(dim=-1).tolist(),
            "action_mse_by_dimension": mean_error.mean(dim=0).tolist(),
            "samples": self.sample_count,
            "windows": self.sample_count,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--trace_limit", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = run_open_loop_diagnostics(
        run_dir=args.run_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        dry_run=args.dry_run,
        max_batches=args.max_batches,
        trace_limit=args.trace_limit,
        device_name=args.device,
        seed=args.seed,
        command=[sys.executable, "-m", "src.eval.open_loop_diagnostics", *(argv or sys.argv[1:])],
    )
    print(f"open_loop_diagnostics_dir={output_dir}")
    return 0


def run_open_loop_diagnostics(
    *,
    run_dir: Path,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    split: str = "val",
    dry_run: bool = False,
    max_batches: int | None = None,
    trace_limit: int = 256,
    device_name: str = "cpu",
    seed: int = 0,
    command: Sequence[str] | None = None,
) -> Path:
    run_dir = run_dir.expanduser()
    config_path = config_path or run_dir / "config.yaml"
    checkpoint_path = checkpoint_path or run_dir / "best.pt"
    output_dir = output_dir or run_dir / "open_loop_diagnostics"
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if trace_limit < 0:
        raise ValueError("trace_limit must be non-negative")

    config = load_config(config_path)
    validate_training_scope(config)
    eval_dry_run = dry_run or bool(config.get("runtime", {}).get("dry_run", False))
    seed_everything(seed)

    train_dataset, val_dataset, split_metadata, action_transform, _ = build_datasets(
        config,
        dry_run=eval_dry_run,
    )
    dataset = train_dataset if split == "train" else val_dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_action_batch,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
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
    model = build_offline_model(
        config,
        action_dim=action_dim,
        latent_dim=latent_dim,
        state_dim=infer_state_dim(sample),
        num_tasks=infer_task_count(config, split_metadata),
    )
    device = torch.device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    train_mean, train_std = estimate_train_action_stats(
        train_loader,
        action_transform=action_transform,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    accumulators = {
        "model": BaselineAccumulator(),
        "zero_action": BaselineAccumulator(),
        "random_action_train_gaussian": BaselineAccumulator(),
        "mean_action_train": BaselineAccumulator(),
        "last_action": BaselineAccumulator(),
    }
    trace_rows: list[dict[str, str]] = []
    model_name = str(config["model"].get("name", config["model"]["temporal_adapter"]))
    run_id = run_dir.name

    with torch.no_grad():
        for batch_index, batch in enumerate(eval_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            action_history = batch["action_history"].to(device)
            target_actions = batch["target_actions"].to(device)
            outputs = forward_offline_model(model, batch, device=device)
            pred_actions = outputs["pred_actions"] if isinstance(outputs, Mapping) else outputs

            metric_pred = maybe_denormalize(pred_actions.detach(), action_transform)
            metric_target = maybe_denormalize(target_actions.detach(), action_transform)
            metric_history = maybe_denormalize(action_history.detach(), action_transform)

            baselines = {
                "model": metric_pred,
                "zero_action": torch.zeros_like(metric_target),
                "random_action_train_gaussian": random_action_like(
                    metric_target,
                    mean=train_mean,
                    std=train_std,
                    generator=generator,
                ),
                "mean_action_train": mean_action_like(metric_target, train_mean),
                "last_action": last_action_like(metric_target, metric_history),
            }
            for name, prediction in baselines.items():
                accumulators[name].update(prediction, metric_target)

            if len(trace_rows) < trace_limit:
                trace_rows.extend(
                    build_trace_rows(
                        run_id=run_id,
                        model_name=model_name,
                        split=split,
                        batch=batch,
                        pred_actions=metric_pred.detach().cpu(),
                        target_actions=metric_target.detach().cpu(),
                        remaining=trace_limit - len(trace_rows),
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "open_loop_metrics.csv"
    trace_path = output_dir / "action_trace_diagnostics.csv"
    summary_path = output_dir / "summary.json"

    metric_rows = []
    for baseline, accumulator in accumulators.items():
        metrics = accumulator.to_metrics()
        metric_rows.append(
            {
                "run_id": run_id,
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "split": split,
                "baseline": baseline,
                "action_mse": metrics["action_mse"],
                "action_mse_by_horizon": json.dumps(metrics["action_mse_by_horizon"]),
                "action_mse_by_dimension": json.dumps(metrics["action_mse_by_dimension"]),
                "samples": metrics["samples"],
                "windows": metrics["windows"],
                "action_units": "raw_action_units",
            }
        )
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(metric_rows)
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(trace_rows)

    rows_by_name = {row["baseline"]: row for row in metric_rows}
    model_mse = float(rows_by_name["model"]["action_mse"])
    beats = {
        name: model_mse < float(row["action_mse"])
        for name, row in rows_by_name.items()
        if name != "model"
    }
    summary = {
        "status": "teacher_forced_open_loop_diagnostic",
        "run_dir": str(run_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": split,
        "seed": seed,
        "dry_run": eval_dry_run,
        "max_batches": max_batches,
        "trace_limit": trace_limit,
        "metrics_csv": str(metrics_path),
        "trace_csv": str(trace_path),
        "split_metadata": split_metadata,
        "model_action_mse": model_mse,
        "beats_baselines": beats,
        "non_claims": [
            "not_closed_loop_success",
            "not_future_latent_benefit_evidence",
            "not_policy_robustness_evidence",
        ],
    }
    write_reproducibility_files(
        output_dir=output_dir,
        summary=summary,
        metric_rows=metric_rows,
        command=command,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output_dir


def estimate_train_action_stats(
    loader: DataLoader,
    *,
    action_transform: ActionTransform | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    for batch in loader:
        target = maybe_denormalize(batch["target_actions"], action_transform)
        rows.append(target.reshape(-1, target.shape[-1]).detach().cpu())
    if not rows:
        raise ValueError("train loader produced no actions for baseline stats")
    actions = torch.cat(rows, dim=0)
    std = actions.std(dim=0, unbiased=False)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return actions.mean(dim=0), std


def maybe_denormalize(
    tensor: torch.Tensor,
    action_transform: ActionTransform | None,
) -> torch.Tensor:
    if action_transform is None:
        return tensor.detach().cpu()
    return action_transform.denormalize_tensor(tensor).detach().cpu()


def mean_action_like(target: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return mean.to(dtype=target.dtype).view(1, 1, -1).expand_as(target).clone()


def random_action_like(
    target: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
    return noise * std.to(dtype=target.dtype).view(1, 1, -1) + mean.to(dtype=target.dtype).view(1, 1, -1)


def last_action_like(target: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
    last = action_history[:, -1:, :]
    return last.expand_as(target).clone()


def build_trace_rows(
    *,
    run_id: str,
    model_name: str,
    split: str,
    batch: Mapping[str, Any],
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    remaining: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    batch_size, horizon, _ = target_actions.shape
    for batch_index in range(batch_size):
        trajectory_id = str(batch["trajectory_id"][batch_index])
        time_index = int(batch["time_index"][batch_index])
        for h in range(horizon):
            if len(rows) >= remaining:
                return rows
            expert = target_actions[batch_index, h]
            pred = pred_actions[batch_index, h]
            error = pred - expert
            cosine = F.cosine_similarity(
                pred.view(1, -1),
                expert.view(1, -1),
                dim=-1,
                eps=1e-8,
            ).item()
            rows.append(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "split": split,
                    "trajectory_id": trajectory_id,
                    "time_index": str(time_index),
                    "target_action_index": str(time_index + 1 + h),
                    "horizon": str(h),
                    "expert_action": json.dumps(expert.tolist()),
                    "pred_action": json.dumps(pred.tolist()),
                    "expert_action_norm": f"{float(expert.norm().item()):.10g}",
                    "pred_action_norm": f"{float(pred.norm().item()):.10g}",
                    "expert_gripper": f"{float(expert[-1].item()):.10g}",
                    "pred_gripper": f"{float(pred[-1].item()):.10g}",
                    "cosine_similarity": f"{float(cosine):.10g}",
                    "per_dimension_error": json.dumps(error.tolist()),
                    "squared_error": f"{float(error.pow(2).mean().item()):.10g}",
                }
            )
    return rows


def write_reproducibility_files(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    command: Sequence[str] | None,
    checkpoint_path: Path,
    config_path: Path,
) -> None:
    (output_dir / "command.txt").write_text(
        (" ".join(command) if command else " ".join(sys.argv)) + "\n",
        encoding="utf-8",
    )
    (output_dir / "checkpoint_path.txt").write_text(str(checkpoint_path) + "\n", encoding="utf-8")
    (output_dir / "config_path.txt").write_text(str(config_path) + "\n", encoding="utf-8")
    (output_dir / "seeds.txt").write_text(f"{summary.get('seed', 'unknown')}\n", encoding="utf-8")
    git_info = get_git_info()
    (output_dir / "git_commit.txt").write_text(
        f"commit={git_info['commit']}\n"
        f"dirty={git_info['dirty']}\n",
        encoding="utf-8",
    )
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(env_info, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    notes = [
        "# Open-Loop Diagnostic Notes",
        "",
        "Teacher-forced evaluation uses demonstration action history and current latents.",
        "It does not measure closed-loop success or robustness.",
        "The gripper diagnostic uses the last action dimension.",
        "Future-latent benefit/harm is not claimed from this diagnostic.",
        "",
        f"Metrics CSV: {summary['metrics_csv']}",
        f"Trace CSV: {summary['trace_csv']}",
    ]
    (output_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (output_dir / "diagnostic_summary.md").write_text(
        build_diagnostic_summary(summary=summary, metric_rows=metric_rows),
        encoding="utf-8",
    )


def build_diagnostic_summary(
    *,
    summary: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Teacher-Forced Open-Loop Diagnostic Summary",
        "",
        f"Run directory: `{summary['run_dir']}`",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Split: `{summary['split']}`",
        f"Dry run: `{summary['dry_run']}`",
        "",
        "## Action MSE",
        "",
        "| Baseline | Action MSE | Beats baseline? |",
        "|---|---:|---|",
    ]
    beats = summary.get("beats_baselines", {})
    for row in metric_rows:
        baseline = str(row["baseline"])
        if baseline == "model":
            verdict = "reference"
        else:
            verdict = "yes" if bool(beats.get(baseline, False)) else "no"
        lines.append(f"| {baseline} | {float(row['action_mse']):.8g} | {verdict} |")
    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- This is teacher-forced open-loop action prediction on demonstration windows.",
            "- It does not measure closed-loop success, policy robustness, or future-latent rollout benefit.",
            "- The gripper diagnostic uses the last action dimension.",
            "",
            f"Metrics CSV: `{summary['metrics_csv']}`",
            f"Trace CSV: `{summary['trace_csv']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def get_git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"commit": commit, "dirty": str(bool(status))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": "unknown"}


if __name__ == "__main__":
    raise SystemExit(main())
