#!/usr/bin/env python3
"""Offline checkpoint evaluation for action and WAM-GRU smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.registry import build_offline_model, count_parameters  # noqa: E402
from src.train.train_offline import (  # noqa: E402
    METRIC_FIELDNAMES,
    build_datasets,
    capture_environment_json,
    collate_action_batch,
    format_metric_row,
    has_future_latent_targets,
    infer_action_dim,
    infer_latent_dim,
    run_one_split,
    validate_training_scope,
    write_json,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.experiment_io import format_command  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


EVAL_FIELDNAMES = ["checkpoint", "source_split", *METRIC_FIELDNAMES]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path, help="Training run dir.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override. Defaults to run_dir/config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. Defaults to run_dir/best.pt.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to run_dir/eval_offline.csv.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="val",
        help="Dataset split to evaluate. Default: val.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Force mock dataset evaluation even if saved config lacks runtime.dry_run.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum evaluation steps per split.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device. Default is cpu for smoke-test portability.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_csv = run_eval_offline(
        run_dir=args.run_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        split=args.split,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        device_name=args.device,
        command=[sys.executable, "src/train/eval_offline.py", *(argv or sys.argv[1:])],
    )
    print(f"eval_offline_csv={output_csv}")
    return 0


def run_eval_offline(
    *,
    run_dir: Path,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    output_csv: Path | None = None,
    split: str = "val",
    dry_run: bool = False,
    max_steps: int | None = None,
    device_name: str = "cpu",
    command: Sequence[str] | None = None,
) -> Path:
    """Evaluate one offline checkpoint and return `eval_offline.csv` path."""

    run_dir = run_dir.expanduser()
    config_path = config_path or run_dir / "config.yaml"
    checkpoint_path = checkpoint_path or run_dir / "best.pt"
    output_csv = output_csv or run_dir / "eval_offline.csv"
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    config = load_config(config_path)
    validate_training_scope(config)
    eval_dry_run = dry_run or bool(config.get("runtime", {}).get("dry_run", False))
    seed_everything(int(config["experiment"]["seed"]))
    train_dataset, val_dataset, _, action_transform, _ = build_datasets(
        config,
        dry_run=eval_dry_run,
    )
    datasets = {"train": train_dataset, "val": val_dataset}
    selected_splits = ["train", "val"] if split == "both" else [split]

    sample = train_dataset[0]
    action_dim = infer_action_dim(sample)
    latent_dim = infer_latent_dim(sample) if has_future_latent_targets(sample) else None
    model = build_offline_model(config, action_dim=action_dim, latent_dim=latent_dim)
    parameter_counts = count_parameters(model)
    device = torch.device(device_name)
    model.to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows: list[dict[str, Any]] = []
    for source_split in selected_splits:
        loader = DataLoader(
            datasets[source_split],
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            collate_fn=collate_action_batch,
        )
        metrics = run_one_split(
            model,
            loader,
            device=device,
            optimizer=None,
            lambda_action=float(config["training"]["lambda_action"]),
            lambda_future=float(config["training"]["lambda_future"]),
            grad_clip_norm=None,
            max_steps=max_steps,
            action_transform=action_transform,
        )
        row = format_metric_row(
            int(checkpoint.get("epoch", -1)),
            f"eval_{source_split}",
            metrics,
            parameter_counts=parameter_counts,
        )
        row["checkpoint"] = str(checkpoint_path)
        row["source_split"] = source_split
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    (run_dir / "eval_command.txt").write_text(
        format_command(command),
        encoding="utf-8",
    )
    write_json(run_dir / "eval_environment.json", capture_environment_json())
    write_json(
        run_dir / "eval_summary.json",
        {
            "status": "offline_smoke_eval_not_closed_loop",
            "run_dir": str(run_dir),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "output_csv": str(output_csv),
            "dry_run": eval_dry_run,
            "split": split,
            "max_steps": max_steps,
            "rows": rows,
            "non_claims": [
                "not_closed_loop",
                "not_success_rate",
                "not_reportable_wam_improvement",
            ],
        },
    )
    return output_csv


def load_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    """Load a PyTorch checkpoint across versions with explicit map location."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - for older torch.
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a train_offline.py checkpoint")
    return checkpoint


if __name__ == "__main__":
    raise SystemExit(main())
