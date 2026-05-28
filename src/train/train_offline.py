#!/usr/bin/env python3
"""Config-driven offline action prediction training for action-only baselines."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.trajectory_window import (  # noqa: E402
    RawTrajectory,
    TrajectoryWindowDataset,
)
from src.data.split_normalization import (  # noqa: E402
    FieldStats,
    fit_train_only_standardization_stats,
)
from src.models.registry import build_action_model, count_parameters  # noqa: E402
from src.train.metrics import action_mse  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.experiment_io import create_experiment_dir, format_command  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


METRIC_FIELDNAMES = [
    "epoch",
    "split",
    "total_loss",
    "action_loss",
    "action_loss_units",
    "future_loss",
    "spike_loss",
    "action_mse",
    "action_mse_units",
    "steps",
    "samples",
    "parameter_count",
    "trainable_parameter_count",
    "lower_is_better",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Training YAML config.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use a tiny deterministic mock dataset and force at least one epoch.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum optimizer/eval steps per split per epoch.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override config output.output_dir.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Optional run directory name, mainly for deterministic tests.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device. Default is cpu for smoke-test portability.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = run_training(
        args.config,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        run_id=args.run_id,
        device_name=args.device,
        command=["python3", "src/train/train_offline.py", *(argv or sys.argv[1:])],
    )
    print(f"run_dir={run_dir}")
    return 0


def run_training(
    config_path: Path,
    *,
    dry_run: bool = False,
    max_steps: int | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
    device_name: str = "cpu",
    command: Sequence[str] | None = None,
) -> Path:
    """Run offline MLP action training and return the result directory."""

    config = prepare_runtime_config(
        load_config(config_path),
        dry_run=dry_run,
        max_steps=max_steps,
        output_dir=output_dir,
    )
    validate_training_scope(config)
    seed_everything(int(config["experiment"]["seed"]))
    train_dataset, val_dataset, split_metadata, action_transform, normalization_stats = (
        build_datasets(config, dry_run=dry_run)
    )

    notes = build_notes(dry_run=dry_run)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = create_experiment_dir(
        config,
        command=command,
        notes=notes,
        run_id=run_id,
        timestamp=timestamp,
    )
    write_command_script(run_dir / "command.sh", command)
    write_json(run_dir / "environment.json", capture_environment_json())
    (run_dir / "seeds.txt").write_text(
        f"{int(config['experiment']['seed'])}\n", encoding="utf-8"
    )

    write_json(run_dir / "split.json", split_metadata)
    write_json(run_dir / "normalization_stats.json", normalization_stats)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_action_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_action_batch,
    )

    sample = train_dataset[0]
    action_dim = infer_action_dim(sample)
    model = build_action_model(config, action_dim=action_dim)
    parameter_counts = count_parameters(model)
    device = torch.device(device_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))
    epochs = int(config["training"]["epochs"])
    if epochs <= 0:
        raise ValueError("training.epochs must be positive for train_offline.py")

    metrics_path = run_dir / "metrics.csv"
    best_metric = float("inf")
    best_epoch = -1
    rows: list[dict[str, Any]] = []
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()

        for epoch in range(epochs):
            train_metrics = run_one_split(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                lambda_action=float(config["training"]["lambda_action"]),
                grad_clip_norm=config["training"]["grad_clip_norm"],
                max_steps=max_steps,
                action_transform=action_transform,
            )
            val_metrics = run_one_split(
                model,
                val_loader,
                device=device,
                optimizer=None,
                lambda_action=float(config["training"]["lambda_action"]),
                grad_clip_norm=None,
                max_steps=max_steps,
                action_transform=action_transform,
            )

            train_row = format_metric_row(
                epoch,
                "train",
                train_metrics,
                parameter_counts=parameter_counts,
            )
            val_row = format_metric_row(
                epoch,
                "val",
                val_metrics,
                parameter_counts=parameter_counts,
            )
            writer.writerow(train_row)
            writer.writerow(val_row)
            handle.flush()
            rows.extend([train_row, val_row])

            current_metric = float(val_metrics["action_mse"])
            checkpoint = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                best_metric=min(best_metric, current_metric),
                best_epoch=best_epoch,
                metrics={"train": train_metrics, "val": val_metrics},
            )
            torch.save(checkpoint, run_dir / "checkpoint.pt")
            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch
                checkpoint["best_metric"] = best_metric
                checkpoint["best_epoch"] = best_epoch
                torch.save(checkpoint, run_dir / "best.pt")

    if not (run_dir / "best.pt").exists():
        torch.save(
            checkpoint_payload(
                epoch=epochs - 1,
                model=model,
                optimizer=optimizer,
                config=config,
                best_metric=best_metric,
                best_epoch=best_epoch,
                metrics={},
            ),
            run_dir / "best.pt",
        )
    if not rows:
        raise RuntimeError("metrics.csv was not populated")
    write_json(
        run_dir / "summary.json",
        build_summary(
            config=config,
            split_metadata=split_metadata,
            normalization_stats=normalization_stats,
            rows=rows,
            action_dim=action_dim,
            train_windows=len(train_dataset),
            val_windows=len(val_dataset),
            best_metric=best_metric,
            best_epoch=best_epoch,
            parameter_counts=parameter_counts,
        ),
    )
    return run_dir


def prepare_runtime_config(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
    max_steps: int | None,
    output_dir: Path | None,
) -> dict[str, Any]:
    runtime_config = deepcopy(dict(config))
    if output_dir is not None:
        runtime_config["output"]["output_dir"] = str(output_dir)
    if dry_run:
        runtime_config["training"]["epochs"] = max(
            1, int(runtime_config["training"]["epochs"])
        )
        runtime_config["training"]["batch_size"] = min(
            int(runtime_config["training"]["batch_size"]), 8
        )
        tags = list(runtime_config["experiment"].get("tags", []))
        for tag in ("dry_run", "mock_data"):
            if tag not in tags:
                tags.append(tag)
        runtime_config["experiment"]["tags"] = tags
    runtime_config["runtime"] = {
        "dry_run": dry_run,
        "max_steps": max_steps,
        "data_source": "mock" if dry_run else "libero_hdf5",
    }
    return runtime_config


def validate_training_scope(config: Mapping[str, Any]) -> None:
    """Fail closed for adapters/losses outside action-only baselines."""

    if config["model"]["temporal_adapter"] not in {"mlp", "gru"}:
        raise ValueError(
            "train_offline.py currently supports only temporal_adapter=mlp or gru"
        )
    if config["model"]["visual_encoder"] != "stub":
        raise ValueError("only visual_encoder=stub is implemented for this baseline")
    if config["model"]["text_encoder"] != "stub":
        raise ValueError("only text_encoder=stub is implemented for this baseline")
    if float(config["training"]["lambda_future"]) != 0.0:
        raise ValueError("future latent loss is not implemented for the MLP baseline")
    if float(config["training"]["lambda_spike"]) != 0.0:
        raise ValueError("spike loss is not implemented for the MLP baseline")


def build_notes(*, dry_run: bool) -> str:
    if dry_run:
        return (
            "# Notes\n\n"
            "Dry-run smoke training on deterministic mock trajectories only. "
            "This run is for code-path validation and must not be reported as a "
            "scientific result.\n\n"
            "This baseline is action-only with stub visual/text encoders: "
            "no future latent loss, no WAM claim, no SNN, and no closed-loop rollout.\n"
        )
    return (
        "# Notes\n\n"
        "Offline action-only baseline. Stub visual/text encoders are used; "
        "no future latent loss, WAM claim, SNN, or closed-loop rollout is "
        "implemented by this trainer.\n"
    )


def build_datasets(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
) -> tuple[
    TrajectoryWindowDataset,
    TrajectoryWindowDataset,
    dict[str, Any],
    ActionTransform | None,
    dict[str, Any],
]:
    history_len = int(config["data"]["history_len"])
    action_horizon = int(config["data"]["action_horizon"])
    future_horizon = int(config["data"]["future_horizon"])

    if dry_run:
        length = max(history_len + action_horizon + future_horizon + 8, 12)
        train_dataset = make_mock_action_dataset(
            trajectory_id="mock_train_0",
            split="train",
            length=length,
            history_len=history_len,
            action_horizon=action_horizon,
            future_horizon=future_horizon,
            action_dim=7,
        )
        val_dataset = make_mock_action_dataset(
            trajectory_id="mock_val_0",
            split="val",
            length=length,
            history_len=history_len,
            action_horizon=action_horizon,
            future_horizon=future_horizon,
            action_dim=7,
        )
        metadata = {
            "suite": config["data"]["suite"],
            "split_unit": "trajectory",
            "train": ["mock_train_0"],
            "val": ["mock_val_0"],
            "test": [],
            "seed": int(config["experiment"]["seed"]),
            "method": "mock_dry_run_separate_synthetic_trajectories",
            "mock": True,
            "reportable_scientific_result": False,
        }
        return train_dataset, val_dataset, metadata, None, no_normalization_record()

    trajectories, metadata = load_real_libero_trajectories(config)
    action_transform, normalization_stats = build_action_transform(trajectories, config)
    if action_transform is not None:
        trajectories = apply_action_transform(trajectories, action_transform)
    train_dataset = TrajectoryWindowDataset(
        trajectories,
        split="train",
        history_len=history_len,
        action_horizon=action_horizon,
        future_horizon=future_horizon,
    )
    val_dataset = TrajectoryWindowDataset(
        trajectories,
        split="val",
        history_len=history_len,
        action_horizon=action_horizon,
        future_horizon=future_horizon,
    )
    if len(train_dataset) == 0:
        raise ValueError("real LIBERO train split produced zero valid windows")
    if len(val_dataset) == 0:
        raise ValueError("real LIBERO val split produced zero valid windows")
    return train_dataset, val_dataset, metadata, action_transform, normalization_stats


@dataclass(frozen=True)
class ActionTransform:
    """Train-fitted per-action standardization for tensors shaped `[B, H, A]`."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def normalize_row(self, row: Sequence[float]) -> list[float]:
        return [
            (float(value) - self.mean[index]) / self.std[index]
            for index, value in enumerate(row)
        ]

    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        return tensor * std + mean


def build_action_transform(
    trajectories: Sequence[RawTrajectory],
    config: Mapping[str, Any],
) -> tuple[ActionTransform | None, dict[str, Any]]:
    """Build action normalization from train trajectories only."""

    normalization = config.get("normalization", {})
    actions_config = normalization.get("actions", {}) if isinstance(normalization, Mapping) else {}
    mode = actions_config.get("mode", "none") if isinstance(actions_config, Mapping) else "none"
    if mode == "none":
        return None, no_normalization_record()
    if mode != "standardize_train":
        raise ValueError(
            "normalization.actions.mode must be 'none' or 'standardize_train'"
        )

    stats = fit_train_only_standardization_stats(
        trajectories,
        split="train",
        fields=("actions",),
    )["actions"]
    transform = ActionTransform(mean=tuple(stats.mean), std=tuple(stats.std))
    return transform, standardization_record(stats)


def apply_action_transform(
    trajectories: Sequence[RawTrajectory],
    transform: ActionTransform,
) -> list[RawTrajectory]:
    """Return trajectories with normalized `[T, A]` action arrays."""

    normalized: list[RawTrajectory] = []
    for trajectory in trajectories:
        actions = [transform.normalize_row(row) for row in trajectory.actions]
        normalized.append(replace(trajectory, actions=actions))
    return normalized


def no_normalization_record() -> dict[str, Any]:
    return {
        "actions": {
            "mode": "none",
            "units": "raw_action_units",
            "source_split": "train",
            "note": "No action normalization is applied.",
        },
        "images": {"mode": "stub_encoder_no_image_stats"},
        "language": {"mode": "stub_encoder_no_tokenizer_stats"},
    }


def standardization_record(stats: FieldStats) -> dict[str, Any]:
    return {
        "actions": {
            "mode": "standardize_train",
            "mean": stats.mean,
            "std": stats.std,
            "count": stats.count,
            "source_split": stats.source_split,
            "training_loss_units": "normalized_action_units",
            "reported_action_mse_units": "raw_action_units",
        },
        "images": {"mode": "stub_encoder_no_image_stats"},
        "language": {"mode": "stub_encoder_no_tokenizer_stats"},
    }


def make_mock_action_dataset(
    *,
    trajectory_id: str,
    split: str,
    length: int,
    history_len: int,
    action_horizon: int,
    future_horizon: int,
    action_dim: int,
) -> TrajectoryWindowDataset:
    """Create a deterministic action-only mock dataset for smoke training."""

    actions = [
        [float(timestep) + 0.01 * float(dim) for dim in range(action_dim)]
        for timestep in range(length)
    ]
    frame_refs = [f"{trajectory_id}:frame:{timestep}" for timestep in range(length)]
    trajectory = RawTrajectory(
        images=frame_refs,
        actions=actions,
        states=None,
        frame_refs=frame_refs,
        language="mock instruction",
        trajectory_id=trajectory_id,
        split=split,
    )
    return TrajectoryWindowDataset(
        [trajectory],
        split=split,
        history_len=history_len,
        action_horizon=action_horizon,
        future_horizon=future_horizon,
    )


def load_real_libero_trajectories(
    config: Mapping[str, Any],
) -> tuple[list[RawTrajectory], dict[str, Any]]:
    """Load action trajectories from configured LIBERO HDF5 files."""

    try:
        import h5py  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional environment dependency.
        raise RuntimeError("h5py and numpy are required for real LIBERO loading") from exc

    dataset_root = resolve_dataset_root(str(config["data"]["dataset_root"]))
    suite = str(config["data"]["suite"])
    files = find_demo_files(dataset_root, suite)
    if not files:
        raise FileNotFoundError(
            f"no .hdf5/.h5 files found for suite={suite!r} under {dataset_root}"
        )
    max_files = optional_positive_int(config["data"].get("max_files"), "data.max_files")
    if max_files is not None:
        files = files[:max_files]

    trajectories: list[RawTrajectory] = []
    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            for demo_path, group in iter_demo_groups(handle):
                if "actions" not in group:
                    continue
                actions = np.asarray(group["actions"][()], dtype=np.float32)
                if actions.ndim != 2:
                    raise ValueError(
                        f"{file_path}:{demo_path}/actions must have shape [T, A]"
                    )
                length = int(actions.shape[0])
                trajectory_id = f"{safe_relative(file_path, dataset_root)}:{demo_path}"
                frame_refs = [
                    f"{trajectory_id}:obs/agentview_rgb:{index}" for index in range(length)
                ]
                trajectories.append(
                    RawTrajectory(
                        images=frame_refs,
                        actions=actions,
                        states=None,
                        frame_refs=frame_refs,
                        language=extract_language(handle, group),
                        trajectory_id=trajectory_id,
                        split="unspecified",
                    )
                )

    if not trajectories:
        raise ValueError(f"no action trajectories found under {dataset_root}")
    split_trajectories, split_metadata = assign_splits(trajectories, config)
    return split_trajectories, split_metadata


def resolve_dataset_root(value: str) -> Path:
    if value.startswith("env:"):
        env_name = value.split(":", maxsplit=1)[1]
        env_value = os.environ.get(env_name)
        if not env_value:
            raise EnvironmentError(f"{env_name} is not set")
        return Path(env_value).expanduser()
    return Path(os.path.expandvars(value)).expanduser()


def find_demo_files(dataset_root: Path, suite: str) -> list[Path]:
    roots = [
        dataset_root / suite,
        dataset_root / f"{suite}_no_noops",
        dataset_root / "datasets" / suite,
        dataset_root,
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        for pattern in ("*.hdf5", "*.h5"):
            files.extend(sorted(root.rglob(pattern)))
    return sorted(dict.fromkeys(files))


def iter_demo_groups(handle: Any) -> list[tuple[str, Any]]:
    if "data" in handle and hasattr(handle["data"], "keys"):
        data_group = handle["data"]
        return [
            (f"data/{key}", data_group[key])
            for key in sorted(data_group.keys(), key=demo_sort_key)
            if hasattr(data_group[key], "keys")
        ]
    if "actions" in handle:
        return [("/", handle)]
    return []


def demo_sort_key(value: str) -> tuple[str, int]:
    suffix = value.rsplit("_", maxsplit=1)[-1]
    return (value.rsplit("_", maxsplit=1)[0], int(suffix) if suffix.isdigit() else -1)


def extract_language(handle: Any, group: Any) -> str:
    for attrs in (group.attrs, getattr(handle.get("data", None), "attrs", {}), handle.attrs):
        language = language_from_attrs(attrs)
        if language:
            return language
    return ""


def language_from_attrs(attrs: Any) -> str:
    for key in ("language_instruction", "language", "instruction"):
        if key in attrs:
            return stringify_attr(attrs[key])
    if "problem_info" in attrs:
        value = stringify_attr(attrs["problem_info"])
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            parsed = value
        if isinstance(parsed, Mapping):
            language = parsed.get("language_instruction")
            if language is not None:
                return stringify_attr(language)
    return ""


def stringify_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        value = value.tolist()
    return str(value)


def assign_splits(
    trajectories: Sequence[RawTrajectory],
    config: Mapping[str, Any],
) -> tuple[list[RawTrajectory], dict[str, Any]]:
    split_config = config["data"]["split"]
    explicit_train = set(split_config.get("train", []))
    explicit_val = set(split_config.get("val", []))
    explicit_test = set(split_config.get("test", []))

    sorted_trajectories = sorted(trajectories, key=lambda item: item.trajectory_id)
    if explicit_train or explicit_val or explicit_test:
        split_by_id: dict[str, str] = {}
        for trajectory in sorted_trajectories:
            identifiers = {
                trajectory.trajectory_id,
                trajectory.trajectory_id.split(":", maxsplit=1)[-1],
            }
            if identifiers & explicit_train:
                split_by_id[trajectory.trajectory_id] = "train"
            elif identifiers & explicit_val:
                split_by_id[trajectory.trajectory_id] = "val"
            elif identifiers & explicit_test:
                split_by_id[trajectory.trajectory_id] = "test"
        assigned = [
            replace(trajectory, split=split_by_id[trajectory.trajectory_id])
            for trajectory in sorted_trajectories
            if trajectory.trajectory_id in split_by_id
        ]
        method = "config_explicit_trajectory_ids"
    else:
        split_names = deterministic_split_names(len(sorted_trajectories))
        assigned = [
            replace(trajectory, split=split)
            for trajectory, split in zip(sorted_trajectories, split_names, strict=True)
        ]
        method = "deterministic_sorted_demo_ids"

    assigned = limit_trajectories_by_split(assigned, config)
    metadata = {
        "suite": config["data"]["suite"],
        "split_unit": "trajectory",
        "train": [item.trajectory_id for item in assigned if item.split == "train"],
        "val": [item.trajectory_id for item in assigned if item.split == "val"],
        "test": [item.trajectory_id for item in assigned if item.split == "test"],
        "seed": int(config["experiment"]["seed"]),
        "method": method,
        "mock": False,
        "limits": {
            "max_files": config["data"].get("max_files"),
            "max_train_trajectories": config["data"].get("max_train_trajectories"),
            "max_val_trajectories": config["data"].get("max_val_trajectories"),
            "max_test_trajectories": config["data"].get("max_test_trajectories"),
        },
    }
    return assigned, metadata


def limit_trajectories_by_split(
    trajectories: Sequence[RawTrajectory],
    config: Mapping[str, Any],
) -> list[RawTrajectory]:
    limits = {
        "train": optional_nonnegative_int(
            config["data"].get("max_train_trajectories"),
            "data.max_train_trajectories",
        ),
        "val": optional_nonnegative_int(
            config["data"].get("max_val_trajectories"),
            "data.max_val_trajectories",
        ),
        "test": optional_nonnegative_int(
            config["data"].get("max_test_trajectories"),
            "data.max_test_trajectories",
        ),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    output: list[RawTrajectory] = []
    for trajectory in trajectories:
        split = trajectory.split
        limit = limits.get(split)
        if limit is not None and counts[split] >= limit:
            continue
        output.append(trajectory)
        if split in counts:
            counts[split] += 1
    return output


def optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer when set")
    return value


def optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer when set")
    return value


def deterministic_split_names(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["train"]
    if count == 2:
        return ["train", "val"]

    train_count = max(1, int(count * 0.8))
    val_count = max(1, int(count * 0.1))
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1
    test_count = count - train_count - val_count
    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def collate_action_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate action-only fields into `[B, T, A]` and `[B, H, A]` tensors."""

    action_history = torch.stack(
        [
            torch.as_tensor(sample["action_history"], dtype=torch.float32)
            for sample in samples
        ],
        dim=0,
    )
    target_actions = torch.stack(
        [
            torch.as_tensor(sample["target_actions"], dtype=torch.float32)
            for sample in samples
        ],
        dim=0,
    )
    return {
        "action_history": action_history,
        "target_actions": target_actions,
        "trajectory_id": [sample["trajectory_id"] for sample in samples],
        "time_index": [sample["time_index"] for sample in samples],
    }


def infer_action_dim(sample: Mapping[str, Any]) -> int:
    target = torch.as_tensor(sample["target_actions"])
    if target.ndim != 2:
        raise ValueError(f"target_actions must have shape [H, A], got {tuple(target.shape)}")
    return int(target.shape[-1])


def run_one_split(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lambda_action: float,
    grad_clip_norm: float | None,
    max_steps: int | None,
    action_transform: ActionTransform | None,
) -> dict[str, float | int]:
    is_train = optimizer is not None
    model.train(is_train)

    squared_error_sum = 0.0
    element_count = 0
    total_loss_sum = 0.0
    action_loss_sum = 0.0
    steps = 0
    samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            if max_steps is not None and steps >= max_steps:
                break
            action_history = batch["action_history"].to(device)
            target_actions = batch["target_actions"].to(device)
            pred_actions = model(action_history)
            action_loss = action_mse(pred_actions, target_actions)
            total_loss = lambda_action * action_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("non-finite training loss")

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

            metric_pred = pred_actions.detach()
            metric_target = target_actions
            if action_transform is not None:
                metric_pred = action_transform.denormalize_tensor(metric_pred)
                metric_target = action_transform.denormalize_tensor(metric_target)

            squared_error_sum += (metric_pred - metric_target).pow(2).sum().item()
            element_count += int(metric_target.numel())
            action_loss_sum += float(action_loss.detach().item()) * int(target_actions.numel())
            total_loss_sum += float(total_loss.detach().item()) * int(target_actions.numel())
            steps += 1
            samples += int(target_actions.shape[0])

    if steps == 0 or element_count == 0:
        raise ValueError("split produced no batches")
    action_mse_value = squared_error_sum / element_count
    return {
        "total_loss": total_loss_sum / element_count,
        "action_loss": action_loss_sum / element_count,
        "action_loss_units": (
            "normalized_action_units"
            if action_transform is not None
            else "raw_action_units"
        ),
        "future_loss": 0.0,
        "spike_loss": 0.0,
        "action_mse": action_mse_value,
        "steps": steps,
        "samples": samples,
    }


def format_metric_row(
    epoch: int,
    split: str,
    metrics: Mapping[str, float | int],
    *,
    parameter_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "split": split,
        "total_loss": format_float(float(metrics["total_loss"])),
        "action_loss": format_float(float(metrics["action_loss"])),
        "action_loss_units": str(metrics["action_loss_units"]),
        "future_loss": format_float(float(metrics["future_loss"])),
        "spike_loss": format_float(float(metrics["spike_loss"])),
        "action_mse": format_float(float(metrics["action_mse"])),
        "action_mse_units": "raw_action_units",
        "steps": int(metrics["steps"]),
        "samples": int(metrics["samples"]),
        "parameter_count": int(parameter_counts["parameter_count"]),
        "trainable_parameter_count": int(
            parameter_counts["trainable_parameter_count"]
        ),
        "lower_is_better": "true",
    }


def format_float(value: float) -> str:
    return f"{value:.10g}"


def checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    best_metric: float,
    best_epoch: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(config),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "metrics": dict(metrics),
    }


def build_summary(
    *,
    config: Mapping[str, Any],
    split_metadata: Mapping[str, Any],
    normalization_stats: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    action_dim: int,
    train_windows: int,
    val_windows: int,
    best_metric: float,
    best_epoch: int,
    parameter_counts: Mapping[str, int],
) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    first_train_loss = float(train_rows[0]["total_loss"]) if train_rows else None
    last_train_loss = float(train_rows[-1]["total_loss"]) if train_rows else None
    return {
        "status": "engineering_smoke_test_not_scientific_result",
        "model": {
            "name": config["model"].get(
                "name",
                f"action_only_{config['model']['temporal_adapter']}",
            ),
            "temporal_adapter": config["model"]["temporal_adapter"],
            "visual_encoder": config["model"]["visual_encoder"],
            "text_encoder": config["model"]["text_encoder"],
            "action_dim": action_dim,
            "hidden_dim": config["model"]["hidden_dim"],
            "parameter_count": int(parameter_counts["parameter_count"]),
            "trainable_parameter_count": int(
                parameter_counts["trainable_parameter_count"]
            ),
        },
        "data": {
            "suite": config["data"]["suite"],
            "dataset_root": config["data"]["dataset_root"],
            "history_len": config["data"]["history_len"],
            "action_horizon": config["data"]["action_horizon"],
            "future_horizon": config["data"]["future_horizon"],
            "split": split_metadata,
            "train_windows": train_windows,
            "val_windows": val_windows,
        },
        "normalization": normalization_stats,
        "metrics": {
            "best_val_action_mse": best_metric,
            "best_epoch": best_epoch,
            "first_train_total_loss": first_train_loss,
            "last_train_total_loss": last_train_loss,
            "train_total_loss_decreased": (
                None
                if first_train_loss is None or last_train_loss is None
                else last_train_loss < first_train_loss
            ),
            "last_train_action_mse": (
                float(train_rows[-1]["action_mse"]) if train_rows else None
            ),
            "last_val_action_mse": (
                float(val_rows[-1]["action_mse"]) if val_rows else None
            ),
            "action_mse_units": "raw_action_units",
            "lower_is_better": True,
        },
        "non_claims": [
            "not_wam",
            "not_vla",
            "not_snn",
            "not_closed_loop",
            "not_generalization_benchmark",
        ],
    }


def write_command_script(path: Path, command: Sequence[str] | None) -> None:
    text = "#!/usr/bin/env bash\nset -euo pipefail\n"
    text += format_command(command)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def capture_environment_json() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "env": {
            "LIBERO_DATASET_ROOT": os.environ.get("LIBERO_DATASET_ROOT", ""),
            "LIBERO_DATA_ROOT": os.environ.get("LIBERO_DATA_ROOT", ""),
            "LIBERO_REPO_ROOT": os.environ.get("LIBERO_REPO_ROOT", ""),
        },
        "packages": {
            "torch": getattr(torch, "__version__", None),
            "cuda_available": torch.cuda.is_available(),
        },
    }
    try:
        import h5py  # type: ignore[import-not-found]

        payload["packages"]["h5py"] = getattr(h5py, "__version__", None)
    except ImportError:
        payload["packages"]["h5py"] = None
    try:
        import numpy as np  # type: ignore[import-not-found]

        payload["packages"]["numpy"] = getattr(np, "__version__", None)
    except ImportError:
        payload["packages"]["numpy"] = None
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
