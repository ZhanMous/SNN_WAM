#!/usr/bin/env python3
"""Config-driven offline training for action and minimal WAM-GRU baselines."""

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
from src.models.encoders import (  # noqa: E402
    build_frozen_visual_encoder,
    encode_sequence,
)
from src.models.registry import build_offline_model, count_parameters  # noqa: E402
from src.train.metrics import (  # noqa: E402
    action_mse,
    action_mse_per_horizon,
    action_mse_per_dimension,
    future_latent_cosine_error,
    future_latent_mse,
    patch_cosine_error,
    patch_mean_cosine_error,
    patch_mse,
)
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
    "future_latent_cosine_error",
    "future_latent_mse",
    "future_latent_cosine_error_by_horizon",
    "future_latent_mse_by_horizon",
    "patch_mse",
    "patch_cosine_error",
    "patch_mean_cosine_error",
    "patch_mse_by_horizon",
    "action_mse_by_horizon",
    "action_mse_by_dimension",
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
        command=[sys.executable, "src/train/train_offline.py", *(argv or sys.argv[1:])],
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
    """Run offline training and return the result directory."""

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
    latent_dim = None
    if has_future_latent_targets(sample):
        latent_dim = infer_latent_dim(sample)
    elif has_current_latent(sample):
        latent_dim = infer_current_latent_dim(sample)
    state_dim = infer_state_dim(sample)
    model = build_offline_model(
        config,
        action_dim=action_dim,
        latent_dim=latent_dim,
        state_dim=state_dim,
        num_tasks=infer_task_count(config, split_metadata),
    )
    parameter_counts = count_parameters(model)
    device = torch.device(device_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))
    epochs = int(config["training"]["epochs"])
    if epochs <= 0:
        raise ValueError("training.epochs must be positive for train_offline.py")

    metrics_path = run_dir / "metrics.csv"
    train_log_path = run_dir / "train.log"
    best_metric = float("inf")
    best_epoch = -1
    rows: list[dict[str, Any]] = []
    last_checkpoint: dict[str, Any] | None = None
    train_log_path.write_text(
        "epoch,split,total_loss,action_loss,future_loss,action_mse,"
        "future_latent_cosine_error\n",
        encoding="utf-8",
    )
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
                lambda_future=float(config["training"]["lambda_future"]),
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
                lambda_future=float(config["training"]["lambda_future"]),
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
            append_train_log(train_log_path, train_row)
            append_train_log(train_log_path, val_row)
            rows.extend([train_row, val_row])

            current_metric = checkpoint_selection_metric(config, val_metrics)
            last_checkpoint = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                best_metric=min(best_metric, current_metric),
                best_epoch=best_epoch,
                metrics={"train": train_metrics, "val": val_metrics},
            )
            torch.save(last_checkpoint, run_dir / "checkpoint.pt")
            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch
                last_checkpoint["best_metric"] = best_metric
                last_checkpoint["best_epoch"] = best_epoch
                torch.save(last_checkpoint, run_dir / "best.pt")

    if last_checkpoint is None:
        raise RuntimeError("training loop produced no checkpoints")
    if not (run_dir / "best.pt").exists():
        last_checkpoint["best_metric"] = best_metric
        last_checkpoint["best_epoch"] = best_epoch
        torch.save(last_checkpoint, run_dir / "best.pt")
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
            latent_dim=latent_dim,
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
    """Fail closed for adapters/losses outside implemented Phase-1 scope."""

    adapter = config["model"]["temporal_adapter"]
    if adapter not in {"mlp", "gru", "wam_gru", "bc_gru"}:
        raise ValueError(
            "train_offline.py currently supports temporal_adapter=mlp, gru, "
            "wam_gru, or bc_gru"
        )
    if config["model"]["text_encoder"] != "stub":
        raise ValueError("only text_encoder=stub is implemented for this baseline")
    if adapter in {"mlp", "gru"} and config["model"]["visual_encoder"] != "stub":
        raise ValueError("action-only baselines currently require visual_encoder=stub")
    if adapter in {"mlp", "gru"} and float(config["training"]["lambda_future"]) != 0.0:
        raise ValueError("future latent loss requires temporal_adapter=wam_gru")
    if adapter == "bc_gru":
        if config["model"]["visual_encoder"] == "stub":
            raise ValueError("bc_gru requires a frozen visual encoder, not stub")
        if float(config["training"]["lambda_future"]) != 0.0:
            raise ValueError("bc_gru is an action-only baseline; set lambda_future=0")
    if adapter == "wam_gru":
        if int(config["data"]["future_horizon"]) <= 0:
            raise ValueError("wam_gru requires data.future_horizon > 0")
        if config["model"]["visual_encoder"] == "stub":
            raise ValueError("wam_gru requires a frozen visual encoder, not stub")
    if float(config["training"]["lambda_spike"]) != 0.0:
        raise ValueError("spike loss is not implemented for this trainer")


def build_notes(*, dry_run: bool) -> str:
    if dry_run:
        return (
            "# Notes\n\n"
            "Dry-run smoke training on deterministic mock trajectories only. "
            "This run is for code-path validation and must not be reported as a "
            "scientific result.\n\n"
            "When configured with WAM-GRU, future latent targets are produced by "
            "the frozen smoke time-index encoder. No real visual backbone, SNN, "
            "or closed-loop rollout is evaluated.\n"
        )
    return (
        "# Notes\n\n"
        "Offline Phase-1 trainer. Real WAM-GRU runs require precomputed or "
        "adapter-produced frozen visual latents. No SNN or closed-loop rollout "
        "is implemented by this trainer.\n"
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
    include_current_latent = uses_current_latent(config)
    include_future_latents = requires_future_latents(config)

    if dry_run:
        length = max(history_len + action_horizon + future_horizon + 8, 12)
        visual_encoder = (
            build_frozen_visual_encoder(config["model"]) if include_current_latent else None
        )
        train_dataset = make_mock_action_dataset(
            trajectory_id="mock_train_0",
            split="train",
            length=length,
            history_len=history_len,
            action_horizon=action_horizon,
            future_horizon=future_horizon,
            action_dim=7,
            visual_encoder=visual_encoder,
            include_current_latent=include_current_latent,
            include_future_latents=include_future_latents,
        )
        val_dataset = make_mock_action_dataset(
            trajectory_id="mock_val_0",
            split="val",
            length=length,
            history_len=history_len,
            action_horizon=action_horizon,
            future_horizon=future_horizon,
            action_dim=7,
            visual_encoder=visual_encoder,
            include_current_latent=include_current_latent,
            include_future_latents=include_future_latents,
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
        normalization_stats = no_normalization_record()
        if visual_encoder is not None:
            normalization_stats["visual_latents"] = {
                "mode": "frozen_encoder_no_fitted_stats",
                "source_split": "none",
                "target_only_future": include_future_latents,
                "current_latent_input": include_current_latent,
                "encoder": visual_encoder.metadata(),
            }
        return train_dataset, val_dataset, metadata, None, normalization_stats

    if include_current_latent or include_future_latents:
        # Check if pre-extracted latents are available
        latent_dir = config["data"].get("latent_dir")
        if not latent_dir:
            raise NotImplementedError(
                "real-data WAM training requires precomputed frozen visual latents or "
                "a real FrozenVisualEncoderAdapter integration; dry_run supports the "
                "smoke time-index encoder only. Set data.latent_dir to use pre-extracted latents."
            )

    trajectories, metadata = load_real_libero_trajectories(config)
    action_transform, normalization_stats = build_action_transform(trajectories, config)
    if action_transform is not None:
        trajectories = apply_action_transform(trajectories, action_transform)

    # Add visual latents normalization record if using pre-extracted latents
    if (include_current_latent or include_future_latents) and config["data"].get("latent_dir"):
        normalization_stats.update(
            preextracted_latents_record(
                config,
                include_current_latent=include_current_latent,
                include_future_latents=include_future_latents,
            )
        )

    train_dataset = TrajectoryWindowDataset(
        trajectories,
        split="train",
        history_len=history_len,
        action_horizon=action_horizon,
        future_horizon=future_horizon,
        include_current_latent=include_current_latent,
        include_future_latents=include_future_latents,
    )
    val_dataset = TrajectoryWindowDataset(
        trajectories,
        split="val",
        history_len=history_len,
        action_horizon=action_horizon,
        future_horizon=future_horizon,
        include_current_latent=include_current_latent,
        include_future_latents=include_future_latents,
    )
    if len(train_dataset) == 0:
        raise ValueError("real LIBERO train split produced zero valid windows")
    if len(val_dataset) == 0:
        raise ValueError("real LIBERO val split produced zero valid windows")
    return train_dataset, val_dataset, metadata, action_transform, normalization_stats


def requires_future_latents(config: Mapping[str, Any]) -> bool:
    """Return whether this run needs future frozen visual latent targets."""

    return (
        str(config["model"]["temporal_adapter"]) == "wam_gru"
        or float(config["training"]["lambda_future"]) > 0.0
    )


def uses_current_latent(config: Mapping[str, Any]) -> bool:
    """Return whether this run uses the current frozen visual latent as input."""

    adapter = str(config["model"]["temporal_adapter"])
    return adapter in {"wam_gru", "bc_gru"} or float(config["training"]["lambda_future"]) > 0.0


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
        "visual_latents": {"mode": "not_used"},
    }


def preextracted_latents_record(
    config: Mapping[str, Any],
    *,
    include_current_latent: bool = True,
    include_future_latents: bool = True,
) -> dict[str, Any]:
    """Create normalization record for pre-extracted latents."""
    return {
        "visual_latents": {
            "mode": "preextracted",
            "encoder": config["model"].get("visual_encoder", "unknown"),
            "model_id": config["model"].get("model_id", "unknown"),
            "revision": config["model"].get("revision", "unknown"),
            "output_token": config["model"].get("output_token", "cls"),
            "latent_dim": config["data"].get("latent_dim", 384),
            "source_split": "train",
            "target_only_future": include_future_latents,
            "current_latent_input": include_current_latent,
        }
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
        "visual_latents": {"mode": "not_used"},
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
    visual_encoder: Any | None = None,
    include_current_latent: bool = False,
    include_future_latents: bool = False,
) -> TrajectoryWindowDataset:
    """Create a deterministic mock dataset for action-only or WAM smoke training."""

    actions = [
        [float(timestep) + 0.01 * float(dim) for dim in range(action_dim)]
        for timestep in range(length)
    ]
    states = [
        [float(timestep) + 0.001 * float(dim) for dim in range(9)]
        for timestep in range(length)
    ]
    frame_refs = [f"{trajectory_id}:frame:{timestep}" for timestep in range(length)]
    visual_latents = None
    if include_current_latent or include_future_latents:
        if visual_encoder is None:
            raise ValueError("latent fields require a visual_encoder")
        visual_latents = encode_sequence(visual_encoder, frame_refs)
    trajectory = RawTrajectory(
        images=frame_refs,
        actions=actions,
        states=states,
        visual_latents=visual_latents,
        task_id=0,
        task_name="mock_task",
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
        include_current_latent=include_current_latent,
        include_future_latents=include_future_latents,
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
    task_names = [task_name_from_file(path) for path in files]
    task_id_by_name = {name: index for index, name in enumerate(sorted(set(task_names)))}

    # Check if we should load pre-extracted latents
    latent_dir = config["data"].get("latent_dir")
    latent_format = config["data"].get("latent_format", "hdf5")

    trajectories: list[RawTrajectory] = []
    for file_path in files:
        task_name = task_name_from_file(file_path)
        task_id = task_id_by_name[task_name]
        with h5py.File(file_path, "r") as handle:
            for demo_path, group in list_demo_groups(handle):
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
                states = load_proprio_states(group, expected_length=length)

                # Load pre-extracted latents if available
                visual_latents = None
                if latent_dir:
                    visual_latents = load_preextracted_latents(
                        latent_dir, file_path, demo_path, latent_format
                    )

                trajectories.append(
                    RawTrajectory(
                        images=frame_refs,
                        actions=actions,
                        states=states,
                        visual_latents=visual_latents,
                        task_id=task_id,
                        task_name=task_name,
                        frame_refs=frame_refs,
                        language=extract_language(handle, group),
                        trajectory_id=trajectory_id,
                        split="unspecified",
                    )
                )

    if not trajectories:
        raise ValueError(f"no action trajectories found under {dataset_root}")
    split_trajectories, split_metadata = assign_splits(trajectories, config)
    split_metadata["task_id_map"] = task_id_by_name
    return split_trajectories, split_metadata


def task_name_from_file(path: Path) -> str:
    """Return stable LIBERO task name from a demonstration HDF5 file path."""

    return path.stem.removesuffix("_demo")


def load_proprio_states(group: Any, *, expected_length: int) -> Any | None:
    """Load current proprio/state input with shape `[T, state_dim]` when present."""

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - optional environment dependency.
        return None

    if "robot_states" in group:
        states = np.asarray(group["robot_states"][()], dtype=np.float32)
    elif "obs" in group and "ee_states" in group["obs"] and "gripper_states" in group["obs"]:
        ee_states = np.asarray(group["obs"]["ee_states"][()], dtype=np.float32)
        gripper_states = np.asarray(group["obs"]["gripper_states"][()], dtype=np.float32)
        states = np.concatenate([ee_states, gripper_states], axis=-1)
    else:
        return None
    if states.ndim != 2 or int(states.shape[0]) != expected_length:
        raise ValueError(
            "proprio states must have shape [T, state_dim], "
            f"got {tuple(states.shape)} for expected length {expected_length}"
        )
    return states


def load_preextracted_latents(
    latent_dir: Path | str,
    source_file: Path,
    demo_path: str,
    latent_format: str = "hdf5",
) -> list[list[float]] | None:
    """Load pre-extracted latents from HDF5 or Zarr storage."""

    latent_dir = Path(latent_dir)
    source_name = source_file.stem

    if latent_format == "hdf5":
        latent_file = latent_dir / f"{source_name}_dinov2_vits14.hdf5"
        if not latent_file.exists():
            return None

        try:
            import h5py
            import numpy as np
        except ImportError:
            return None

        with h5py.File(latent_file, "r") as f:
            if demo_path not in f:
                return None
            demo_group = f[demo_path]
            if "latents" not in demo_group:
                return None
            latents = np.array(demo_group["latents"])
            return latents.tolist()

    elif latent_format == "zarr":
        try:
            import zarr
            import numpy as np
        except ImportError:
            return None

        latent_file = latent_dir / f"{source_name}_dinov2_vits14.zarr"
        if not latent_file.exists():
            return None

        store = zarr.open(str(latent_file), mode="r")
        if demo_path not in store:
            return None
        latents = np.array(store[demo_path]["latents"])
        return latents.tolist()

    return None


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


def list_demo_groups(handle: Any) -> list[tuple[str, Any]]:
    """Return a sorted list of ``(path, group)`` demo groups from an HDF5 handle."""
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
        if split not in limits:
            raise ValueError(
                f"trajectory {trajectory.trajectory_id} has unrecognized split={split!r}; "
                f"expected one of {sorted(limits)}"
            )
        limit = limits[split]
        if limit is not None and counts[split] >= limit:
            continue
        output.append(trajectory)
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
    """Collate fields into tensors.

    Output shapes:

    - `action_history`: `[B, history_len, action_dim]`.
    - `target_actions`: `[B, action_horizon, action_dim]`.
    - optional `optional_state_t`: `[B, state_dim]`.
    - optional `z_t`: `[B, latent_dim]`.
    - optional `task_id`: `[B]`.
    - optional `target_future_latents`: `[B, future_horizon, latent_dim]`.
    """

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
    batch: dict[str, Any] = {
        "action_history": action_history,
        "target_actions": target_actions,
        "trajectory_id": [sample["trajectory_id"] for sample in samples],
        "time_index": [sample["time_index"] for sample in samples],
        "language": [sample.get("language", "") for sample in samples],
        "task_name": [sample.get("task_name", "") for sample in samples],
    }
    if all(sample.get("optional_state_t") is not None for sample in samples):
        batch["optional_state_t"] = torch.stack(
            [
                torch.as_tensor(sample["optional_state_t"], dtype=torch.float32)
                for sample in samples
            ],
            dim=0,
        )
    if all(sample.get("z_t") is not None for sample in samples):
        batch["z_t"] = torch.stack(
            [torch.as_tensor(sample["z_t"], dtype=torch.float32) for sample in samples],
            dim=0,
        )
    if all(sample.get("task_id") is not None for sample in samples):
        batch["task_id"] = torch.as_tensor(
            [int(sample["task_id"]) for sample in samples],
            dtype=torch.long,
        )
    if all("target_future_latents" in sample for sample in samples):
        batch["target_future_latents"] = torch.stack(
            [
                torch.as_tensor(sample["target_future_latents"], dtype=torch.float32)
                for sample in samples
            ],
            dim=0,
        )
    if all(sample.get("z_t_patch") is not None for sample in samples):
        batch["z_t_patch"] = torch.stack(
            [
                torch.as_tensor(sample["z_t_patch"], dtype=torch.float32)
                for sample in samples
            ],
            dim=0,
        )
    if all("target_future_patch_latents" in sample for sample in samples):
        batch["target_future_patch_latents"] = torch.stack(
            [
                torch.as_tensor(
                    sample["target_future_patch_latents"], dtype=torch.float32
                )
                for sample in samples
            ],
            dim=0,
        )
    return batch


def infer_action_dim(sample: Mapping[str, Any]) -> int:
    target = torch.as_tensor(sample["target_actions"])
    if target.ndim != 2:
        raise ValueError(f"target_actions must have shape [H, A], got {tuple(target.shape)}")
    return int(target.shape[-1])


def infer_state_dim(sample: Mapping[str, Any]) -> int | None:
    state = sample.get("optional_state_t")
    if state is None:
        return None
    tensor = torch.as_tensor(state)
    if tensor.ndim != 1:
        raise ValueError(f"optional_state_t must have shape [S], got {tuple(tensor.shape)}")
    return int(tensor.shape[-1])


def infer_task_count(config: Mapping[str, Any], split_metadata: Mapping[str, Any]) -> int:
    configured = config["model"].get("num_tasks")
    if configured is not None:
        count = int(configured)
    else:
        task_id_map = split_metadata.get("task_id_map", {})
        count = len(task_id_map) if isinstance(task_id_map, Mapping) else 0
    return max(1, count)


def has_current_latent(sample: Mapping[str, Any]) -> bool:
    return sample.get("z_t") is not None


def infer_current_latent_dim(sample: Mapping[str, Any]) -> int:
    latent = torch.as_tensor(sample["z_t"])
    if latent.ndim != 1:
        raise ValueError(f"z_t must have shape [Z], got {tuple(latent.shape)}")
    return int(latent.shape[-1])


def has_future_latent_targets(sample: Mapping[str, Any]) -> bool:
    return "target_future_latents" in sample


def infer_latent_dim(sample: Mapping[str, Any]) -> int:
    target = torch.as_tensor(sample["target_future_latents"])
    if target.ndim != 2:
        raise ValueError(
            "target_future_latents must have shape [H, Z], "
            f"got {tuple(target.shape)}"
        )
    return int(target.shape[-1])


def has_current_patch_latent(sample: Mapping[str, Any]) -> bool:
    return sample.get("z_t_patch") is not None


def has_future_patch_latent_targets(sample: Mapping[str, Any]) -> bool:
    return "target_future_patch_latents" in sample


def infer_patch_latent_dims(sample: Mapping[str, Any]) -> tuple[int, int]:
    """Return (num_patches, feature_dim) from a sample's z_t_patch."""
    patch = torch.as_tensor(sample["z_t_patch"])
    if patch.ndim != 2:
        raise ValueError(f"z_t_patch must have shape [N, D], got {tuple(patch.shape)}")
    return int(patch.shape[0]), int(patch.shape[-1])


def pool_patch_latents(patch_latents: torch.Tensor) -> torch.Tensor:
    """Mean-pool patch latents from [B, N, D] to [B, D].

    This is the compatibility smoke path for existing WAM-GRU which expects
    a flat [B, Z] latent. It is NOT full DINO-WM spatial modeling.
    """
    if patch_latents.ndim == 3:
        return patch_latents.mean(dim=1)
    if patch_latents.ndim == 4:
        # [B, H, N, D] -> [B, H, D]
        return patch_latents.mean(dim=2)
    raise ValueError(
        f"patch_latents must be [B, N, D] or [B, H, N, D], "
        f"got {tuple(patch_latents.shape)}"
    )


def forward_offline_model(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor | Mapping[str, torch.Tensor]:
    """Forward a batch through an offline model using only declared inputs."""

    action_history = batch["action_history"].to(device)
    if getattr(model, "uses_proprio_task", False):
        missing = [
            name
            for name in ("z_t", "optional_state_t", "task_id")
            if name not in batch
        ]
        if missing:
            raise ValueError(f"BC-GRU batch is missing required inputs: {missing}")
        return model(
            action_history,
            batch["z_t"].to(device),
            batch["optional_state_t"].to(device),
            batch["task_id"].to(device),
        )
    if "z_t" in batch:
        return model(action_history, batch["z_t"].to(device))
    if "z_t_patch" in batch:
        # Mean-patch pooling compatibility path: project [B, N, D] -> [B, D]
        z_pooled = pool_patch_latents(batch["z_t_patch"].to(device))
        return model(action_history, z_pooled)
    return model(action_history)


def run_one_split(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lambda_action: float,
    lambda_future: float,
    grad_clip_norm: float | None,
    max_steps: int | None,
    action_transform: ActionTransform | None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    squared_error_sum = 0.0
    element_count = 0
    total_loss_sum = 0.0
    action_loss_sum = 0.0
    future_loss_sum = 0.0
    loss_weight_sum = 0
    future_error_sum = 0.0
    future_error_count = 0
    future_error_by_horizon_sum: torch.Tensor | None = None
    future_error_by_horizon_count: torch.Tensor | None = None
    future_mse_sum = 0.0
    future_mse_count = 0
    future_mse_by_horizon_sum: torch.Tensor | None = None
    future_mse_by_horizon_count: torch.Tensor | None = None
    action_mse_by_horizon_sum: torch.Tensor | None = None
    action_mse_by_horizon_count: torch.Tensor | None = None
    action_mse_by_dim_sum: torch.Tensor | None = None
    action_mse_by_dim_count: torch.Tensor | None = None
    patch_mse_by_horizon_sum: torch.Tensor | None = None
    patch_mse_by_horizon_count: torch.Tensor | None = None
    patch_mse_metric = 0.0
    patch_cosine_err_metric = 0.0
    patch_mean_cosine_err_metric = 0.0

    steps = 0
    samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            if max_steps is not None and steps >= max_steps:
                break
            action_history = batch["action_history"].to(device)
            target_actions = batch["target_actions"].to(device)
            target_future_latents = None
            target_future_patch_latents = None
            if "target_future_latents" in batch:
                target_future_latents = batch["target_future_latents"].to(device)
            if "target_future_patch_latents" in batch:
                target_future_patch_latents = batch["target_future_patch_latents"].to(
                    device
                )
                # For WAM-GRU future loss, pool patch latents to flat [B, H, D]
                if target_future_latents is None:
                    target_future_latents = pool_patch_latents(
                        target_future_patch_latents
                    )

            outputs = forward_offline_model(model, batch, device=device)
            if isinstance(outputs, Mapping):
                pred_actions = outputs["pred_actions"]
                pred_future_latents = outputs.get("pred_future_latents")
            else:
                pred_actions = outputs
                pred_future_latents = None
            if isinstance(outputs, Mapping) and "pred_gripper_logits" in outputs:
                if action_transform is not None:
                    raise ValueError(
                        "split gripper head currently requires raw action targets "
                        "(normalization.actions.mode=none)"
                    )
                action_loss = split_gripper_action_loss(outputs, target_actions)
            else:
                action_loss = action_mse(pred_actions, target_actions)
            future_loss = torch.zeros((), dtype=action_loss.dtype, device=device)
            if target_future_latents is not None:
                if pred_future_latents is None:
                    raise ValueError("batch has future latent targets but model returned none")
                future_loss = future_latent_cosine_error(
                    pred_future_latents,
                    target_future_latents,
                )
            elif lambda_future != 0.0:
                raise ValueError("lambda_future is nonzero but batch has no future latents")

            total_loss = lambda_action * action_loss + lambda_future * future_loss
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
            batch_size = int(target_actions.shape[0])
            action_loss_sum += float(action_loss.detach().item()) * batch_size
            future_loss_sum += float(future_loss.detach().item()) * batch_size
            total_loss_sum += float(total_loss.detach().item()) * batch_size
            loss_weight_sum += batch_size

            # Per-horizon action MSE
            action_mse_horizon = action_mse_per_horizon(metric_pred, metric_target)
            horizon_size = int(action_mse_horizon.shape[0])
            if action_mse_by_horizon_sum is None:
                action_mse_by_horizon_sum = action_mse_horizon.detach().cpu()
                action_mse_by_horizon_count = torch.full_like(action_mse_by_horizon_sum, batch_size)
            else:
                action_mse_by_horizon_sum += action_mse_horizon.detach().cpu() * batch_size
                action_mse_by_horizon_count += batch_size

            # Per-dimension action MSE
            action_mse_dim = action_mse_per_dimension(metric_pred, metric_target)
            if action_mse_by_dim_sum is None:
                action_mse_by_dim_sum = action_mse_dim.detach().cpu()
                action_mse_by_dim_count = torch.full_like(action_mse_by_dim_sum, batch_size)
            else:
                action_mse_by_dim_sum += action_mse_dim.detach().cpu() * batch_size
                action_mse_by_dim_count += batch_size

            if target_future_latents is not None and pred_future_latents is not None:
                future_errors = future_latent_cosine_error(
                    pred_future_latents.detach(),
                    target_future_latents,
                    reduction="none",
                )
                future_error_sum += future_errors.sum().item()
                future_error_count += int(future_errors.numel())
                horizon_sum = future_errors.sum(dim=0).detach().cpu()
                horizon_count = torch.full_like(horizon_sum, future_errors.shape[0])
                if future_error_by_horizon_sum is None:
                    future_error_by_horizon_sum = horizon_sum
                    future_error_by_horizon_count = horizon_count
                else:
                    future_error_by_horizon_sum += horizon_sum
                    future_error_by_horizon_count += horizon_count

                # Future latent MSE
                future_mse_errors = future_latent_mse(
                    pred_future_latents.detach(),
                    target_future_latents,
                    reduction="none",
                )
                future_mse_sum += future_mse_errors.sum().item()
                future_mse_count += int(future_mse_errors.numel())
                mse_horizon_sum = future_mse_errors.sum(dim=0).detach().cpu()
                mse_horizon_count = torch.full_like(mse_horizon_sum, future_mse_errors.shape[0])
                if future_mse_by_horizon_sum is None:
                    future_mse_by_horizon_sum = mse_horizon_sum
                    future_mse_by_horizon_count = mse_horizon_count
                else:
                    future_mse_by_horizon_sum += mse_horizon_sum
                    future_mse_by_horizon_count += mse_horizon_count

            # Patch latent metrics (computed on raw patch targets, not pooled)
            if (
                target_future_patch_latents is not None
                and pred_future_latents is not None
            ):
                # Pool predictions to match patch target shape for metric:
                # pred is [B, H, D], target is [B, H, N, D]
                # We expand pred to [B, H, 1, D] for patch_mse / patch_cosine_error
                pred_expanded = pred_future_latents.unsqueeze(2).expand_as(
                    target_future_patch_latents
                )
                p_mse = patch_mse(
                    pred_expanded.detach(),
                    target_future_patch_latents,
                    reduction="none",
                )
                p_cos = patch_cosine_error(
                    pred_expanded.detach(),
                    target_future_patch_latents,
                    reduction="none",
                )
                p_mean_cos = patch_mean_cosine_error(
                    pred_expanded.detach(),
                    target_future_patch_latents,
                    reduction="none",
                )
                patch_mse_metric += p_mse.sum().item()
                patch_cosine_err_metric += p_cos.sum().item()
                patch_mean_cosine_err_metric += p_mean_cos.sum().item()
                # Per-horizon accumulation for patch_mse
                p_mse_horizon = p_mse.sum(dim=0).detach().cpu()
                p_mse_hcount = torch.full_like(p_mse_horizon, p_mse.shape[0])
                if patch_mse_by_horizon_sum is None:
                    patch_mse_by_horizon_sum = p_mse_horizon
                    patch_mse_by_horizon_count = p_mse_hcount
                else:
                    patch_mse_by_horizon_sum += p_mse_horizon
                    patch_mse_by_horizon_count += p_mse_hcount

            steps += 1
            samples += int(target_actions.shape[0])

    if steps == 0 or element_count == 0 or loss_weight_sum == 0:
        raise ValueError("split produced no batches")
    action_mse_value = squared_error_sum / element_count
    future_metric = 0.0
    future_by_horizon: list[float] = []
    if future_error_count > 0:
        future_metric = future_error_sum / future_error_count
        if future_error_by_horizon_sum is None or future_error_by_horizon_count is None:
            raise RuntimeError("future horizon accumulators unexpectedly missing")
        future_by_horizon = (
            future_error_by_horizon_sum / future_error_by_horizon_count
        ).tolist()

    future_mse_metric = 0.0
    future_mse_by_horizon: list[float] = []
    if future_mse_count > 0:
        future_mse_metric = future_mse_sum / future_mse_count
        if future_mse_by_horizon_sum is not None and future_mse_by_horizon_count is not None:
            future_mse_by_horizon = (
                future_mse_by_horizon_sum / future_mse_by_horizon_count
            ).tolist()

    # Finalize patch metrics
    patch_mse_count = loss_weight_sum  # same number of batches
    patch_mse_value = patch_mse_metric / max(patch_mse_count, 1)
    patch_cosine_err_value = patch_cosine_err_metric / max(patch_mse_count, 1)
    patch_mean_cosine_err_value = patch_mean_cosine_err_metric / max(patch_mse_count, 1)
    patch_mse_by_horizon: list[float] = []
    if patch_mse_by_horizon_sum is not None and patch_mse_by_horizon_count is not None:
        patch_mse_by_horizon = (
            patch_mse_by_horizon_sum / patch_mse_by_horizon_count
        ).tolist()

    action_mse_by_horizon: list[float] = []
    if action_mse_by_horizon_sum is not None and action_mse_by_horizon_count is not None:
        action_mse_by_horizon = (action_mse_by_horizon_sum / action_mse_by_horizon_count).tolist()

    action_mse_by_dim: list[float] = []
    if action_mse_by_dim_sum is not None and action_mse_by_dim_count is not None:
        action_mse_by_dim = (action_mse_by_dim_sum / action_mse_by_dim_count).tolist()

    return {
        "total_loss": total_loss_sum / loss_weight_sum,
        "action_loss": action_loss_sum / loss_weight_sum,
        "action_loss_units": (
            "normalized_action_units"
            if action_transform is not None
            else "raw_action_units"
        ),
        "future_loss": future_loss_sum / loss_weight_sum,
        "future_latent_cosine_error": future_metric,
        "future_latent_cosine_error_by_horizon": future_by_horizon,
        "future_latent_mse": future_mse_metric,
        "future_latent_mse_by_horizon": future_mse_by_horizon,
        "patch_mse": patch_mse_value,
        "patch_cosine_error": patch_cosine_err_value,
        "patch_mean_cosine_error": patch_mean_cosine_err_value,
        "patch_mse_by_horizon": patch_mse_by_horizon,
        "action_mse_by_horizon": action_mse_by_horizon,
        "action_mse_by_dimension": action_mse_by_dim,
        "spike_loss": 0.0,
        "action_mse": action_mse_value,
        "steps": steps,
        "samples": samples,
    }


def split_gripper_action_loss(
    outputs: Mapping[str, torch.Tensor],
    target_actions: torch.Tensor,
) -> torch.Tensor:
    """Continuous SmoothL1 for dims 0-5 plus BCE gripper classification."""

    if "pred_continuous_actions" not in outputs or "pred_gripper_logits" not in outputs:
        raise ValueError("split gripper outputs are missing continuous actions or logits")
    continuous_target = target_actions[..., :-1]
    gripper_target = (target_actions[..., -1] > 0).to(target_actions.dtype)
    continuous_loss = torch.nn.functional.smooth_l1_loss(
        outputs["pred_continuous_actions"],
        continuous_target,
    )
    gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["pred_gripper_logits"],
        gripper_target,
    )
    return continuous_loss + gripper_loss


def format_metric_row(
    epoch: int,
    split: str,
    metrics: Mapping[str, Any],
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
        "future_latent_cosine_error": format_float(
            float(metrics["future_latent_cosine_error"])
        ),
        "future_latent_cosine_error_by_horizon": json.dumps(
            [
                round(float(value), 10)
                for value in metrics["future_latent_cosine_error_by_horizon"]
            ]
        ),
        "future_latent_mse": format_float(float(metrics.get("future_latent_mse", 0.0))),
        "future_latent_mse_by_horizon": json.dumps(
            [
                round(float(value), 10)
                for value in metrics.get("future_latent_mse_by_horizon", [])
            ]
        ),
        "patch_mse": format_float(float(metrics.get("patch_mse", 0.0))),
        "patch_cosine_error": format_float(
            float(metrics.get("patch_cosine_error", 0.0))
        ),
        "patch_mean_cosine_error": format_float(
            float(metrics.get("patch_mean_cosine_error", 0.0))
        ),
        "patch_mse_by_horizon": json.dumps(
            [
                round(float(value), 10)
                for value in metrics.get("patch_mse_by_horizon", [])
            ]
        ),
        "action_mse_by_horizon": json.dumps(
            [
                round(float(value), 10)
                for value in metrics.get("action_mse_by_horizon", [])
            ]
        ),
        "action_mse_by_dimension": json.dumps(
            [
                round(float(value), 10)
                for value in metrics.get("action_mse_by_dimension", [])
            ]
        ),
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


def checkpoint_selection_metric(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> float:
    """Return the validation metric selected by `output.save_best_by`."""

    metric_name = str(config["output"]["save_best_by"]).split("/", maxsplit=1)[-1]
    if metric_name not in metrics:
        raise ValueError(
            f"output.save_best_by requested {metric_name!r}, "
            f"available metrics are {sorted(metrics)}"
        )
    return float(metrics[metric_name])


def append_train_log(path: Path, row: Mapping[str, Any]) -> None:
    """Append one compact human-readable loss row to `train.log`."""

    line = ",".join(
        [
            str(row["epoch"]),
            str(row["split"]),
            str(row["total_loss"]),
            str(row["action_loss"]),
            str(row["future_loss"]),
            str(row["action_mse"]),
            str(row["future_latent_cosine_error"]),
        ]
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


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
    latent_dim: int | None,
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
            "latent_dim": latent_dim,
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
            "best_metric_name": config["output"]["save_best_by"],
            "best_metric": best_metric,
            "best_val_action_mse": (
                best_metric
                if str(config["output"]["save_best_by"]).endswith("/action_mse")
                else None
            ),
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
            "last_train_future_latent_cosine_error": (
                float(train_rows[-1]["future_latent_cosine_error"])
                if train_rows
                else None
            ),
            "last_val_future_latent_cosine_error": (
                float(val_rows[-1]["future_latent_cosine_error"]) if val_rows else None
            ),
            "action_mse_units": "raw_action_units",
            "lower_is_better": True,
        },
        "non_claims": [
            "not_reportable_wam_result",
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
