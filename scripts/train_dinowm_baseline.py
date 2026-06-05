#!/usr/bin/env python3
"""Standalone training script for DINOwM Transformer on cached patch latents.

Uses PatchLatentTransitionDataset directly with the DINOwMTransformer model.
Writes full reproducibility artifacts.

Usage:
    python scripts/train_dinowm_baseline.py --config configs/reportable/dinowm_baseline_real.yaml
    python scripts/train_dinowm_baseline.py --config configs/reportable/dinowm_baseline_real.yaml --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.patch_latent_dataset import (  # noqa: E402
    PatchLatentTransitionDataset,
    build_trajectory_split_indices,
    create_dinowm_transition_dataset,
)
from src.models.dinowm_transformer import DINOwMTransformer, build_dinowm_model  # # noqa: E402
from src.train.metrics import (  # noqa: E402
    action_mse,
    action_mse_per_horizon,
    patch_cosine_error,
    patch_mean_cosine_error,
    patch_mse,
)
from src.utils.seed import seed_everything  # # noqa: E402


METRIC_FIELDNAMES = [
    "epoch",
    "split",
    "total_loss",
    "patch_cosine_loss",
    "action_loss",
    "patch_mse",
    "patch_cosine_error",
    "patch_mean_cosine_error",
    "patch_cosine_error_by_horizon",
    "patch_mse_by_horizon",
    "action_mse",
    "action_mse_by_horizon",
    "steps",
    "samples",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Training YAML config.")
    parser.add_argument("--dry_run", action="store_true", help="Use tiny mock data for code validation.")
    parser.add_argument("--max_steps", type=int, default=None, help="Max optimizer steps per epoch.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override config output dir.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def capture_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def capture_git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True
        ).strip()
        return {"commit": commit, "dirty": len(dirty) > 0}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def patch_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate PatchLatentTransitionDataset samples into batched tensors."""
    z_context = torch.stack([s["z_context"] for s in samples], dim=0)
    actions = torch.stack([s["actions"] for s in samples], dim=0)
    future_actions = torch.stack([s["future_actions"] for s in samples], dim=0)
    z_target = torch.stack([s["z_target"] for s in samples], dim=0)
    return {
        "z_context": z_context,
        "actions": actions,
        "future_actions": future_actions,
        "z_target": z_target,
    }


def build_datasets(
    config: Mapping[str, Any], *, dry_run: bool
) -> tuple[Subset, Subset, dict[str, Any]]:
    """Build train/val datasets from cached patch latents."""
    cache_dir = Path(config["data"]["cache_dir"])
    context_len = int(config["data"]["context_len"])
    future_horizon = int(config["data"]["future_horizon"])

    if dry_run:
        # Create a tiny synthetic dataset for code validation
        dataset = _make_mock_dataset(
            context_len=context_len,
            future_horizon=future_horizon,
            patch_dim=int(config["data"]["patch_dim"]),
            feature_dim=int(config["data"]["feature_dim"]),
            action_dim=int(config["data"]["action_dim"]),
            n_samples=32,
        )
        split_info = {
            "method": "mock_dry_run",
            "train_count": 24,
            "val_count": 8,
            "mock": True,
        }
        trainSubset = Subset(dataset, list(range(24)))
        valSubset = Subset(dataset, list(range(24, 32)))
        return trainSubset, valSubset, split_info

    full_dataset = create_dinowm_transition_dataset(
        cache_dir,
        context_len=context_len,
        future_horizon=future_horizon,
        max_demos=config["data"].get("max_demos"),
        max_frames=config["data"].get("max_frames"),
        split="train",
    )

    train_ratio = float(config["data"].get("train_ratio", 0.9))
    seed = int(config["experiment"]["seed"])
    train_indices, val_indices, split_info = build_trajectory_split_indices(
        full_dataset,
        train_ratio=train_ratio,
        seed=seed,
    )

    return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices), split_info


def _make_mock_dataset(
    *,
    context_len: int,
    future_horizon: int,
    patch_dim: int,
    feature_dim: int,
    action_dim: int,
    n_samples: int,
) -> list[dict[str, Any]]:
    """Create synthetic samples matching PatchLatentTransitionDataset output."""
    samples = []
    for i in range(n_samples):
        samples.append({
            "z_context": torch.randn(context_len, patch_dim, feature_dim),
            "actions": torch.randn(context_len, action_dim),
            "future_actions": torch.randn(future_horizon, action_dim),
            "z_target": torch.randn(future_horizon, patch_dim, feature_dim),
            "metadata": {"trajectory_id": f"mock_{i}", "time_index": i, "split": "train"},
        })
    return samples


def run_one_split(
    model: DINOwMTransformer,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lambda_patch_cosine: float,
    lambda_action: float,
    grad_clip_norm: float | None,
    max_steps: int | None,
) -> dict[str, Any]:
    """Run one train/val split and return aggregated metrics."""
    is_train = optimizer is not None
    model.train(is_train)

    patch_cosine_sum = 0.0
    action_loss_sum = 0.0
    total_loss_sum = 0.0
    patch_mse_sum = 0.0
    patch_cosine_err_sum = 0.0
    patch_mean_cosine_err_sum = 0.0
    action_mse_sum = 0.0
    patch_cosine_by_horizon_sum: torch.Tensor | None = None
    patch_mse_by_horizon_sum: torch.Tensor | None = None
    action_mse_by_horizon_sum: torch.Tensor | None = None
    steps = 0
    samples = 0
    metric_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            if max_steps is not None and steps >= max_steps:
                break

            z_context = batch["z_context"].to(device)  # [B, T_ctx, P, D]
            actions = batch["actions"].to(device)  # [B, T_ctx, A]
            future_actions = batch["future_actions"].to(device)  # [B, H, A]
            z_target = batch["z_target"].to(device)  # [B, H, P, D]

            # Forward: model predicts [B, H, P, D] from context latents,
            # context action history, and future candidate actions [B, H, A].
            pred = model(z_context, actions, future_actions=future_actions)  # [B, H, P, D]

            # Losses
            patch_cosine_loss = patch_cosine_error(pred, z_target, reduction="mean")
            # Action MSE: compare predicted future actions? No -- model doesn't predict actions.
            # Use patch_cosine as primary loss. Action loss is 0 for world model.
            action_loss = torch.zeros((), device=device)

            total_loss = lambda_patch_cosine * patch_cosine_loss + lambda_action * action_loss

            if not torch.isfinite(total_loss):
                raise FloatingPointError("non-finite training loss")

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

            B = int(z_context.shape[0])
            total_loss_sum += float(total_loss.detach().item()) * B
            patch_cosine_sum += float(patch_cosine_loss.detach().item()) * B
            action_loss_sum += float(action_loss.detach().item()) * B

            # Patch metrics
            p_mse = patch_mse(pred.detach(), z_target, reduction="none")
            p_cos = patch_cosine_error(pred.detach(), z_target, reduction="none")
            p_mean_cos = patch_mean_cosine_error(pred.detach(), z_target, reduction="none")

            metric_count += int(p_mse.numel())
            patch_mse_sum += float(p_mse.sum().item())
            patch_cosine_err_sum += float(p_cos.sum().item())
            patch_mean_cosine_err_sum += float(p_mean_cos.sum().item())

            # Per-horizon accumulation
            p_cos_h = p_cos.mean(dim=0).detach().cpu()  # [H]
            p_mse_h = p_mse.mean(dim=0).detach().cpu()  # [H]
            if patch_cosine_by_horizon_sum is None:
                patch_cosine_by_horizon_sum = p_cos_h * B
                patch_mse_by_horizon_sum = p_mse_h * B
            else:
                patch_cosine_by_horizon_sum += p_cos_h * B
                patch_mse_by_horizon_sum += p_mse_h * B

            steps += 1
            samples += B

    if steps == 0:
        raise ValueError("split produced no batches")

    patch_cosine_by_horizon: list[float] = []
    patch_mse_by_horizon: list[float] = []
    if patch_cosine_by_horizon_sum is not None:
        patch_cosine_by_horizon = (patch_cosine_by_horizon_sum / samples).tolist()
    if patch_mse_by_horizon_sum is not None:
        patch_mse_by_horizon = (patch_mse_by_horizon_sum / samples).tolist()

    return {
        "total_loss": total_loss_sum / samples,
        "patch_cosine_loss": patch_cosine_sum / samples,
        "action_loss": action_loss_sum / samples,
        "patch_mse": patch_mse_sum / metric_count,
        "patch_cosine_error": patch_cosine_err_sum / metric_count,
        "patch_mean_cosine_error": patch_mean_cosine_err_sum / metric_count,
        "patch_cosine_error_by_horizon": patch_cosine_by_horizon,
        "patch_mse_by_horizon": patch_mse_by_horizon,
        "action_mse": action_mse_sum / samples if action_mse_sum > 0 else 0.0,
        "action_mse_by_horizon": [],
        "steps": steps,
        "samples": samples,
    }


def format_metric_row(epoch: int, split: str, metrics: dict[str, Any]) -> dict[str, str]:
    """Format metrics into a CSV row dict."""
    row = {
        "epoch": str(epoch),
        "split": split,
        "total_loss": f"{metrics['total_loss']:.6f}",
        "patch_cosine_loss": f"{metrics['patch_cosine_loss']:.6f}",
        "action_loss": f"{metrics['action_loss']:.6f}",
        "patch_mse": f"{metrics['patch_mse']:.6f}",
        "patch_cosine_error": f"{metrics['patch_cosine_error']:.6f}",
        "patch_mean_cosine_error": f"{metrics['patch_mean_cosine_error']:.6f}",
        "patch_cosine_error_by_horizon": json.dumps(metrics["patch_cosine_error_by_horizon"]),
        "patch_mse_by_horizon": json.dumps(metrics["patch_mse_by_horizon"]),
        "action_mse": f"{metrics['action_mse']:.6f}",
        "action_mse_by_horizon": "[]",
        "steps": str(metrics["steps"]),
        "samples": str(metrics["samples"]),
    }
    return row


def write_reproducibility_files(run_dir: Path, config: dict[str, Any], argv: list[str]) -> None:
    """Write standard reproducibility artifacts."""
    write_json(run_dir / "config.yaml", config)
    command = " ".join(shlex.quote(str(item)) for item in argv)
    (run_dir / "command.sh").write_text(command + "\n")
    git_info = capture_git_info()
    (run_dir / "git_commit.txt").write_text(
        f"commit: {git_info['commit']}\ndirty: {git_info['dirty']}\n"
    )
    environment = capture_environment()
    seeds = {
        "seed": config["experiment"]["seed"],
        "torch_manual_seed": config["experiment"]["seed"],
    }
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "seeds.json", seeds)
    (run_dir / "environment.txt").write_text(
        json.dumps(environment, indent=2, default=str) + "\n"
    )
    (run_dir / "seeds.txt").write_text(json.dumps(seeds, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    command_argv = [sys.executable, *sys.argv] if argv is None else [
        sys.executable,
        "scripts/train_dinowm_baseline.py",
        *argv,
    ]
    args = parse_args(argv)
    config = load_config(args.config)

    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.output_dir is not None:
        config["output"]["output_dir"] = str(args.output_dir)

    seed_everything(int(config["experiment"]["seed"]))
    device = torch.device(args.device)

    # Build datasets
    train_dataset, val_dataset, split_info = build_datasets(config, dry_run=args.dry_run)
    print(f"Train: {len(train_dataset)} windows, Val: {len(val_dataset)} windows")

    # Build model
    model_cfg = config["model"]
    model = DINOwMTransformer(
        patch_dim=int(model_cfg["patch_dim"]),
        feature_dim=int(model_cfg["feature_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        future_horizon=int(model_cfg["future_horizon"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params ({trainable_params:,} trainable)")

    # Build dataloaders
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=patch_collate_fn,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=patch_collate_fn,
        num_workers=0,
    )

    # Build optimizer
    lr = float(config["training"]["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup scheduler
    warmup_steps = int(config["training"].get("warmup_steps", 100))
    total_epochs = int(config["training"]["epochs"])

    # Setup output dir
    run_dir = Path(config["output"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    write_reproducibility_files(run_dir, config, command_argv)
    write_json(run_dir / "split.json", split_info)

    # Write notes
    notes_path = run_dir / "notes.md"
    notes_path.write_text(
        "# DINOwM Transformer Baseline\n\n"
        "Standalone training on cached DINOv2 patch latents.\n"
        "Model: DINOwMTransformer with explicit future action conditioning.\n"
        "Primary loss: patch_cosine_error (1 - cosine similarity, averaged over patches).\n"
        "No action prediction head -- this is a pure world model.\n\n"
        "## Known limitations\n"
        "- No SNN adapter yet (ANN-only baseline)\n"
        "- No closed-loop evaluation\n"
        "- Future action sequences are inputs, not predicted\n"
    )

    # CSV metrics file
    csv_path = run_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDNAMES)
    csv_writer.writeheader()

    best_metric = float("inf")
    best_epoch = -1
    train_metrics_at_best: dict[str, Any] = {}
    final_train_metrics: dict[str, Any] = {}
    final_val_metrics: dict[str, Any] = {}
    lambda_pc = float(config["training"]["lambda_patch_cosine"])
    lambda_ac = float(config["training"]["lambda_action"])
    grad_clip = config["training"].get("grad_clip_norm")
    grad_clip_norm = float(grad_clip) if grad_clip is not None else None

    print(f"\nTraining for {total_epochs} epochs on {device}")
    print(f"  lambda_patch_cosine={lambda_pc}, lambda_action={lambda_ac}")
    print(f"  output_dir={run_dir}\n")

    for epoch in range(1, total_epochs + 1):
        # Linear warmup
        if epoch * len(train_loader) < warmup_steps:
            warmup_factor = max(1e-6, (epoch * len(train_loader)) / warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * warmup_factor
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        train_metrics = run_one_split(
            model, train_loader, device=device, optimizer=optimizer,
            lambda_patch_cosine=lambda_pc, lambda_action=lambda_ac,
            grad_clip_norm=grad_clip_norm, max_steps=args.max_steps,
        )
        val_metrics = run_one_split(
            model, val_loader, device=device, optimizer=None,
            lambda_patch_cosine=lambda_pc, lambda_action=lambda_ac,
            grad_clip_norm=None, max_steps=args.max_steps,
        )

        # Write CSV rows
        csv_writer.writerow(format_metric_row(epoch, "train", train_metrics))
        csv_writer.writerow(format_metric_row(epoch, "val", val_metrics))
        csv_file.flush()

        # Save best model
        val_metric = val_metrics["patch_cosine_error"]
        if val_metric < best_metric:
            best_metric = val_metric
            best_epoch = epoch
            train_metrics_at_best = dict(train_metrics)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_patch_cosine_error": val_metric,
                "config": config,
            }, run_dir / "best.pt")

        # Track final epoch metrics
        final_train_metrics = dict(train_metrics)
        final_val_metrics = dict(val_metrics)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"train_cos={train_metrics['patch_cosine_error']:.4f} | "
                f"val_cos={val_metrics['patch_cosine_error']:.4f} | "
                f"val_mse={val_metrics['patch_mse']:.6f} | "
                f"best={best_metric:.4f}@ep{best_epoch}"
            )

    # Save last checkpoint
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, run_dir / "last.pt")

    csv_file.close()

    # Write summary
    git_info = capture_git_info()

    # Compute dataset action statistics for downstream baselines (train split only)
    action_stats = None
    if not args.dry_run:
        try:
            all_actions = []
            for s in train_dataset:
                all_actions.append(s["future_actions"])
            all_actions = torch.stack(all_actions, dim=0)  # [N, H, A]
            action_stats = {
                "mean": all_actions.mean(dim=[0, 1]).tolist(),
                "std": all_actions.std(dim=[0, 1]).clamp(min=1e-6).tolist(),
                "source_split": "train",
                "n_samples": len(train_dataset),
                "note": "Statistics computed from train split future_actions only to avoid validation leakage",
            }
            # Write action_stats.json for planning baseline use
            write_json(run_dir / "action_stats.json", action_stats)
        except Exception:
            action_stats = None

    summary = {
        "best_epoch": best_epoch,
        "best_val_patch_cosine_error": best_metric,
        "total_epochs": total_epochs,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "num_tasks": split_info.get("num_tasks"),
        "num_episodes": split_info.get("num_episodes"),
        "patch_latent_shape": split_info.get("patch_latent_shape"),
        "action_shape": split_info.get("action_shape"),
        "context_len": int(config["data"]["context_len"]),
        "future_horizon": int(config["data"]["future_horizon"]),
        "dataset_cache_dir": config["data"]["cache_dir"],
        "config_path": str(args.config),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "seed": int(config["experiment"]["seed"]),
        "device": str(device),
        "train_patch_cosine_error_at_best_epoch": train_metrics_at_best.get("patch_cosine_error"),
        "train_val_gap_at_best_epoch": (
            train_metrics_at_best.get("patch_cosine_error", 0) - best_metric
            if train_metrics_at_best else None
        ),
        "final_train_patch_cosine_error": final_train_metrics.get("patch_cosine_error"),
        "final_val_patch_cosine_error": final_val_metrics.get("patch_cosine_error"),
        "action_stats": action_stats,
    }
    write_json(run_dir / "summary.json", summary)

    print(f"\nDone. Best val_patch_cosine_error={best_metric:.4f} at epoch {best_epoch}")
    print(f"Output: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
