#!/usr/bin/env python3
"""Preflight check for cached patch latent dataset.

Verifies cache existence, shapes, splits, and reports dataset stats before training.

Usage:
    python scripts/preflight_patch_latents.py \
        --config configs/reportable/dinowm_baseline_real.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.patch_latent_dataset import (  # noqa: E402
    PatchLatentTransitionDataset,
    load_patch_latent_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def capture_git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True
        ).strip()
        return {"commit": commit, "dirty": len(dirty) > 0, "dirty_files": dirty[:500]}
    except Exception:
        return {"commit": "unknown", "dirty": None, "dirty_files": ""}


def check_nan_inf(tensor_sample: Any, name: str) -> list[str]:
    import torch
    errors = []
    if isinstance(tensor_sample, torch.Tensor):
        if torch.isnan(tensor_sample).any():
            errors.append(f"{name}: contains NaN")
        if torch.isinf(tensor_sample).any():
            errors.append(f"{name}: contains Inf")
    return errors


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    git_info = capture_git_info()

    cache_dir = Path(config["data"]["cache_dir"])
    context_len = int(config["data"]["context_len"])
    future_horizon = int(config["data"]["future_horizon"])
    seed = int(config["experiment"]["seed"])

    report: dict[str, Any] = {
        "config_path": str(args.config),
        "cache_dir": str(cache_dir),
        "git": git_info,
        "seed": seed,
        "context_len": context_len,
        "future_horizon": future_horizon,
        "errors": [],
        "warnings": [],
    }

    # 1. Check cache exists
    if not cache_dir.exists():
        report["errors"].append(f"Cache directory not found: {cache_dir}")
        _write_report(args, report)
        return 1

    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        report["errors"].append(f"metadata.json not found in {cache_dir}")
        _write_report(args, report)
        return 1

    metadata = json.loads(metadata_path.read_text())
    report["cache_metadata"] = {
        "encoder_type": metadata.get("encoder_type"),
        "encoder_name": metadata.get("encoder_name"),
        "image_size": metadata.get("image_size"),
        "patch_size": metadata.get("patch_size"),
        "num_patches": metadata.get("num_patches"),
        "feature_dim": metadata.get("feature_dim"),
    }

    # 2. Load raw trajectories to count demos/tasks
    trajectories = load_patch_latent_cache(cache_dir)
    report["num_trajectories"] = len(trajectories)

    # Extract task names and episode IDs
    task_names: set[str] = set()
    episode_ids: list[str] = []
    for traj in trajectories:
        task_name = traj.task_name or "unknown"
        task_names.add(task_name)
        episode_ids.append(traj.trajectory_id)

    report["num_tasks"] = len(task_names)
    report["task_names"] = sorted(task_names)
    report["episode_ids"] = episode_ids

    # 3. Check shapes from first trajectory
    if trajectories:
        import torch
        first_traj = trajectories[0]
        patch_latents = torch.tensor(first_traj.patch_latents)
        actions = torch.tensor(first_traj.actions)
        report["patch_latent_shape"] = list(patch_latents.shape)
        report["action_shape"] = list(actions.shape)
        report["num_frames_first_demo"] = int(patch_latents.shape[0])

        # Check for NaN/Inf
        errors = check_nan_inf(patch_latents, "patch_latents")
        errors.extend(check_nan_inf(actions, "actions"))
        report["errors"].extend(errors)

        # Random sample check
        import random
        rng = random.Random(seed)
        idx = rng.randint(0, len(trajectories) - 1)
        sample_patch = torch.tensor(trajectories[idx].patch_latents)
        sample_actions = torch.tensor(trajectories[idx].actions)
        errors = check_nan_inf(sample_patch, f"random_sample[{idx}].patch_latents")
        errors.extend(check_nan_inf(sample_actions, f"random_sample[{idx}].actions"))
        report["errors"].extend(errors)

    # 4. Build full dataset and check splits
    full_dataset = PatchLatentTransitionDataset(
        cache_dir,
        context_len=context_len,
        future_horizon=future_horizon,
        split="train",
    )
    report["num_windows_total"] = len(full_dataset)

    # Split using same logic as trainer (90/10 by window index)
    train_ratio = float(config["data"].get("train_ratio", 0.9))
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)

    import torch as _torch
    rng = _torch.Generator().manual_seed(seed)
    indices = _torch.randperm(n_total, generator=rng).tolist()
    train_indices = set(indices[:n_train])
    val_indices = set(indices[n_train:])

    report["num_windows_train"] = len(train_indices)
    report["num_windows_val"] = len(val_indices)
    report["train_ratio"] = train_ratio

    # 5. Check split leakage: trajectory overlap
    train_episodes: set[str] = set()
    val_episodes: set[str] = set()
    train_tasks: dict[str, int] = {}
    val_tasks: dict[str, int] = {}

    for i in train_indices:
        traj_idx, t = full_dataset._index[i]
        traj = full_dataset.trajectories[traj_idx]
        ep_id = traj.trajectory_id
        task = traj.task_name or "unknown"
        train_episodes.add(ep_id)
        train_tasks[task] = train_tasks.get(task, 0) + 1

    for i in val_indices:
        traj_idx, t = full_dataset._index[i]
        traj = full_dataset.trajectories[traj_idx]
        ep_id = traj.trajectory_id
        task = traj.task_name or "unknown"
        val_episodes.add(ep_id)
        val_tasks[task] = val_tasks.get(task, 0) + 1

    episode_overlap = train_episodes & val_episodes
    report["split_leakage"] = {
        "train_episode_ids": sorted(train_episodes),
        "val_episode_ids": sorted(val_episodes),
        "episode_overlap_count": len(episode_overlap),
        "episode_overlap_ids": sorted(episode_overlap),
        "task_distribution_train": dict(sorted(train_tasks.items())),
        "task_distribution_val": dict(sorted(val_tasks.items())),
        "window_stride": 1,
        "note": "Split is by window index (random shuffle), not by trajectory. "
                "Windows from the same trajectory may appear in both train and val. "
                "This is standard for time-series forecasting but must be acknowledged.",
    }

    if len(episode_overlap) > 0:
        report["warnings"].append(
            f"{len(episode_overlap)} episodes have windows in both train and val. "
            "This is expected for index-based splits but means val metrics may be "
            "inflated by temporal proximity to training windows."
        )

    # 6. Write report
    _write_report(args, report)

    # Print summary
    print(f"Preflight: {len(trajectories)} trajectories, {len(task_names)} tasks")
    print(f"  Patch shape: {report.get('patch_latent_shape')}")
    print(f"  Action shape: {report.get('action_shape')}")
    print(f"  Windows: {n_total} total ({len(train_indices)} train, {len(val_indices)} val)")
    print(f"  Episode overlap: {len(episode_overlap)} episodes in both splits")
    if report["errors"]:
        print(f"  ERRORS: {len(report['errors'])}")
        for e in report["errors"]:
            print(f"    - {e}")
        return 1
    if report["warnings"]:
        print(f"  WARNINGS: {len(report['warnings'])}")
        for w in report["warnings"]:
            print(f"    - {w}")
    print("  Status: PASS")
    return 0


def _write_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    out_dir = args.output_dir or Path(report.get("cache_dir", ".")).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preflight_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
