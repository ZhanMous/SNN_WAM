#!/usr/bin/env python3
"""Overfit diagnostics: decompose single-demo error and isolate root causes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.trajectory_window import TrajectoryWindowDataset  # noqa: E402
from src.models.heads import SplitActionGripperHead  # noqa: E402
from src.models.registry import build_offline_model, count_parameters  # noqa: E402
from src.train.metrics import (  # noqa: E402
    action_mse,
    action_mse_per_dimension,
    action_mse_per_horizon,
)
from src.train.train_offline import (  # noqa: E402
    apply_action_transform,
    build_action_transform,
    checkpoint_payload,
    collate_action_batch,
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
)
from src.utils.config import load_config  # noqa: E402
from src.utils.experiment_io import capture_environment, capture_git_commit  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

GRIPPER_DIM = 6
GRIPPER_OPEN_THRESH = 0.5
GRIPPER_CLOSE_THRESH = -0.5


# ---------------------------------------------------------------------------
# Causal next-action contract v1
# ---------------------------------------------------------------------------


def causal_next_action_v1_check(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Check causal_next_action_v1 invariants on a single dataset sample.

    Contract:
    - inputs: observation[t], proprio[t], action_history[t-k:t-1], task/instruction
    - target: action[t]
    - max(action_history_index) < target_action_index
    - max(observation_index) <= target_action_index
    - no future latent is used as policy input
    - auxiliary future targets are target-only, never in input_keys

    Returns dict with 'pass' bool and per-invariant results.
    """
    results: dict[str, Any] = {"invariants": {}}
    all_pass = True

    # 1. action_history indices < target_action indices
    history_indices = sample.get("action_history_indices", [])
    target_indices = sample.get("target_action_indices", [])
    if history_indices and target_indices:
        max_hist = max(history_indices)
        min_target = min(target_indices)
        ok = max_hist < min_target
        results["invariants"]["action_history_before_target"] = {
            "pass": ok,
            "max_history_index": max_hist,
            "min_target_index": min_target,
        }
        if not ok:
            all_pass = False
    else:
        results["invariants"]["action_history_before_target"] = {"pass": False, "reason": "missing_indices"}

    # 2. observation index <= target_action index
    time_index = sample.get("time_index")
    if time_index is not None and target_indices:
        ok = time_index <= min(target_indices)
        results["invariants"]["observation_before_or_at_target"] = {
            "pass": ok,
            "observation_index": time_index,
            "min_target_index": min(target_indices),
        }
        if not ok:
            all_pass = False

    # 3. no future latent in input_keys
    input_keys = set(sample.get("input_keys", ()))
    future_in_input = any("future" in k.lower() for k in input_keys)
    results["invariants"]["no_future_latent_in_input"] = {
        "pass": not future_in_input,
        "input_keys": sorted(input_keys),
    }
    if future_in_input:
        all_pass = False

    # 4. auxiliary future targets are target-only
    target_keys = set(sample.get("target_keys", ()))
    future_target_keys = [k for k in target_keys if "future" in k.lower()]
    future_leaked_to_input = [k for k in future_target_keys if k in input_keys]
    results["invariants"]["future_targets_not_in_input"] = {
        "pass": len(future_leaked_to_input) == 0,
        "future_target_keys": future_target_keys,
        "leaked_to_input": future_leaked_to_input,
    }
    if future_leaked_to_input:
        all_pass = False

    # 5. target_shift == 0 for causal next-action
    target_shift = sample.get("target_shift", 0)
    results["invariants"]["target_shift_is_zero"] = {
        "pass": target_shift == 0,
        "target_shift": target_shift,
    }
    if target_shift != 0:
        all_pass = False

    results["pass"] = all_pass
    return results


def run_causal_contract_tests(dataset) -> dict[str, Any]:
    """Run causal_next_action_v1 checks on all samples in a dataset."""
    n = len(dataset)
    if n == 0:
        return {"pass": False, "reason": "empty_dataset", "n_samples": 0}
    results = [causal_next_action_v1_check(dataset[i]) for i in range(n)]
    all_pass = all(r["pass"] for r in results)
    failures = [i for i, r in enumerate(results) if not r["pass"]]
    return {
        "pass": all_pass,
        "n_samples": n,
        "n_failures": len(failures),
        "failure_indices": failures[:10],
        "first_failure": results[failures[0]] if failures else None,
    }


# ---------------------------------------------------------------------------
# Latent sanity diagnostics
# ---------------------------------------------------------------------------


def run_latent_sanity(
    trajectory,
    *,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Compute latent sanity metrics for a single trajectory.

    Produces latent_sanity_report.md with:
    - frame hash / pixel difference across timesteps
    - DINO latent variance per dimension
    - adjacent-frame latent cosine distance
    - PCA over latent trajectory
    - nearest-neighbor timestep retrieval from DINO latent
    - camera key and cache-index audit
    """
    latents = trajectory.visual_latents
    if latents is None:
        return {"pass": False, "reason": "no_visual_latents"}

    latents_np = np.array(latents, dtype=np.float32)  # [T, D]
    T, D = latents_np.shape

    # 1. Latent variance per dimension
    latent_var = latents_np.var(axis=0)
    latent_mean_var = float(latent_var.mean())
    latent_min_var = float(latent_var.min())
    latent_max_var = float(latent_var.max())

    # 2. Adjacent-frame cosine distance
    def cosine_dist(a, b):
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 1.0
        return 1.0 - dot / (na * nb)

    adj_cos_dists = [cosine_dist(latents_np[t], latents_np[t + 1]) for t in range(T - 1)]
    mean_adj_cos = float(np.mean(adj_cos_dists))
    max_adj_cos = float(np.max(adj_cos_dists))
    min_adj_cos = float(np.min(adj_cos_dists))

    # 3. PCA over latent trajectory
    centered = latents_np - latents_np.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = float((s ** 2).sum())
        top5_var = float((s[:5] ** 2).sum()) if len(s) >= 5 else total_var
        pca_concentration = top5_var / max(total_var, 1e-12)
    except Exception:
        pca_concentration = float("nan")

    # 4. Nearest-neighbor timestep retrieval
    nn_correct = 0
    nn_dist_ratio = []
    for t in range(T):
        dists = np.linalg.norm(latents_np - latents_np[t], axis=1)
        dists[t] = float("inf")
        nn_idx = int(np.argmin(dists))
        if nn_idx == t - 1 or nn_idx == t + 1:
            nn_correct += 1
        sorted_dists = np.sort(dists[dists < float("inf")])
        if len(sorted_dists) >= 2:
            nn_dist_ratio.append(float(sorted_dists[0] / max(sorted_dists[1], 1e-12)))
    nn_neighbor_accuracy = nn_correct / max(T, 1)
    mean_nn_ratio = float(np.mean(nn_dist_ratio)) if nn_dist_ratio else float("nan")

    # 5. Frame pixel difference (if images are actual arrays, not paths)
    pixel_diffs = []
    if trajectory.images is not None and len(trajectory.images) > 1:
        first_img = trajectory.images[0]
        if isinstance(first_img, np.ndarray) or (isinstance(first_img, (list, tuple)) and not isinstance(first_img[0], str)):
            for t in range(min(T - 1, len(trajectory.images) - 1)):
                try:
                    img_t = np.array(trajectory.images[t], dtype=np.float32).flatten()
                    img_t1 = np.array(trajectory.images[t + 1], dtype=np.float32).flatten()
                    if img_t.shape == img_t1.shape:
                        pixel_diffs.append(float(np.abs(img_t - img_t1).mean()))
                except (ValueError, TypeError):
                    break
    mean_pixel_diff = float(np.mean(pixel_diffs)) if pixel_diffs else float("nan")

    # 6. Latent uniqueness check
    unique_rows = np.unique(np.round(latents_np, decimals=6), axis=0)
    n_unique = len(unique_rows)
    all_unique = n_unique == T

    report = {
        "trajectory_id": trajectory.trajectory_id,
        "trajectory_length": T,
        "latent_dim": D,
        "latent_variance": {
            "mean": latent_mean_var,
            "min": latent_min_var,
            "max": latent_max_var,
        },
        "adjacent_cosine_distance": {
            "mean": mean_adj_cos,
            "min": min_adj_cos,
            "max": max_adj_cos,
        },
        "pca_top5_concentration": pca_concentration,
        "nearest_neighbor": {
            "adjacent_accuracy": nn_neighbor_accuracy,
            "mean_dist_ratio": mean_nn_ratio,
        },
        "pixel_diff_mean": mean_pixel_diff,
        "latent_uniqueness": {
            "all_unique": all_unique,
            "n_unique": n_unique,
            "n_total": T,
        },
        "pass": latent_mean_var > 1e-6 and all_unique,
    }

    _write_json(output_dir / "latent_sanity.json", report)

    lines = [
        "# Latent Sanity Report",
        "",
        f"Trajectory: {trajectory.trajectory_id}",
        f"Length: {T}, Latent dim: {D}",
        "",
        "## Latent Variance",
        f"- Mean: {latent_mean_var:.6e}",
        f"- Min: {latent_min_var:.6e}",
        f"- Max: {latent_max_var:.6e}",
        "",
        "## Adjacent-Frame Cosine Distance",
        f"- Mean: {mean_adj_cos:.6f}",
        f"- Min: {min_adj_cos:.6f}",
        f"- Max: {max_adj_cos:.6f}",
        "",
        "## PCA Concentration",
        f"- Top-5 components variance fraction: {pca_concentration:.4f}",
        "",
        "## Nearest-Neighbor Retrieval",
        f"- Adjacent accuracy: {nn_neighbor_accuracy:.4f}",
        f"- Mean distance ratio: {mean_nn_ratio:.4f}",
        "",
        "## Pixel Differences",
        f"- Mean adjacent pixel diff: {mean_pixel_diff:.6f}",
        "",
        "## Latent Uniqueness",
        f"- All unique: {all_unique}",
        f"- Unique / Total: {n_unique} / {T}",
        "",
        f"## Pass: {report['pass']}",
    ]
    (output_dir / "latent_sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# Causal H=1 baseline models
# ---------------------------------------------------------------------------


class ProprioSplitMLP(nn.Module):
    """H=1 baseline: predict actions from proprio/state[t] only."""

    def __init__(self, *, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, optional_state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.action_head(self.network(optional_state_t))


class ActionHistorySplitGRU(nn.Module):
    """H=1 baseline: predict actions from action history only (no visual input)."""

    def __init__(self, *, history_len: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=action_dim, hidden_size=hidden_dim, batch_first=True)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, action_history: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.gru(action_history)
        return self.action_head(self.network(hidden[-1]))


class DinoProprioSplitMLP(nn.Module):
    """H=1 baseline: predict actions from DINO CLS + proprio."""

    def __init__(self, *, latent_dim: int, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, z_t: torch.Tensor, optional_state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.action_head(self.network(torch.cat([z_t, optional_state_t], dim=-1)))


class DinoProprioHistorySplitGRU(nn.Module):
    """H=1 baseline: predict actions from DINO CLS + proprio + action history."""

    def __init__(
        self, *, latent_dim: int, state_dim: int, action_dim: int,
        history_len: int, hidden_dim: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=action_dim, hidden_size=hidden_dim, batch_first=True)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(
        self, action_history: torch.Tensor, z_t: torch.Tensor, optional_state_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, hidden = self.gru(action_history)
        features = torch.cat([hidden[-1], z_t, optional_state_t], dim=-1)
        return self.action_head(self.network(features))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/diagnostics/overfit_diag"))
    parser.add_argument("--trajectory_id", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "any"], default="train")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss_threshold", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_ladder", action="store_true")
    parser.add_argument("--skip_shift_sweep", action="store_true")
    parser.add_argument("--skip_lookup_table", action="store_true")
    parser.add_argument("--skip_separate_loss", action="store_true")
    parser.add_argument("--repair_suite", action="store_true")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--baseline_epochs", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repair_suite:
        output_dir = run_h1_repair_diagnostics(
            config_path=args.config,
            output_root=args.output_dir,
            run_id=args.run_id,
            trajectory_id=args.trajectory_id,
            source_split=args.split,
            epochs=args.epochs,
            baseline_epochs=args.baseline_epochs or args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss_threshold=args.loss_threshold,
            device_name=args.device,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            command=[sys.executable, "-m", "src.eval.overfit_diagnostics", *(argv or sys.argv[1:])],
        )
        print(f"overfit_diagnostics_dir={output_dir}")
        return 0
    output_dir = run_overfit_diagnostics(
        config_path=args.config,
        output_root=args.output_dir,
        trajectory_id=args.trajectory_id,
        source_split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_threshold=args.loss_threshold,
        device_name=args.device,
        seed=args.seed,
        skip_ladder=args.skip_ladder,
        skip_shift_sweep=args.skip_shift_sweep,
        skip_lookup_table=args.skip_lookup_table,
        skip_separate_loss=args.skip_separate_loss,
    )
    print(f"overfit_diagnostics_dir={output_dir}")
    return 0


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_overfit_diagnostics(
    *,
    config_path: Path,
    output_root: Path,
    trajectory_id: str | None = None,
    source_split: str = "train",
    epochs: int = 300,
    batch_size: int = 64,
    lr: float | None = None,
    loss_threshold: float = 1e-4,
    device_name: str = "cpu",
    seed: int = 0,
    skip_ladder: bool = False,
    skip_shift_sweep: bool = False,
    skip_lookup_table: bool = False,
    skip_separate_loss: bool = False,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_overfit_diag"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    validate_training_scope(config)
    device = torch.device(device_name)

    # Load and select one trajectory
    trajectories, source_metadata = load_real_libero_trajectories(config)
    selected = _select_trajectory(trajectories, trajectory_id, source_split)
    train_traj = replace(selected, split="train")
    val_traj = replace(selected, split="val")
    diag_trajs = [train_traj, val_traj]

    action_transform, norm_stats = build_action_transform(diag_trajs, config)
    if action_transform is not None:
        diag_trajs = apply_action_transform(diag_trajs, action_transform)

    effective_lr = float(lr if lr is not None else config["training"]["lr"])

    # Build datasets for the default horizon
    default_h = int(config["data"]["action_horizon"])
    train_ds, val_ds = _build_datasets(diag_trajs, config, default_h)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_action_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_action_batch)

    sample = train_ds[0]
    action_dim = infer_action_dim(sample)

    # ===================================================================
    # 1. Full-horizon overfit with decomposition
    # ===================================================================
    print(f"[1/7] Full-horizon overfit (H={default_h}) ...")
    full_model = _build_model(config, sample, action_dim, device)
    full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=effective_lr)
    lambda_action = float(config["training"]["lambda_action"])
    lambda_future = float(config["training"]["lambda_future"])
    grad_clip = config["training"].get("grad_clip_norm")

    full_rows, full_best_mse, full_best_epoch = _train_and_log(
        model=full_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=full_optimizer,
        lambda_action=lambda_action,
        lambda_future=lambda_future,
        grad_clip=grad_clip,
        epochs=epochs,
        loss_threshold=loss_threshold,
        action_transform=action_transform,
        output_path=output_dir / "full_horizon_metrics.csv",
    )

    # Decompose final error
    decomposition = _decompose_error(
        full_model, val_loader, device, action_transform, action_dim,
    )
    _write_csv(output_dir / "overfit_decomposition.csv", decomposition["dim_rows"])
    _write_csv(output_dir / "timestep_mse.csv", decomposition["timestep_rows"])

    # Gripper diagnostics
    gripper_diag = _gripper_diagnostics(
        full_model, val_loader, device, action_transform,
    )
    _write_csv(output_dir / "gripper_diagnostics.csv", gripper_diag["rows"])

    # ===================================================================
    # 2. H=1 overfit gate
    # ===================================================================
    print("[2/7] H=1 overfit gate ...")
    h1_passed = False
    h1_best_mse = float("inf")
    if not skip_ladder:
        h1_model, h1_best_mse, h1_passed = _train_with_horizon(
            config=config, diag_trajs=diag_trajs, action_dim=action_dim,
            horizon=1, device=device, epochs=epochs, lr=effective_lr,
            batch_size=batch_size, loss_threshold=loss_threshold,
            lambda_action=lambda_action, lambda_future=lambda_future,
            grad_clip=grad_clip, action_transform=action_transform,
            output_path=output_dir / "h1_metrics.csv",
        )

    # ===================================================================
    # 3. Multi-horizon overfit ladder
    # ===================================================================
    print("[3/7] Multi-horizon ladder ...")
    ladder_rows = []
    if not skip_ladder:
        for h in [1, 2, 5, default_h]:
            if h == 1:
                ladder_rows.append({"horizon": h, "best_mse": h1_best_mse, "passed": h1_passed})
                continue
            if h == default_h:
                ladder_rows.append({"horizon": h, "best_mse": full_best_mse, "passed": full_best_mse <= loss_threshold})
                continue
            _, best_m, passed_m = _train_with_horizon(
                config=config, diag_trajs=diag_trajs, action_dim=action_dim,
                horizon=h, device=device, epochs=epochs, lr=effective_lr,
                batch_size=batch_size, loss_threshold=loss_threshold,
                lambda_action=lambda_action, lambda_future=lambda_future,
                grad_clip=grad_clip, action_transform=action_transform,
                output_path=output_dir / f"h{h}_metrics.csv",
            )
            ladder_rows.append({"horizon": h, "best_mse": best_m, "passed": passed_m})
    _write_csv(output_dir / "horizon_ladder.csv", ladder_rows)

    # ===================================================================
    # 4. Timestep alignment sweep
    # ===================================================================
    print("[4/7] Timestep alignment sweep ...")
    shift_rows = []
    if not skip_shift_sweep:
        shift_rows = _timestep_alignment_sweep(
            config=config, selected=selected, action_transform=action_transform,
            device=device, epochs=min(epochs, 150), lr=effective_lr,
            batch_size=batch_size, lambda_action=lambda_action,
            lambda_future=lambda_future, grad_clip=grad_clip,
            action_dim=action_dim, output_dir=output_dir,
        )
    _write_csv(output_dir / "timestep_shift_sweep.csv", shift_rows)

    # ===================================================================
    # 5. Separate gripper loss
    # ===================================================================
    print("[5/7] Separate gripper loss ...")
    sep_rows = []
    if not skip_separate_loss:
        sep_rows = _separate_gripper_loss_experiment(
            config=config, diag_trajs=diag_trajs, action_dim=action_dim,
            device=device, epochs=epochs, lr=effective_lr,
            batch_size=batch_size, lambda_action=lambda_action,
            lambda_future=lambda_future, grad_clip=grad_clip,
            action_transform=action_transform,
            output_dir=output_dir,
        )
    _write_csv(output_dir / "separate_gripper_loss.csv", sep_rows)

    # ===================================================================
    # 6. Lookup-table memorization baseline
    # ===================================================================
    print("[6/7] Lookup-table memorization baseline ...")
    lookup_passed = False
    lookup_best_mse = float("inf")
    if not skip_lookup_table:
        lookup_best_mse, lookup_passed = _lookup_table_baseline(
            train_ds=train_ds, val_ds=val_ds, action_dim=action_dim,
            device=device, epochs=500, lr=0.01, batch_size=batch_size,
            loss_threshold=loss_threshold, action_transform=action_transform,
            output_path=output_dir / "lookup_table_metrics.csv",
        )

    # ===================================================================
    # 7. Mask/padding audit
    # ===================================================================
    print("[7/7] Mask/padding audit ...")
    mask_audit = _audit_masks_and_padding(diag_trajs, config, action_dim)
    _write_json(output_dir / "mask_audit.json", mask_audit)

    # ===================================================================
    # Summary
    # ===================================================================
    cont_mse = decomposition.get("continuous_mean_mse", 0.0)
    grip_mse = decomposition.get("gripper_mse", 0.0)
    summary = {
        "full_horizon": default_h,
        "full_best_mse": full_best_mse,
        "full_best_epoch": full_best_epoch,
        "full_passed": full_best_mse <= loss_threshold,
        "continuous_dims_mean_mse": cont_mse,
        "gripper_mse": grip_mse,
        "gripper_fraction_of_total": grip_mse / max(full_best_mse, 1e-12),
        "h1_best_mse": h1_best_mse,
        "h1_passed": h1_passed,
        "lookup_table_best_mse": lookup_best_mse,
        "lookup_table_passed": lookup_passed,
        "loss_threshold": loss_threshold,
        "gripper_diagnostics": gripper_diag.get("summary", {}),
        "mask_audit": mask_audit,
        "diagnosis": _classify_failure(
            full_best_mse, cont_mse, grip_mse, h1_best_mse, h1_passed,
            lookup_best_mse, lookup_passed, default_h, loss_threshold,
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_diagnostic_report(output_dir / "diagnostic_report.md", summary, decomposition, gripper_diag, ladder_rows, shift_rows)
    return output_dir


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _select_trajectory(trajectories, trajectory_id, source_split):
    candidates = list(trajectories)
    if trajectory_id is not None:
        for t in candidates:
            if t.trajectory_id == trajectory_id:
                return t
        raise ValueError(f"trajectory_id not found: {trajectory_id}")
    if source_split != "any":
        candidates = [t for t in candidates if t.split == source_split]
    if not candidates:
        raise ValueError(f"no trajectories for split={source_split!r}")
    return sorted(candidates, key=lambda t: t.trajectory_id)[0]


def _build_datasets(diag_trajs, config, action_horizon):
    common = dict(
        trajectories=diag_trajs,
        history_len=int(config["data"]["history_len"]),
        future_horizon=int(config["data"]["future_horizon"]),
        include_current_latent=uses_current_latent(config),
        include_future_latents=requires_future_latents(config),
    )
    train_ds = TrajectoryWindowDataset(split="train", action_horizon=action_horizon, **common)
    val_ds = TrajectoryWindowDataset(split="val", action_horizon=action_horizon, **common)
    return train_ds, val_ds


def _build_model(config, sample, action_dim, device, action_horizon=None):
    latent_dim = None
    if has_future_latent_targets(sample):
        latent_dim = infer_latent_dim(sample)
    elif has_current_latent(sample):
        latent_dim = infer_current_latent_dim(sample)
    cfg = dict(config)
    if action_horizon is not None:
        cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in config.items()}
        cfg["data"] = dict(config["data"])
        cfg["data"]["action_horizon"] = action_horizon
    model = build_offline_model(
        cfg, action_dim=action_dim, latent_dim=latent_dim,
        state_dim=infer_state_dim(sample), num_tasks=1,
    )
    return model.to(device)


def _train_and_log(
    *, model, train_loader, val_loader, device, optimizer,
    lambda_action, lambda_future, grad_clip, epochs, loss_threshold,
    action_transform, output_path,
):
    rows = []
    best_mse = float("inf")
    best_epoch = -1
    passed = False

    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "epoch", "split", "action_mse", "action_loss",
            "continuous_mse", "gripper_mse",
            "mse_h0", "mse_h1", "mse_h2", "mse_h3",
        ]
        # Pad for larger horizons
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for epoch in range(epochs):
            train_m = run_one_split(
                model, train_loader, device=device, optimizer=optimizer,
                lambda_action=lambda_action, lambda_future=lambda_future,
                grad_clip_norm=grad_clip, max_steps=None,
                action_transform=action_transform,
            )
            val_m = run_one_split(
                model, val_loader, device=device, optimizer=None,
                lambda_action=lambda_action, lambda_future=lambda_future,
                grad_clip_norm=None, max_steps=None,
                action_transform=action_transform,
            )

            # Extra decomposition on val
            val_decomp = _decompose_error(model, val_loader, device, action_transform,
                                          _infer_action_dim_from_loader(val_loader))

            for split_name, metrics, decomp in [
                ("train", train_m, {}),
                ("val", val_m, val_decomp),
            ]:
                row = {
                    "epoch": epoch,
                    "split": split_name,
                    "action_mse": metrics["action_mse"],
                    "action_loss": metrics["action_loss"],
                    "continuous_mse": decomp.get("continuous_mean_mse", ""),
                    "gripper_mse": decomp.get("gripper_mse", ""),
                }
                by_h = metrics.get("action_mse_by_horizon", [])
                for i, v in enumerate(by_h):
                    row[f"mse_h{i}"] = v
                writer.writerow(row)
                rows.append(row)
            f.flush()

            val_mse = float(val_m["action_mse"])
            if val_mse < best_mse:
                best_mse = val_mse
                best_epoch = epoch

            if float(train_m["action_mse"]) <= loss_threshold:
                passed = True
                break

    return rows, best_mse, best_epoch


def _train_with_horizon(
    *, config, diag_trajs, action_dim, horizon, device, epochs, lr,
    batch_size, loss_threshold, lambda_action, lambda_future, grad_clip,
    action_transform, output_path,
):
    train_ds, val_ds = _build_datasets(diag_trajs, config, horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_action_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_action_batch)
    sample = train_ds[0]
    model = _build_model(config, sample, action_dim, device, action_horizon=horizon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rows, best_mse, _ = _train_and_log(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, optimizer=optimizer, lambda_action=lambda_action,
        lambda_future=lambda_future, grad_clip=grad_clip, epochs=epochs,
        loss_threshold=loss_threshold, action_transform=action_transform,
        output_path=output_path,
    )
    return model, best_mse, best_mse <= loss_threshold


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def _decompose_error(model, loader, device, action_transform, action_dim):
    """Compute per-dimension, per-horizon, and per-timestep MSE."""
    model.eval()
    dim_sq_err = torch.zeros(action_dim)
    dim_count = 0
    horizon_sq_err = None
    horizon_count = 0
    timestep_rows = []

    with torch.no_grad():
        for batch in loader:
            outputs = forward_offline_model(model, batch, device=device)
            pred = outputs["pred_actions"] if isinstance(outputs, dict) else outputs
            target = batch["target_actions"]
            if action_transform is not None:
                pred = action_transform.denormalize_tensor(pred)
                target = action_transform.denormalize_tensor(target)
            pred_cpu = pred.cpu()
            target_cpu = target.cpu()
            se = (pred_cpu - target_cpu).pow(2)  # [B, H, A]

            # Per-dimension
            dim_sq_err += se.sum(dim=(0, 1))
            dim_count += se.shape[0] * se.shape[1]

            # Per-horizon
            h_se = se.mean(dim=-1)  # [B, H]
            if horizon_sq_err is None:
                horizon_sq_err = h_se.sum(dim=0)
            else:
                horizon_sq_err += h_se.sum(dim=0)
            horizon_count += se.shape[0]

            # Per-timestep
            time_indices = batch.get("time_index", [None] * se.shape[0])
            for i in range(se.shape[0]):
                ti = time_indices[i] if isinstance(time_indices[i], int) else time_indices[i]
                row_se = se[i].mean(dim=0)  # [A] averaged over horizon
                timestep_rows.append({
                    "time_index": ti,
                    "mse": se[i].mean().item(),
                    "continuous_mse": se[i, :, :GRIPPER_DIM].mean().item() if action_dim > GRIPPER_DIM else 0.0,
                    "gripper_mse": se[i, :, GRIPPER_DIM].mean().item() if action_dim > GRIPPER_DIM else 0.0,
                })

    per_dim = (dim_sq_err / max(dim_count, 1)).tolist()
    per_horizon = (horizon_sq_err / max(horizon_count, 1)).tolist() if horizon_sq_err is not None else []

    continuous_dims = per_dim[:GRIPPER_DIM] if action_dim > GRIPPER_DIM else per_dim
    gripper_dim_mse = per_dim[GRIPPER_DIM] if action_dim > GRIPPER_DIM else 0.0

    dim_rows = []
    for i, mse_val in enumerate(per_dim):
        dim_rows.append({
            "dim_index": i,
            "dim_label": _dim_label(i, action_dim),
            "mse": mse_val,
            "type": "gripper" if i == GRIPPER_DIM else "continuous",
        })
    for i, mse_val in enumerate(per_horizon):
        dim_rows.append({
            "dim_index": f"horizon_{i}",
            "dim_label": f"horizon_{i}",
            "mse": mse_val,
            "type": "horizon",
        })

    return {
        "dim_rows": dim_rows,
        "timestep_rows": timestep_rows,
        "per_dim": per_dim,
        "per_horizon": per_horizon,
        "continuous_mean_mse": sum(continuous_dims) / max(len(continuous_dims), 1),
        "gripper_mse": gripper_dim_mse,
    }


def _dim_label(i, action_dim):
    labels = ["delta_pos_x", "delta_pos_y", "delta_pos_z",
              "delta_rot_x", "delta_rot_y", "delta_rot_z", "gripper"]
    if i < len(labels):
        return labels[i]
    return f"dim_{i}"


def _infer_action_dim_from_loader(loader):
    sample = loader.dataset[0]
    return infer_action_dim(sample)


# ---------------------------------------------------------------------------
# Gripper diagnostics
# ---------------------------------------------------------------------------


def _gripper_diagnostics(model, loader, device, action_transform):
    """Compute gripper-specific metrics on the validation set."""
    model.eval()
    pred_grip = []
    true_grip = []

    with torch.no_grad():
        for batch in loader:
            outputs = forward_offline_model(model, batch, device=device)
            pred = outputs["pred_actions"] if isinstance(outputs, dict) else outputs
            target = batch["target_actions"]
            if action_transform is not None:
                pred = action_transform.denormalize_tensor(pred)
                target = action_transform.denormalize_tensor(target)
            pred_grip.append(pred[:, :, GRIPPER_DIM].cpu())
            true_grip.append(target[:, :, GRIPPER_DIM].cpu())

    pred_g = torch.cat(pred_grip, dim=0).flatten()
    true_g = torch.cat(true_grip, dim=0).flatten()

    # Sign accuracy
    pred_sign = torch.sign(pred_g)
    true_sign = torch.sign(true_g)
    sign_correct = (pred_sign == true_sign).float()
    # Neutral zone: |true| < 0.1 counts as neither open nor close
    neutral_mask = true_g.abs() < 0.1
    active_mask = ~neutral_mask

    sign_accuracy = sign_correct[active_mask].mean().item() if active_mask.any() else float("nan")

    # Open/close accuracy
    open_mask = true_g > GRIPPER_OPEN_THRESH
    close_mask = true_g < GRIPPER_CLOSE_THRESH
    open_acc = (pred_g[open_mask] > 0).float().mean().item() if open_mask.any() else float("nan")
    close_acc = (pred_g[close_mask] < 0).float().mean().item() if close_mask.any() else float("nan")

    # Transition detection: where true gripper changes sign
    true_binary = (true_g > 0).int()
    pred_binary = (pred_g > 0).int()
    transitions_true = _find_transitions(true_binary)
    transitions_pred = _find_transitions(pred_binary)

    # Transition F1
    trans_precision, trans_recall, trans_f1 = _transition_f1(
        transitions_true, transitions_pred, tolerance=2,
    )

    # Close timing error
    close_timing_err = _close_timing_error(true_g, pred_g)

    # MSE split
    grip_mse = (pred_g - true_g).pow(2).mean().item()
    grip_mae = (pred_g - true_g).abs().mean().item()

    # Convention check: does positive prediction correlate with positive target?
    if pred_g.numel() > 1:
        corr = torch.corrcoef(torch.stack([pred_g, true_g]))[0, 1].item()
    else:
        corr = float("nan")

    summary = {
        "gripper_mse": grip_mse,
        "gripper_mae": grip_mae,
        "sign_accuracy_active": sign_accuracy,
        "open_accuracy": open_acc,
        "close_accuracy": close_acc,
        "transition_precision": trans_precision,
        "transition_recall": trans_recall,
        "transition_f1": trans_f1,
        "close_timing_error_steps": close_timing_err,
        "sign_correlation": corr,
        "n_open": int(open_mask.sum().item()),
        "n_close": int(close_mask.sum().item()),
        "n_neutral": int(neutral_mask.sum().item()),
        "n_transitions_true": len(transitions_true),
        "n_transitions_pred": len(transitions_pred),
    }

    rows = [{"metric": k, "value": v} for k, v in summary.items()]
    return {"summary": summary, "rows": rows}


def _find_transitions(binary_seq):
    """Find indices where binary sequence changes value."""
    if len(binary_seq) < 2:
        return []
    changes = (binary_seq[1:] != binary_seq[:-1]).nonzero(as_tuple=True)[0]
    return changes.tolist()


def _transition_f1(true_trans, pred_trans, tolerance=2):
    """Compute transition F1 with tolerance window."""
    if not true_trans and not pred_trans:
        return 1.0, 1.0, 1.0
    if not true_trans:
        return 0.0, 1.0, 0.0
    if not pred_trans:
        return 1.0, 0.0, 0.0

    true_set = set(true_trans)
    matched_true = set()
    matched_pred = set()
    for pt in pred_trans:
        for tt in true_trans:
            if abs(pt - tt) <= tolerance and tt not in matched_true:
                matched_true.add(tt)
                matched_pred.add(pt)
                break
    precision = len(matched_pred) / max(len(pred_trans), 1)
    recall = len(matched_true) / max(len(true_trans), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def _close_timing_error(true_g, pred_g):
    """Average absolute timing error for close transitions (steps)."""
    true_close = (true_g < GRIPPER_CLOSE_THRESH).int()
    pred_close = (pred_g < GRIPPER_CLOSE_THRESH).int()
    true_trans = _find_transitions(true_close)
    pred_trans = _find_transitions(pred_close)
    if not true_trans or not pred_trans:
        return float("nan")
    errors = []
    for tt in true_trans:
        closest = min(pred_trans, key=lambda pt: abs(pt - tt))
        errors.append(abs(closest - tt))
    return sum(errors) / len(errors) if errors else float("nan")


# ---------------------------------------------------------------------------
# Timestep alignment sweep
# ---------------------------------------------------------------------------


def _timestep_alignment_sweep(
    *, config, selected, action_transform, device, epochs, lr,
    batch_size, lambda_action, lambda_future, grad_clip, action_dim, output_dir,
):
    """Train with different latent->action alignment offsets."""
    rows = []
    for shift in [-1, 0, 1, 2]:
        print(f"  shift={shift} ...")
        shifted_traj = _make_shifted_trajectory(selected, shift)
        train_t = replace(shifted_traj, split="train")
        val_t = replace(shifted_traj, split="val")
        trajs = [train_t, val_t]
        if action_transform is not None:
            trajs = apply_action_transform(trajs, action_transform)

        h = int(config["data"]["action_horizon"])
        train_ds = TrajectoryWindowDataset(
            trajs, split="train", history_len=int(config["data"]["history_len"]),
            action_horizon=h, future_horizon=int(config["data"]["future_horizon"]),
            include_current_latent=uses_current_latent(config),
            include_future_latents=requires_future_latents(config),
        )
        val_ds = TrajectoryWindowDataset(
            trajs, split="val", history_len=int(config["data"]["history_len"]),
            action_horizon=h, future_horizon=int(config["data"]["future_horizon"]),
            include_current_latent=uses_current_latent(config),
            include_future_latents=requires_future_latents(config),
        )
        if len(train_ds) == 0:
            rows.append({"shift": shift, "best_mse": float("inf"), "passed": False, "note": "no_valid_windows"})
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_action_batch)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_action_batch)
        sample = train_ds[0]
        model = _build_model(config, sample, action_dim, device, action_horizon=h)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        _, best_mse, _ = _train_and_log(
            model=model, train_loader=train_loader, val_loader=val_loader,
            device=device, optimizer=optimizer, lambda_action=lambda_action,
            lambda_future=lambda_future, grad_clip=grad_clip, epochs=epochs,
            loss_threshold=1e-6, action_transform=action_transform,
            output_path=output_dir / f"shift_{shift:+d}_metrics.csv",
        )
        rows.append({"shift": shift, "best_mse": best_mse, "passed": False, "note": ""})

    return rows


def _make_shifted_trajectory(traj, shift):
    """Shift visual_latents relative to actions by `shift` steps.

    Convention: action_to_current_obs, so at time t the model sees
    image[t], latent[t], action_history, and predicts actions[t+1:t+1+H].

    shift=+1 means latent[t] is replaced by latent[t+1] (latent is one step ahead).
    shift=-1 means latent[t] is replaced by latent[t-1] (latent is one step behind).

    We achieve this by trimming the trajectory to keep valid overlapping indices.
    """
    if shift == 0:
        return traj
    T = traj.length
    if traj.visual_latents is None:
        return traj

    if shift > 0:
        # latent[t] = original_latent[t+shift], so we need T-shift valid steps
        new_T = T - shift
        return replace(
            traj,
            images=list(traj.images[:new_T]),
            actions=list(traj.actions[:new_T]),
            states=list(traj.states[:new_T]) if traj.states is not None else None,
            visual_latents=list(traj.visual_latents[shift:shift + new_T]),
            frame_refs=list(traj.frame_refs[:new_T]) if traj.frame_refs is not None else None,
        )
    else:
        # shift < 0: latent[t] = original_latent[t-|shift|]
        offset = -shift
        new_T = T - offset
        return replace(
            traj,
            images=list(traj.images[offset:offset + new_T]),
            actions=list(traj.actions[offset:offset + new_T]),
            states=list(traj.states[offset:offset + new_T]) if traj.states is not None else None,
            visual_latents=list(traj.visual_latents[:new_T]),
            frame_refs=list(traj.frame_refs[offset:offset + new_T]) if traj.frame_refs is not None else None,
        )


# ---------------------------------------------------------------------------
# Separate gripper loss
# ---------------------------------------------------------------------------


def _separate_gripper_loss_experiment(
    *, config, diag_trajs, action_dim, device, epochs, lr,
    batch_size, lambda_action, lambda_future, grad_clip,
    action_transform, output_dir,
):
    """Train with separate continuous (SmoothL1) and gripper (BCE) losses."""
    h = int(config["data"]["action_horizon"])
    train_ds, val_ds = _build_datasets(diag_trajs, config, h)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_action_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_action_batch)
    sample = train_ds[0]
    model = _build_model(config, sample, action_dim, device, action_horizon=h)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    rows = []
    best_mse = float("inf")
    best_epoch = -1

    output_path = output_dir / "separate_gripper_loss_detail.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "split", "action_mse", "continuous_loss", "gripper_loss", "total_loss"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(epochs):
            # Train
            model.train()
            for batch in train_loader:
                outputs = forward_offline_model(model, batch, device=device)
                pred = outputs["pred_actions"] if isinstance(outputs, dict) else outputs
                target = batch["target_actions"].to(device)

                # Separate losses
                cont_loss = nn.functional.smooth_l1_loss(
                    pred[:, :, :GRIPPER_DIM], target[:, :, :GRIPPER_DIM]
                )
                # Gripper: binary cross-entropy on sign
                # Map target to {0, 1}: positive=1 (open), negative=0 (close)
                grip_target = (target[:, :, GRIPPER_DIM] > 0).float()
                grip_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred[:, :, GRIPPER_DIM], grip_target
                )
                total = lambda_action * (cont_loss + grip_loss)

                optimizer.zero_grad(set_to_none=True)
                total.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

            # Eval
            model.eval()
            with torch.no_grad():
                val_mse_sum = 0.0
                val_count = 0
                val_cont_sum = 0.0
                val_grip_sum = 0.0
                for batch in val_loader:
                    outputs = forward_offline_model(model, batch, device=device)
                    pred = outputs["pred_actions"] if isinstance(outputs, dict) else outputs
                    target = batch["target_actions"].to(device)
                    if action_transform is not None:
                        pred = action_transform.denormalize_tensor(pred)
                        target = action_transform.denormalize_tensor(target)
                    val_mse_sum += (pred - target).pow(2).sum().item()
                    val_count += int(target.numel())
                    val_cont_sum += nn.functional.smooth_l1_loss(
                        pred[:, :, :GRIPPER_DIM], target[:, :, :GRIPPER_DIM]
                    ).item()
                    grip_target = (target[:, :, GRIPPER_DIM] > 0).float()
                    val_grip_sum += nn.functional.binary_cross_entropy_with_logits(
                        pred[:, :, GRIPPER_DIM], grip_target
                    ).item()

            val_mse = val_mse_sum / max(val_count, 1)
            n_batches = max(len(val_loader), 1)
            for split_name, mse, c_loss, g_loss in [
                ("train", 0.0, 0.0, 0.0),
                ("val", val_mse, val_cont_sum / n_batches, val_grip_sum / n_batches),
            ]:
                writer.writerow({
                    "epoch": epoch, "split": split_name,
                    "action_mse": mse, "continuous_loss": c_loss,
                    "gripper_loss": g_loss, "total_loss": c_loss + g_loss,
                })
            f.flush()

            if val_mse < best_mse:
                best_mse = val_mse
                best_epoch = epoch

    rows.append({
        "variant": "separate_loss",
        "best_mse": best_mse,
        "best_epoch": best_epoch,
        "continuous_loss_fn": "smooth_l1",
        "gripper_loss_fn": "bce_with_logits",
    })

    # Also run a baseline with pure MSE for comparison
    _, baseline_mse, _ = _train_with_horizon(
        config=config, diag_trajs=diag_trajs, action_dim=action_dim,
        horizon=h, device=device, epochs=epochs, lr=lr,
        batch_size=batch_size, loss_threshold=1e-6,
        lambda_action=lambda_action, lambda_future=lambda_future,
        grad_clip=grad_clip, action_transform=action_transform,
        output_path=output_dir / "separate_loss_baseline_mse.csv",
    )
    rows.append({
        "variant": "pure_mse_baseline",
        "best_mse": baseline_mse,
        "best_epoch": -1,
        "continuous_loss_fn": "mse",
        "gripper_loss_fn": "mse",
    })

    return rows


# ---------------------------------------------------------------------------
# Lookup-table memorization
# ---------------------------------------------------------------------------


def _lookup_table_baseline(
    *, train_ds, val_ds, action_dim, device, epochs, lr,
    batch_size, loss_threshold, action_transform, output_path,
):
    """Verify the pipeline can reach near-zero MSE with a direct per-window lookup."""
    n_train = len(train_ds)
    n_val = len(val_ds)
    sample = train_ds[0]
    h = sample["target_actions"].shape[0] if hasattr(sample["target_actions"], "shape") else len(sample["target_actions"])

    class WindowLookup(nn.Module):
        """One learnable parameter vector per window."""
        def __init__(self, n_windows, action_horizon, action_dim):
            super().__init__()
            self.table = nn.Parameter(torch.zeros(n_windows, action_horizon, action_dim))

        def forward(self, idx):
            return self.table[idx]

    model = WindowLookup(n_train, h, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Build flat target tensors for train and val
    train_targets = torch.stack([
        torch.as_tensor(train_ds[i]["target_actions"], dtype=torch.float32)
        for i in range(n_train)
    ]).to(device)
    val_targets = torch.stack([
        torch.as_tensor(val_ds[i]["target_actions"], dtype=torch.float32)
        for i in range(n_val)
    ]).to(device)

    best_mse = float("inf")
    passed = False

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "split", "action_mse"])
        writer.writeheader()

        for epoch in range(epochs):
            model.train()
            pred = model(torch.arange(n_train))
            loss = nn.functional.mse_loss(pred, train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(torch.arange(min(n_val, n_train)))
                val_target = val_targets[:n_train] if n_val >= n_train else val_targets
                if action_transform is not None:
                    val_pred = action_transform.denormalize_tensor(val_pred)
                    val_target = action_transform.denormalize_tensor(val_target)
                val_mse = nn.functional.mse_loss(val_pred, val_target).item()

            writer.writerow({"epoch": epoch, "split": "val", "action_mse": val_mse})
            writer.writerow({"epoch": epoch, "split": "train", "action_mse": loss.item()})
            f.flush()

            if val_mse < best_mse:
                best_mse = val_mse
            if val_mse <= loss_threshold:
                passed = True
                break

    return best_mse, passed


# ---------------------------------------------------------------------------
# Mask/padding audit
# ---------------------------------------------------------------------------


def _audit_masks_and_padding(diag_trajs, config, action_dim):
    """Verify no padded timesteps contribute to loss and horizon masks are correct."""
    issues = []
    details = {}

    for traj in diag_trajs:
        T = traj.length
        h_len = int(config["data"]["history_len"])
        a_horizon = int(config["data"]["action_horizon"])
        f_horizon = int(config["data"]["future_horizon"])

        # Check that all actions are within valid range
        import numpy as np
        actions = np.array(traj.actions)
        if actions.shape[1] != action_dim:
            issues.append(f"action_dim mismatch: {actions.shape[1]} vs {action_dim}")

        # Check gripper values
        gripper_vals = actions[:, GRIPPER_DIM]
        unique_grip = set(np.unique(gripper_vals))
        details["gripper_unique_values"] = sorted(float(v) for v in unique_grip)
        details["gripper_range"] = [float(gripper_vals.min()), float(gripper_vals.max())]

        # Check trajectory length vs horizon requirements
        from src.data.trajectory_window import valid_time_indices
        valid_ts = valid_time_indices(
            T, history_len=h_len, action_horizon=a_horizon, future_horizon=f_horizon,
        )
        n_windows = len(valid_ts)
        details["trajectory_length"] = T
        details["valid_windows"] = n_windows
        details["excluded_edge_timesteps"] = T - n_windows

        if n_windows == 0:
            issues.append(f"trajectory {traj.trajectory_id}: zero valid windows")

        # Verify no window crosses episode boundary
        # (single trajectory = single episode, so this is automatically satisfied)
        details["episode_boundary_crossing"] = False

        # Check that action targets don't reference beyond trajectory
        if valid_ts:
            last_t = max(valid_ts)
            target_end = last_t + 1 + a_horizon
            if target_end > T:
                issues.append(f"target actions exceed trajectory length at t={last_t}")
            future_end = last_t + 1 + f_horizon
            if f_horizon > 0 and future_end > T:
                issues.append(f"future latents exceed trajectory length at t={last_t}")

    return {
        "issues": issues,
        "details": details,
        "has_issues": len(issues) > 0,
    }


# ---------------------------------------------------------------------------
# Classification and reporting
# ---------------------------------------------------------------------------


def _classify_failure(
    full_mse, cont_mse, grip_mse, h1_mse, h1_passed,
    lookup_mse, lookup_passed, horizon, threshold,
):
    """Classify the overfit failure root cause."""
    if full_mse <= threshold:
        return "PASS: full-horizon overfit succeeds"

    if not lookup_passed:
        return (
            "CRITICAL: lookup-table baseline fails. "
            "Basic pipeline/metric/loader problem. "
            "Check data loading, loss computation, and optimizer."
        )

    if not h1_passed:
        return (
            "H1_FAIL: single-step prediction cannot reach threshold. "
            "Basic action-target alignment or optimization problem. "
            "Check timestep alignment, action convention, and learning rate."
        )

    grip_fraction = grip_mse / max(full_mse, 1e-12)
    if grip_fraction > 0.5:
        return (
            "GRIPPER_DOMINANT: gripper dimension accounts for >50% of total error. "
            "Gripper is categorical {-1,+1} being regressed as continuous. "
            "Separate gripper loss (BCE) or clamp gripper predictions."
        )

    if horizon > 1 and h1_passed:
        return (
            "MULTI_HORIZON: H=1 passes but full horizon fails. "
            "Error compounds across prediction steps. "
            "Check late-horizon targets and consider horizon-weighted loss."
        )

    return (
        "MIXED: continuous dims and gripper both contribute significantly. "
        "Consider separate gripper loss, horizon weighting, and longer training."
    )


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _write_diagnostic_report(path, summary, decomposition, gripper_diag, ladder_rows, shift_rows):
    lines = [
        "# Overfit Diagnostic Report",
        "",
        f"## Full-Horizon Overfit (H={summary['full_horizon']})",
        f"- Best MSE: {summary['full_best_mse']:.6f} (threshold: {summary['loss_threshold']})",
        f"- Best epoch: {summary['full_best_epoch']}",
        f"- Passed: {summary['full_passed']}",
        "",
        "## Error Decomposition",
        f"- Continuous dims mean MSE: {summary['continuous_dims_mean_mse']:.6f}",
        f"- Gripper dim MSE: {summary['gripper_mse']:.6f}",
        f"- Gripper fraction of total: {summary['gripper_fraction_of_total']:.1%}",
        "",
        "## H=1 Gate",
        f"- Best MSE: {summary['h1_best_mse']:.6f}",
        f"- Passed: {summary['h1_passed']}",
        "",
        "## Lookup-Table Baseline",
        f"- Best MSE: {summary['lookup_table_best_mse']:.6f}",
        f"- Passed: {summary['lookup_table_passed']}",
        "",
        "## Diagnosis",
        summary["diagnosis"],
        "",
    ]

    if ladder_rows:
        lines.append("## Multi-Horizon Ladder")
        lines.append("| Horizon | Best MSE | Passed |")
        lines.append("|---------|----------|--------|")
        for r in ladder_rows:
            lines.append(f"| {r['horizon']} | {r['best_mse']:.6f} | {r['passed']} |")
        lines.append("")

    if shift_rows:
        lines.append("## Timestep Alignment Sweep")
        lines.append("| Shift | Best MSE |")
        lines.append("|-------|----------|")
        for r in shift_rows:
            lines.append(f"| {r['shift']:+d} | {r['best_mse']:.6f} |")
        lines.append("")

    grip = gripper_diag.get("summary", {})
    if grip:
        lines.append("## Gripper Diagnostics")
        for k, v in grip.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# H=1 repair diagnostics
# ---------------------------------------------------------------------------


class ShiftedTargetWindowDataset:
    """Single-demo diagnostic windows with explicit target action shift.

    For current time `t`, action history remains the project convention
    `actions[t-history_len+1:t+1]`. The H=1 target starts at
    `actions[t+1+target_shift]`.
    """

    def __init__(
        self,
        trajectories,
        *,
        split: str,
        config: Mapping[str, Any],
        action_horizon: int = 1,
        target_shift: int = 0,
    ) -> None:
        self.trajectories = [traj for traj in trajectories if traj.split == split]
        self.history_len = int(config["data"]["history_len"])
        self.action_horizon = action_horizon
        self.future_horizon = int(config["data"]["future_horizon"])
        self.include_current_latent = uses_current_latent(config)
        self.include_future_latents = requires_future_latents(config)
        self.target_shift = target_shift
        self._index: list[tuple[int, int]] = []
        for traj_index, traj in enumerate(self.trajectories):
            for t in range(traj.length):
                if self._valid_time(traj, t):
                    self._index.append((traj_index, t))

    def _valid_time(self, traj, t: int) -> bool:
        history_start = t - self.history_len + 1
        target_start = t + 1 + self.target_shift
        target_stop = target_start + self.action_horizon
        future_stop = t + 1 + self.future_horizon
        if history_start < 0 or target_start < 0 or target_stop > traj.length:
            return False
        if self.future_horizon > 0 and future_stop > traj.length:
            return False
        if (self.include_current_latent or self.include_future_latents) and traj.visual_latents is None:
            return False
        return True

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        traj_index, t = self._index[index]
        traj = self.trajectories[traj_index]
        history_start = t - self.history_len + 1
        history_stop = t + 1
        target_start = t + 1 + self.target_shift
        target_stop = target_start + self.action_horizon
        sample = {
            "trajectory_index": traj_index,
            "trajectory_id": traj.trajectory_id,
            "split": traj.split,
            "time_index": t,
            "target_shift": self.target_shift,
            "image_t": traj.images[t],
            "language": traj.language,
            "task_id": traj.task_id,
            "task_name": traj.task_name,
            "action_history": traj.actions[history_start:history_stop],
            "optional_state_t": None if traj.states is None else traj.states[t],
            "z_t": None if traj.visual_latents is None else traj.visual_latents[t],
            "target_actions": traj.actions[target_start:target_stop],
            "action_history_indices": list(range(history_start, history_stop)),
            "target_action_indices": list(range(target_start, target_stop)),
            "target_keys": ("target_actions",),
            "input_keys": ("image_t", "language", "action_history", "z_t"),
        }
        if self.include_future_latents:
            future_start = t + 1
            future_stop = future_start + self.future_horizon
            sample["target_future_latents"] = traj.visual_latents[future_start:future_stop]
            sample["target_future_indices"] = list(range(future_start, future_stop))
        return sample


class TimestepEmbeddingSplitMLP(nn.Module):
    """H=1 diagnostic baseline that predicts actions from timestep id only."""

    def __init__(self, *, max_time_index: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_time_index + 1, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, time_index: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.network(self.embedding(time_index))
        return self.action_head(features)


class LatentSplitMLP(nn.Module):
    """H=1 diagnostic baseline that predicts actions from current DINO latent."""

    def __init__(self, *, latent_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, z_t: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.action_head(self.network(z_t))


def run_h1_repair_diagnostics(
    *,
    config_path: Path,
    output_root: Path,
    run_id: str | None,
    trajectory_id: str | None,
    source_split: str,
    epochs: int,
    baseline_epochs: int,
    batch_size: int,
    lr: float | None,
    loss_threshold: float,
    device_name: str,
    seed: int,
    hidden_dim: int | None,
    command: Sequence[str] | None,
) -> Path:
    seed_everything(seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{timestamp}_h1_overfit_repair_seed{seed}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    config = load_config(config_path)
    validate_training_scope(config)
    effective_lr = float(lr if lr is not None else config["training"]["lr"])
    hidden = int(hidden_dim or config["model"]["hidden_dim"])
    device = torch.device(device_name)

    trajectories, source_metadata = load_real_libero_trajectories(config)
    selected = _select_trajectory(trajectories, trajectory_id, source_split)
    diagnostic_trajs = [replace(selected, split="train"), replace(selected, split="val")]
    action_transform, normalization_stats = build_action_transform(diagnostic_trajs, config)
    if action_transform is not None:
        diagnostic_trajs = apply_action_transform(diagnostic_trajs, action_transform)

    # Causal contract tests on shift=0 dataset
    causal_train_ds, causal_val_ds = _repair_datasets(diagnostic_trajs, config, 0)
    causal_contract = run_causal_contract_tests(causal_val_ds)
    _write_json(output_dir / "causal_contract_tests.json", causal_contract)

    # Latent sanity diagnostics
    latent_report = run_latent_sanity(selected, config=config, output_dir=output_dir)

    action_dim = len(diagnostic_trajs[0].actions[0])
    shift_rows: list[dict[str, Any]] = []
    shift_details: dict[int, dict[str, Any]] = {}
    for shift in [-1, 0, 1, 2]:
        train_ds, val_ds = _repair_datasets(diagnostic_trajs, config, shift)
        train_loader, val_loader = _repair_loaders(train_ds, val_ds, batch_size)
        sample = train_ds[0]
        model_config = _repair_config(config, action_horizon=1, split_gripper=False, hidden_dim=hidden)
        model = _build_model(model_config, sample, action_dim, device, action_horizon=1)
        result = _train_repair_model(
            model=model,
            model_kind="wam_gru",
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=effective_lr,
            weight_decay=0.0,
            loss_mode="mse",
            action_transform=action_transform,
            curve_path=output_dir / f"shift_{shift:+d}_debug_curves.csv",
            checkpoint_path=output_dir / f"wam_gru_shift_{shift:+d}_best.pt",
            log_debug=False,
        )
        metrics = result["best_metrics"]
        row = {
            "target_shift": shift,
            "train_mse": result["best_train_mse"],
            "eval_mse": metrics["action_mse"],
            "continuous_mse": metrics["continuous_mse"],
            "gripper_mse": metrics["gripper_mse"],
            "gripper_sign_accuracy": metrics["gripper_sign_accuracy"],
            "gripper_open_accuracy": metrics["gripper_open_accuracy"],
            "gripper_close_accuracy": metrics["gripper_close_accuracy"],
            "gripper_transition_f1": metrics["gripper_transition_f1"],
            "action_trace_correlation": metrics["action_correlation"],
            "continuous_trace_correlation": metrics["continuous_correlation"],
            "gripper_trace_correlation": metrics["gripper_correlation"],
            "best_epoch": result["best_epoch"],
            "passed": metrics["action_mse"] <= loss_threshold,
        }
        shift_rows.append(row)
        shift_details[shift] = result
    _write_csv(output_dir / "timestep_shift_train_sweep.csv", shift_rows)

    best_shift = min(shift_rows, key=lambda row: float(row["eval_mse"]))["target_shift"]
    best_shift = int(best_shift)
    train_ds, val_ds = _repair_datasets(diagnostic_trajs, config, best_shift)
    train_loader, val_loader = _repair_loaders(train_ds, val_ds, batch_size)
    sample = train_ds[0]

    split_config = _repair_config(config, action_horizon=1, split_gripper=True, hidden_dim=hidden)
    split_model = _build_model(split_config, sample, action_dim, device, action_horizon=1)
    split_result = _train_repair_model(
        model=split_model,
        model_kind="wam_gru",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=effective_lr,
        weight_decay=0.0,
        loss_mode="split",
        action_transform=action_transform,
        curve_path=output_dir / "overfit_debug_curves.csv",
        checkpoint_path=output_dir / "wam_gru_split_gripper_best.pt",
        log_debug=True,
    )

    timestep_model = TimestepEmbeddingSplitMLP(
        max_time_index=max(int(item["time_index"]) for item in (val_ds[i] for i in range(len(val_ds)))),
        hidden_dim=hidden,
        action_dim=action_dim,
    ).to(device)
    timestep_result = _train_repair_model(
        model=timestep_model,
        model_kind="timestep_mlp",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=baseline_epochs,
        lr=effective_lr,
        weight_decay=0.0,
        loss_mode="split",
        action_transform=action_transform,
        curve_path=output_dir / "timestep_embedding_mlp_debug_curves.csv",
        checkpoint_path=output_dir / "timestep_embedding_mlp_best.pt",
        log_debug=False,
    )

    latent_dim = infer_current_latent_dim(sample)
    latent_model = LatentSplitMLP(
        latent_dim=latent_dim,
        hidden_dim=hidden,
        action_dim=action_dim,
    ).to(device)
    latent_result = _train_repair_model(
        model=latent_model,
        model_kind="latent_mlp",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=baseline_epochs,
        lr=effective_lr,
        weight_decay=0.0,
        loss_mode="split",
        action_transform=action_transform,
        curve_path=output_dir / "dinov2_latent_mlp_debug_curves.csv",
        checkpoint_path=output_dir / "dinov2_latent_mlp_best.pt",
        log_debug=False,
    )

    split_rows = [
        _summary_row("wam_gru_split_gripper", best_shift, split_result, loss_threshold),
        _summary_row("timestep_embedding_mlp", best_shift, timestep_result, loss_threshold),
        _summary_row("dinov2_latent_mlp", best_shift, latent_result, loss_threshold),
    ]
    _write_csv(output_dir / "split_head_gripper_diagnostics.csv", split_rows)
    _write_csv(output_dir / "baseline_h1_overfit.csv", split_rows[1:])

    # ===================================================================
    # Causal H=1 baseline ladder (shift=0 only, non-leaking)
    # ===================================================================
    causal_train_ds, causal_val_ds = _repair_datasets(diagnostic_trajs, config, 0)
    causal_train_loader, causal_val_loader = _repair_loaders(causal_train_ds, causal_val_ds, batch_size)
    causal_sample = causal_train_ds[0]
    state_dim = infer_state_dim(causal_sample) or 0
    latent_dim_val = infer_current_latent_dim(causal_sample)

    causal_baselines: list[dict[str, Any]] = []

    # 1. Timestep embedding only (non-causal reference)
    timestep_causal_model = TimestepEmbeddingSplitMLP(
        max_time_index=max(int(causal_val_ds[i]["time_index"]) for i in range(len(causal_val_ds))),
        hidden_dim=hidden, action_dim=action_dim,
    ).to(device)
    timestep_causal_result = _train_repair_model(
        model=timestep_causal_model, model_kind="timestep_mlp",
        train_loader=causal_train_loader, val_loader=causal_val_loader,
        device=device, epochs=baseline_epochs, lr=effective_lr,
        weight_decay=0.0, loss_mode="split", action_transform=action_transform,
        curve_path=output_dir / "causal_timestep_embedding_debug_curves.csv",
        checkpoint_path=output_dir / "causal_timestep_embedding_best.pt", log_debug=False,
    )
    causal_baselines.append(_summary_row("causal_timestep_embedding", 0, timestep_causal_result, loss_threshold))

    # 2. Proprio only
    if state_dim > 0:
        proprio_model = ProprioSplitMLP(
            state_dim=state_dim, hidden_dim=hidden, action_dim=action_dim,
        ).to(device)
        proprio_result = _train_repair_model(
            model=proprio_model, model_kind="proprio",
            train_loader=causal_train_loader, val_loader=causal_val_loader,
            device=device, epochs=baseline_epochs, lr=effective_lr,
            weight_decay=0.0, loss_mode="split", action_transform=action_transform,
            curve_path=output_dir / "causal_proprio_debug_curves.csv",
            checkpoint_path=output_dir / "causal_proprio_best.pt", log_debug=False,
        )
        causal_baselines.append(_summary_row("causal_proprio_only", 0, proprio_result, loss_threshold))

    # 3. Action history only
    history_len = int(config["data"]["history_len"])
    action_hist_model = ActionHistorySplitGRU(
        history_len=history_len, action_dim=action_dim, hidden_dim=hidden,
    ).to(device)
    action_hist_result = _train_repair_model(
        model=action_hist_model, model_kind="action_history_gru",
        train_loader=causal_train_loader, val_loader=causal_val_loader,
        device=device, epochs=baseline_epochs, lr=effective_lr,
        weight_decay=0.0, loss_mode="split", action_transform=action_transform,
        curve_path=output_dir / "causal_action_history_debug_curves.csv",
        checkpoint_path=output_dir / "causal_action_history_best.pt", log_debug=False,
    )
    causal_baselines.append(_summary_row("causal_action_history_gru", 0, action_hist_result, loss_threshold))

    # 4. DINO CLS only (already computed as dinov2_latent_mlp on best_shift, but re-run on shift=0)
    dino_causal_model = LatentSplitMLP(
        latent_dim=latent_dim_val, hidden_dim=hidden, action_dim=action_dim,
    ).to(device)
    dino_causal_result = _train_repair_model(
        model=dino_causal_model, model_kind="latent_mlp",
        train_loader=causal_train_loader, val_loader=causal_val_loader,
        device=device, epochs=baseline_epochs, lr=effective_lr,
        weight_decay=0.0, loss_mode="split", action_transform=action_transform,
        curve_path=output_dir / "causal_dino_cls_debug_curves.csv",
        checkpoint_path=output_dir / "causal_dino_cls_best.pt", log_debug=False,
    )
    causal_baselines.append(_summary_row("causal_dino_cls_only", 0, dino_causal_result, loss_threshold))

    # 5. DINO CLS + proprio
    if state_dim > 0:
        dino_prop_model = DinoProprioSplitMLP(
            latent_dim=latent_dim_val, state_dim=state_dim, hidden_dim=hidden, action_dim=action_dim,
        ).to(device)
        dino_prop_result = _train_repair_model(
            model=dino_prop_model, model_kind="dino_proprio",
            train_loader=causal_train_loader, val_loader=causal_val_loader,
            device=device, epochs=baseline_epochs, lr=effective_lr,
            weight_decay=0.0, loss_mode="split", action_transform=action_transform,
            curve_path=output_dir / "causal_dino_proprio_debug_curves.csv",
            checkpoint_path=output_dir / "causal_dino_proprio_best.pt", log_debug=False,
        )
        causal_baselines.append(_summary_row("causal_dino_proprio", 0, dino_prop_result, loss_threshold))

    # 6. DINO CLS + proprio + action history
    if state_dim > 0:
        dino_prop_hist_model = DinoProprioHistorySplitGRU(
            latent_dim=latent_dim_val, state_dim=state_dim,
            action_dim=action_dim, history_len=history_len, hidden_dim=hidden,
        ).to(device)
        dino_prop_hist_result = _train_repair_model(
            model=dino_prop_hist_model, model_kind="dino_proprio_history_gru",
            train_loader=causal_train_loader, val_loader=causal_val_loader,
            device=device, epochs=baseline_epochs, lr=effective_lr,
            weight_decay=0.0, loss_mode="split", action_transform=action_transform,
            curve_path=output_dir / "causal_dino_proprio_history_debug_curves.csv",
            checkpoint_path=output_dir / "causal_dino_proprio_history_best.pt", log_debug=False,
        )
        causal_baselines.append(_summary_row("causal_dino_proprio_history_gru", 0, dino_prop_hist_result, loss_threshold))

    _write_csv(output_dir / "causal_h1_baseline_ladder.csv", causal_baselines)

    # Update _repair_forward to support new model kinds
    # (done via monkey-patching the dispatch; see _repair_forward below)

    audit = _repair_alignment_audit(
        config=config,
        selected=selected,
        action_transform=action_transform,
        best_shift=best_shift,
        shift_rows=shift_rows,
    )
    _write_json(output_dir / "overfit_alignment_audit.json", audit)

    clearly_nonzero = best_shift != 0 and _nonzero_shift_clearly_best(shift_rows)

    # Identify causal (non-leaking) baselines that pass
    causal_passed = [r for r in causal_baselines if r["passed"]]
    causal_passed_names = [r["variant"] for r in causal_passed]
    any_causal_passes = len(causal_passed) > 0

    # Mark shift=-1 as leakage
    leakage_shifts = [r for r in shift_rows if int(r["target_shift"]) == -1]
    shift_m1_leakage = len(leakage_shifts) > 0 and float(leakage_shifts[0]["eval_mse"]) < loss_threshold

    pipeline_valid = any_causal_passes and causal_contract.get("pass", False)
    summary = {
        "status": "h1_overfit_repair_diagnostic",
        "config": str(config_path),
        "trajectory_id": selected.trajectory_id,
        "best_target_shift": best_shift,
        "nonzero_shift_clearly_best": clearly_nonzero,
        "shift_minus1_leakage_detected": shift_m1_leakage,
        "causal_contract_pass": causal_contract.get("pass", False),
        "latent_sanity_pass": latent_report.get("pass", False),
        "loss_threshold": loss_threshold,
        "timestep_embedding_mlp_passed": split_rows[1]["passed"],
        "dinov2_latent_mlp_passed": split_rows[2]["passed"],
        "wam_gru_split_gripper_passed": split_rows[0]["passed"],
        "wam_gru_split_gripper_eval_mse": split_rows[0]["eval_mse"],
        "wam_gru_split_gripper_continuous_mse": split_rows[0]["continuous_mse"],
        "wam_gru_split_gripper_gripper_sign_accuracy": split_rows[0]["gripper_sign_accuracy"],
        "causal_baselines_passed": causal_passed_names,
        "any_causal_baseline_passes": any_causal_passes,
        "pipeline_valid_for_architecture_claims": pipeline_valid,
        "required_csvs": {
            "timestep_shift_train_sweep": str(output_dir / "timestep_shift_train_sweep.csv"),
            "split_head_gripper_diagnostics": str(output_dir / "split_head_gripper_diagnostics.csv"),
            "causal_h1_baseline_ladder": str(output_dir / "causal_h1_baseline_ladder.csv"),
            "causal_contract_tests": str(output_dir / "causal_contract_tests.json"),
            "latent_sanity": str(output_dir / "latent_sanity.json"),
            "overfit_debug_curves": str(output_dir / "overfit_debug_curves.csv"),
        },
        "non_claims": [
            "not_closed_loop_success",
            "not_future_latent_benefit_evidence",
            "not_architecture_claim_evidence"
            if not pipeline_valid
            else "h1_overfit_only_not_closed_loop",
        ],
    }
    summary["causal_baseline_ladder"] = causal_baselines
    _write_json(output_dir / "summary.json", summary)
    _write_repair_report(output_dir / "diagnostic_report.md", summary, shift_rows, split_rows, audit, causal_baselines)
    _write_repair_repro_files(
        output_dir=output_dir,
        config_path=config_path,
        command=command,
        source_metadata=source_metadata,
        normalization_stats=normalization_stats,
        seed=seed,
    )
    return output_dir


def _repair_datasets(diagnostic_trajs, config, target_shift: int):
    train_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs,
        split="train",
        config=config,
        action_horizon=1,
        target_shift=target_shift,
    )
    val_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs,
        split="val",
        config=config,
        action_horizon=1,
        target_shift=target_shift,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"target_shift={target_shift} produced zero valid H=1 windows")
    return train_ds, val_ds


def _repair_loaders(train_ds, val_ds, batch_size: int):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_action_batch),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_action_batch),
    )


def _repair_config(
    config: Mapping[str, Any],
    *,
    action_horizon: int,
    split_gripper: bool,
    hidden_dim: int,
) -> dict[str, Any]:
    cfg = {key: dict(value) if isinstance(value, Mapping) else value for key, value in config.items()}
    cfg["data"] = dict(config["data"])
    cfg["model"] = dict(config["model"])
    cfg["training"] = dict(config["training"])
    cfg["data"]["action_horizon"] = action_horizon
    cfg["model"]["hidden_dim"] = hidden_dim
    if split_gripper:
        cfg["model"]["action_head_type"] = "split_gripper"
    else:
        cfg["model"].pop("action_head_type", None)
    cfg["training"]["lambda_future"] = 0.0
    return cfg


def _train_repair_model(
    *,
    model,
    model_kind: str,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    loss_mode: str,
    action_transform,
    curve_path: Path,
    checkpoint_path: Path | None,
    log_debug: bool,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    fieldnames = _debug_curve_fieldnames()
    best_metrics: dict[str, Any] | None = None
    best_epoch = -1
    best_train_mse = float("inf")
    best_eval_mse = float("inf")

    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for epoch in range(epochs):
            train_loss, grad_norm, update_norm = _repair_train_epoch(
                model=model,
                model_kind=model_kind,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                loss_mode=loss_mode,
            )
            train_metrics = _repair_evaluate(
                model=model,
                model_kind=model_kind,
                loader=train_loader,
                device=device,
                action_transform=action_transform,
            )
            eval_metrics = _repair_evaluate(
                model=model,
                model_kind=model_kind,
                loader=val_loader,
                device=device,
                action_transform=action_transform,
            )
            train_metrics["loss"] = train_loss
            train_metrics["grad_norm"] = grad_norm
            train_metrics["action_head_update_norm"] = update_norm
            eval_metrics["loss"] = ""
            eval_metrics["grad_norm"] = ""
            eval_metrics["action_head_update_norm"] = ""
            if log_debug:
                writer.writerow(_debug_curve_row(epoch, "train", train_metrics))
                writer.writerow(_debug_curve_row(epoch, "eval", eval_metrics))
            elif epoch == epochs - 1 or eval_metrics["action_mse"] < best_eval_mse:
                writer.writerow(_debug_curve_row(epoch, "eval", eval_metrics))
            handle.flush()
            if eval_metrics["action_mse"] < best_eval_mse:
                best_eval_mse = eval_metrics["action_mse"]
                best_train_mse = train_metrics["action_mse"]
                best_metrics = dict(eval_metrics)
                best_epoch = epoch
                if checkpoint_path is not None:
                    _save_repair_checkpoint(
                        checkpoint_path,
                        model=model,
                        model_kind=model_kind,
                        best_epoch=best_epoch,
                        best_metrics=best_metrics,
                    )
    if best_metrics is None:
        raise RuntimeError("repair training produced no metrics")
    return {
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "best_train_mse": best_train_mse,
        "best_eval_mse": best_eval_mse,
    }


def _save_repair_checkpoint(
    path: Path,
    *,
    model,
    model_kind: str,
    best_epoch: int,
    best_metrics: Mapping[str, Any],
) -> None:
    payload = {
        "model_kind": model_kind,
        "best_epoch": best_epoch,
        "best_metrics": dict(best_metrics),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
    }
    torch.save(payload, path)


def _repair_train_epoch(
    *,
    model,
    model_kind: str,
    loader,
    device,
    optimizer,
    loss_mode: str,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    last_grad_norm = 0.0
    update_norm_sum = 0.0
    for batch in loader:
        before = _action_head_vector(model)
        outputs = _repair_forward(model, model_kind, batch, device)
        target = batch["target_actions"].to(device)
        loss = _repair_loss(outputs, target, mode=loss_mode)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norm = _grad_norm(model)
        optimizer.step()
        after = _action_head_vector(model)
        if before.numel() and after.numel():
            update_norm_sum += float((after - before).norm().item())
        batch_size = int(target.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1), last_grad_norm, update_norm_sum


def _repair_forward(model, model_kind: str, batch, device):
    if model_kind == "wam_gru":
        return forward_offline_model(model, batch, device=device)
    if model_kind == "timestep_mlp":
        return model(torch.as_tensor(batch["time_index"], dtype=torch.long, device=device))
    if model_kind == "latent_mlp":
        return model(batch["z_t"].to(device))
    if model_kind == "proprio":
        return model(batch["optional_state_t"].to(device))
    if model_kind == "action_history_gru":
        return model(batch["action_history"].to(device))
    if model_kind == "dino_proprio":
        return model(batch["z_t"].to(device), batch["optional_state_t"].to(device))
    if model_kind == "dino_proprio_history_gru":
        return model(
            batch["action_history"].to(device),
            batch["z_t"].to(device),
            batch["optional_state_t"].to(device),
        )
    raise ValueError(f"unknown repair model kind: {model_kind}")


def _repair_loss(outputs, target, *, mode: str):
    pred = outputs["pred_actions"] if isinstance(outputs, Mapping) else outputs
    if mode == "mse":
        return nn.functional.mse_loss(pred, target)
    if mode != "split":
        raise ValueError(f"unknown repair loss mode: {mode}")
    if not isinstance(outputs, Mapping) or "pred_gripper_logits" not in outputs:
        raise ValueError("split repair loss requires split gripper outputs")
    continuous_loss = nn.functional.smooth_l1_loss(
        outputs["pred_continuous_actions"],
        target[..., :GRIPPER_DIM],
    )
    gripper_target = (target[..., GRIPPER_DIM] > 0).to(target.dtype)
    gripper_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["pred_gripper_logits"],
        gripper_target,
    )
    return continuous_loss + gripper_loss


def _repair_evaluate(
    *,
    model,
    model_kind: str,
    loader,
    device,
    action_transform,
) -> dict[str, Any]:
    model.eval()
    pred_rows = []
    target_rows = []
    with torch.no_grad():
        for batch in loader:
            outputs = _repair_forward(model, model_kind, batch, device)
            pred = outputs["pred_actions"] if isinstance(outputs, Mapping) else outputs
            target = batch["target_actions"].to(device)
            if action_transform is not None:
                pred = action_transform.denormalize_tensor(pred)
                target = action_transform.denormalize_tensor(target)
            pred_rows.append(pred.detach().cpu())
            target_rows.append(target.detach().cpu())
    pred_all = torch.cat(pred_rows, dim=0)
    target_all = torch.cat(target_rows, dim=0)
    return _repair_metrics_from_tensors(pred_all, target_all)


def _repair_metrics_from_tensors(pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    error = pred - target
    sq = error.pow(2)
    residual = error.mean(dim=(0, 1))
    pred_var = pred.var(dim=(0, 1), unbiased=False)
    target_var = target.var(dim=(0, 1), unbiased=False)
    metrics: dict[str, Any] = {
        "action_mse": float(sq.mean().item()),
        "continuous_mse": float(sq[..., :GRIPPER_DIM].mean().item()),
        "gripper_mse": float(sq[..., GRIPPER_DIM].mean().item()),
        "action_correlation": _pearson_corr(pred.flatten(), target.flatten()),
        "continuous_correlation": _pearson_corr(
            pred[..., :GRIPPER_DIM].flatten(),
            target[..., :GRIPPER_DIM].flatten(),
        ),
        "gripper_correlation": _pearson_corr(
            pred[..., GRIPPER_DIM].flatten(),
            target[..., GRIPPER_DIM].flatten(),
        ),
        "pred_variance": float(pred.var(unbiased=False).item()),
        "target_variance": float(target.var(unbiased=False).item()),
    }
    metrics.update(_gripper_metrics_from_tensors(pred[..., GRIPPER_DIM], target[..., GRIPPER_DIM]))
    for dim in range(pred.shape[-1]):
        metrics[f"residual_dim{dim}"] = float(residual[dim].item())
        metrics[f"pred_var_dim{dim}"] = float(pred_var[dim].item())
        metrics[f"target_var_dim{dim}"] = float(target_var[dim].item())
    return metrics


def _gripper_metrics_from_tensors(pred_g: torch.Tensor, target_g: torch.Tensor) -> dict[str, float]:
    pred_flat = pred_g.flatten()
    target_flat = target_g.flatten()
    pred_sign = torch.where(pred_flat >= 0, torch.ones_like(pred_flat), -torch.ones_like(pred_flat))
    target_sign = torch.where(target_flat >= 0, torch.ones_like(target_flat), -torch.ones_like(target_flat))
    open_mask = target_flat > GRIPPER_OPEN_THRESH
    close_mask = target_flat < GRIPPER_CLOSE_THRESH
    sign_acc = (pred_sign == target_sign).float().mean().item()
    open_acc = (pred_sign[open_mask] > 0).float().mean().item() if open_mask.any() else float("nan")
    close_acc = (pred_sign[close_mask] < 0).float().mean().item() if close_mask.any() else float("nan")
    transitions_true = _find_transitions((target_sign > 0).int())
    transitions_pred = _find_transitions((pred_sign > 0).int())
    precision, recall, f1 = _transition_f1(transitions_true, transitions_pred, tolerance=2)
    return {
        "gripper_sign_accuracy": float(sign_acc),
        "gripper_open_accuracy": float(open_acc),
        "gripper_close_accuracy": float(close_acc),
        "gripper_transition_precision": float(precision),
        "gripper_transition_recall": float(recall),
        "gripper_transition_f1": float(f1),
    }


def _pearson_corr(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom.item()) <= 1e-12:
        return float("nan")
    return float((left_centered * right_centered).sum().div(denom).item())


def _grad_norm(model) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _action_head_vector(model) -> torch.Tensor:
    module = getattr(model, "action_head", model)
    tensors = [
        parameter.detach().flatten().cpu()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return torch.cat(tensors) if tensors else torch.empty(0)


def _debug_curve_fieldnames() -> list[str]:
    base = [
        "epoch", "split", "loss", "action_mse", "continuous_mse", "gripper_mse",
        "gripper_sign_accuracy", "gripper_open_accuracy", "gripper_close_accuracy",
        "gripper_transition_f1", "action_correlation", "continuous_correlation",
        "gripper_correlation", "grad_norm", "action_head_update_norm",
        "pred_variance", "target_variance",
    ]
    for dim in range(7):
        base.extend([f"residual_dim{dim}", f"pred_var_dim{dim}", f"target_var_dim{dim}"])
    return base


def _debug_curve_row(epoch: int, split: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = {"epoch": epoch, "split": split}
    for key in _debug_curve_fieldnames():
        if key in {"epoch", "split"}:
            continue
        row[key] = metrics.get(key, "")
    return row


def _summary_row(name: str, target_shift: int, result: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    metrics = result["best_metrics"]
    return {
        "variant": name,
        "target_shift": target_shift,
        "train_mse": result["best_train_mse"],
        "eval_mse": metrics["action_mse"],
        "continuous_mse": metrics["continuous_mse"],
        "gripper_mse": metrics["gripper_mse"],
        "gripper_sign_accuracy": metrics["gripper_sign_accuracy"],
        "gripper_open_accuracy": metrics["gripper_open_accuracy"],
        "gripper_close_accuracy": metrics["gripper_close_accuracy"],
        "gripper_transition_f1": metrics["gripper_transition_f1"],
        "action_trace_correlation": metrics["action_correlation"],
        "continuous_trace_correlation": metrics["continuous_correlation"],
        "gripper_trace_correlation": metrics["gripper_correlation"],
        "best_epoch": result["best_epoch"],
        "passed": metrics["action_mse"] <= threshold,
    }


def _nonzero_shift_clearly_best(rows: Sequence[Mapping[str, Any]]) -> bool:
    by_shift = {int(row["target_shift"]): float(row["eval_mse"]) for row in rows}
    best_shift = min(by_shift, key=by_shift.get)
    if best_shift == 0:
        return False
    zero_mse = by_shift.get(0, float("inf"))
    best_mse = by_shift[best_shift]
    return (zero_mse - best_mse) / max(zero_mse, 1e-12) >= 0.05


def _repair_alignment_audit(
    *,
    config: Mapping[str, Any],
    selected,
    action_transform,
    best_shift: int,
    shift_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "action_convention": "action_to_current_obs",
        "nominal_target": "actions[t+1] for obs/latent at t",
        "best_target_shift": best_shift,
        "shift_definition": "H=1 target index is t + 1 + target_shift",
        "output_activation": {
            "continuous_dims_0_to_5": "linear",
            "regression_gripper_baseline": "linear",
            "split_gripper_head": "binary logits thresholded to -1/+1 env command",
        },
        "target_scaling": "raw_action_units",
        "action_normalization": "none" if action_transform is None else "standardize_train",
        "masking": "no masks; windows that would cross boundaries are excluded",
        "padding": "none",
        "action_dimension_order": [
            "delta_pos_x", "delta_pos_y", "delta_pos_z",
            "delta_rot_x", "delta_rot_y", "delta_rot_z", "gripper",
        ],
        "gripper_convention": {
            "open_command": 1.0,
            "close_command": -1.0,
            "classification_target": "target_gripper > 0",
        },
        "trajectory_id": selected.trajectory_id,
        "trajectory_length": selected.length,
        "history_len": int(config["data"]["history_len"]),
        "action_horizon": 1,
        "future_horizon": int(config["data"]["future_horizon"]),
        "shift_rows": list(shift_rows),
    }


def _write_repair_repro_files(
    *,
    output_dir: Path,
    config_path: Path,
    command: Sequence[str] | None,
    source_metadata: Mapping[str, Any],
    normalization_stats: Mapping[str, Any],
    seed: int,
) -> None:
    shutil.copyfile(config_path, output_dir / "config.yaml")
    (output_dir / "command.txt").write_text(
        (" ".join(command) if command else " ".join(sys.argv)) + "\n",
        encoding="utf-8",
    )
    (output_dir / "git_commit.txt").write_text(capture_git_commit(), encoding="utf-8")
    (output_dir / "environment.txt").write_text(capture_environment(), encoding="utf-8")
    (output_dir / "seeds.txt").write_text(f"{seed}\n", encoding="utf-8")
    _write_json(output_dir / "split.json", source_metadata)
    _write_json(output_dir / "normalization_stats.json", normalization_stats)
    (output_dir / "notes.md").write_text(
        "H=1 single-demo overfit repair diagnostic only. "
        "This is not closed-loop or architecture-claim evidence.\n",
        encoding="utf-8",
    )


def _write_repair_report(
    path: Path,
    summary: Mapping[str, Any],
    shift_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    causal_baselines: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    lines = [
        "# H=1 Overfit Repair Diagnostic",
        "",
        f"Best target shift: `{summary['best_target_shift']}`",
        f"Nonzero shift clearly best: `{summary['nonzero_shift_clearly_best']}`",
        f"Pipeline valid for architecture claims: `{summary['pipeline_valid_for_architecture_claims']}`",
        "",
        "## Target Shift Sweep",
        "| Shift | Eval MSE | Continuous MSE | Gripper MSE | Corr | Passed |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in shift_rows:
        lines.append(
            f"| {int(row['target_shift']):+d} | {float(row['eval_mse']):.8g} | "
            f"{float(row['continuous_mse']):.8g} | {float(row['gripper_mse']):.8g} | "
            f"{float(row['action_trace_correlation']):.6g} | {row['passed']} |"
        )
    lines.extend([
        "",
        "## Split Head And Baselines",
        "| Variant | Eval MSE | Continuous MSE | Gripper sign acc | Passed |",
        "|---|---:|---:|---:|---|",
    ])
    for row in split_rows:
        lines.append(
            f"| {row['variant']} | {float(row['eval_mse']):.8g} | "
            f"{float(row['continuous_mse']):.8g} | "
            f"{float(row['gripper_sign_accuracy']):.6g} | {row['passed']} |"
        )
    if summary.get("shift_minus1_leakage_detected"):
        lines.extend([
            "",
            "## LEAKAGE WARNING",
            "target_shift=-1 achieved below-threshold MSE. This predicts actions[t] "
            "that are already present in action_history, so it is a leakage condition "
            "and does not validate next-action policy learning.",
        ])

    if causal_baselines:
        lines.extend([
            "",
            "## Causal H=1 Baseline Ladder (shift=0 only)",
            "| Variant | Eval MSE | Continuous MSE | Gripper sign acc | Passed |",
            "|---|---:|---:|---:|---|",
        ])
        for row in causal_baselines:
            lines.append(
                f"| {row['variant']} | {float(row['eval_mse']):.8g} | "
                f"{float(row['continuous_mse']):.8g} | "
                f"{float(row['gripper_sign_accuracy']):.6g} | {row['passed']} |"
            )

    causal_passed = summary.get("causal_baselines_passed", [])
    if causal_passed:
        lines.extend(["", f"Causal baselines that pass: {', '.join(causal_passed)}"])
    else:
        lines.extend(["", "No causal (non-leaking) baseline passes H=1 single-demo overfit."])

    lines.extend([
        "",
        "## Audit",
        f"- Causal contract pass: {summary.get('causal_contract_pass', 'N/A')}",
        f"- Latent sanity pass: {summary.get('latent_sanity_pass', 'N/A')}",
        f"- Nominal target: {audit['nominal_target']}",
        f"- Target scaling: {audit['target_scaling']}",
        f"- Action normalization: {audit['action_normalization']}",
        f"- Masking: {audit['masking']}",
        f"- Padding: {audit['padding']}",
        "",
        "No future-latent, closed-loop, or architecture benefit claim follows from this diagnostic.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
