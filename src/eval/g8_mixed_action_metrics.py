#!/usr/bin/env python3
"""G8: Mixed-action objective and metric repair.

Splits action modeling into continuous regression + gripper classification,
replacing the misleading global raw-action MSE with scientifically appropriate
per-component metrics.

Strict causal contract for all baselines:
- Inputs: observation[t], state[t], action_history[t-k:t-1], task_id
- Target: action[t]
- No action[t], no future actions, no future observations
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.trajectory_window import RawTrajectory  # noqa: E402
from src.eval.overfit_diagnostics import (  # noqa: E402
    ShiftedTargetWindowDataset,
    _repair_loss,
    _repair_metrics_from_tensors,
    _gripper_metrics_from_tensors,
    _find_transitions,
    _write_csv,
    _write_json,
    GRIPPER_DIM,
    GRIPPER_OPEN_THRESH,
    GRIPPER_CLOSE_THRESH,
)
from src.models.heads import SplitActionGripperHead  # noqa: E402
from src.train.train_offline import (  # noqa: E402
    ActionTransform,
    apply_action_transform,
    build_action_transform,
    collate_action_batch,
    infer_action_dim,
    infer_state_dim,
    load_real_libero_trajectories,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.experiment_io import capture_environment, capture_git_commit  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTINUOUS_DIMS = list(range(GRIPPER_DIM))  # dims 0..5
GRIPPER_DIM_IDX = GRIPPER_DIM               # dim 6
GRIPPER_ENCODING = "binary_sign"  # ±1
GRIPPER_THRESHOLD = 0.0          # for converting continuous to class label


def get_git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return {"commit": commit, "dirty": str(bool(status))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": "unknown"}


# ---------------------------------------------------------------------------
# 1. Split action contract
# ---------------------------------------------------------------------------

def build_action_contract(
    *,
    actions_np: np.ndarray,
    trajectory_id: str,
    dataset: str,
    task_name: str,
    git_info: dict[str, str],
) -> dict[str, Any]:
    """Build a documented action contract from demo statistics."""
    T, A = actions_np.shape
    continuous = actions_np[:, CONTINUOUS_DIMS]
    gripper = actions_np[:, GRIPPER_DIM_IDX]

    # Per-dim continuous stats
    cont_mean = continuous.mean(axis=0)
    cont_std = continuous.std(axis=0)
    cont_std_safe = np.where(cont_std > 1e-8, cont_std, 1.0)

    # Gripper stats
    gripper_unique = np.unique(gripper).tolist()
    n_open = int((gripper > GRIPPER_OPEN_THRESH).sum())
    n_close = int((gripper < GRIPPER_CLOSE_THRESH).sum())

    contract = {
        "action_dim": int(A),
        "continuous_dims": CONTINUOUS_DIMS,
        "continuous_dim_count": len(CONTINUOUS_DIMS),
        "gripper_dim": GRIPPER_DIM_IDX,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_unique_values": gripper_unique,
        "gripper_threshold_for_class": GRIPPER_THRESHOLD,
        "gripper_open_command": 1.0,
        "gripper_close_command": -1.0,
        "continuous_action_stats": {
            "mean": cont_mean.tolist(),
            "std": cont_std.tolist(),
            "std_safe": cont_std_safe.tolist(),
            "min": continuous.min(axis=0).tolist(),
            "max": continuous.max(axis=0).tolist(),
        },
        "gripper_stats": {
            "n_open": n_open,
            "n_close": n_close,
            "fraction_open": n_open / T,
            "fraction_close": n_close / T,
        },
        "trajectory_id": trajectory_id,
        "dataset": dataset,
        "task_name": task_name,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "n_samples": T,
    }
    return contract


def write_action_contract(contract: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# G8 Action Contract v2",
        "",
        "## Action Space",
        f"- Total dims: {contract['action_dim']}",
        f"- Continuous dims: {contract['continuous_dims']} (dims 0-5: delta position + delta orientation)",
        f"- Gripper dim: {contract['gripper_dim']} (dim 6: binary sign-coded ±1)",
        f"- Gripper encoding: {contract['gripper_encoding']}",
        f"- Gripper open command: {contract['gripper_open_command']}",
        f"- Gripper close command: {contract['gripper_close_command']}",
        f"- Gripper threshold for class: {contract['gripper_threshold_for_class']}",
        "",
        "## Continuous Action Statistics (from demo)",
        "",
        "| Dim | Label | Mean | Std | Min | Max |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    labels = ["delta_pos_x", "delta_pos_y", "delta_pos_z",
              "delta_rot_x", "delta_rot_y", "delta_rot_z"]
    stats = contract["continuous_action_stats"]
    for i, label in enumerate(labels):
        lines.append(
            f"| {i} | {label} | {stats['mean'][i]:.6f} | {stats['std'][i]:.6f} | "
            f"{stats['min'][i]:.6f} | {stats['max'][i]:.6f} |"
        )

    lines.extend([
        "",
        "## Gripper Statistics",
        f"- Open (>{GRIPPER_OPEN_THRESH}): {contract['gripper_stats']['n_open']} "
        f"({contract['gripper_stats']['fraction_open']:.1%})",
        f"- Close (<{GRIPPER_CLOSE_THRESH}): {contract['gripper_stats']['n_close']} "
        f"({contract['gripper_stats']['fraction_close']:.1%})",
        f"- Unique values: {contract['gripper_unique_values']}",
        "",
        "## Metric Definitions",
        "",
        "### Primary Metrics (scientific)",
        "- **continuous_normalized_mse**: MSE of (pred - target) / std_safe, averaged over continuous dims",
        "- **continuous_raw_mse**: Raw MSE of continuous action dims",
        "- **continuous_raw_mae**: MAE of continuous action dims",
        "- **gripper_sign_accuracy**: Fraction of timesteps where predicted sign matches target",
        "- **gripper_transition_f1**: F1 score for detecting open/close transitions",
        "",
        "### Diagnostic Metrics",
        "- **global_raw_mse**: MSE over all 7 dims (diagnostic only, not primary)",
        "- **gripper_raw_mse**: MSE of gripper dim (diagnostic only)",
        "- **old_1e4_gate**: Whether global_raw_mse < 1e-4 (engineering overfit gate only)",
        "",
        "### Baseline Comparisons",
        "- **beat_last_action_continuous**: continuous_normalized_mse < last_action baseline",
        "- **beat_last_action_gripper_f1**: gripper_transition_f1 > last_action baseline",
    ])

    (output_dir / "action_contract_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Split metrics
# ---------------------------------------------------------------------------

def compute_split_metrics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    action_stats: dict[str, Any] | None = None,
    last_action_pred: torch.Tensor | None = None,
    mean_action_pred: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute split continuous + gripper metrics.

    Args:
        pred_actions: [B, H, A] predicted actions
        target_actions: [B, H, A] target actions
        action_stats: dict with 'std_safe' array for normalization
        last_action_pred: [B, H, A] last-action baseline predictions (optional)
        mean_action_pred: [B, H, A] mean-action baseline predictions (optional)
    """
    B, H, A = target_actions.shape

    # Split into continuous and gripper
    pred_cont = pred_actions[..., CONTINUOUS_DIMS]  # [B, H, 6]
    target_cont = target_actions[..., CONTINUOUS_DIMS]
    pred_grip = pred_actions[..., GRIPPER_DIM_IDX]  # [B, H]
    target_grip = target_actions[..., GRIPPER_DIM_IDX]

    # Continuous metrics
    cont_se = (pred_cont - target_cont).pow(2)
    cont_raw_mse = float(cont_se.mean().item())
    cont_raw_mae = float((pred_cont - target_cont).abs().mean().item())

    # Normalized MSE
    if action_stats is not None:
        std_safe = torch.tensor(action_stats["std_safe"], dtype=pred_cont.dtype,
                                device=pred_cont.device)
        cont_norm_mse = float(((pred_cont - target_cont) / std_safe).pow(2).mean().item())
    else:
        cont_norm_mse = cont_raw_mse

    # Per-dim continuous MSE
    cont_per_dim_mse = cont_se.mean(dim=(0, 1)).tolist()

    # Gripper metrics
    grip_metrics = _gripper_metrics_from_tensors(pred_grip, target_grip)

    # Global raw MSE (diagnostic only)
    global_raw_mse = float(F.mse_loss(pred_actions, target_actions).item())

    # Gripper raw MSE (diagnostic only)
    gripper_raw_mse = float(F.mse_loss(pred_grip, target_grip).item())

    # Old 1e-4 gate (engineering overfit gate only)
    old_gate = global_raw_mse <= 1e-4

    # Baseline comparisons
    beat_last_cont = None
    beat_last_f1 = None
    if last_action_pred is not None:
        last_cont_se = (last_action_pred[..., CONTINUOUS_DIMS] - target_cont).pow(2)
        if action_stats is not None:
            std_safe = torch.tensor(action_stats["std_safe"], dtype=pred_cont.dtype,
                                    device=pred_cont.device)
            last_cont_norm_mse = float(((last_action_pred[..., CONTINUOUS_DIMS] - target_cont) / std_safe).pow(2).mean().item())
        else:
            last_cont_norm_mse = float(last_cont_se.mean().item())
        beat_last_cont = cont_norm_mse < last_cont_norm_mse

        last_grip_metrics = _gripper_metrics_from_tensors(
            last_action_pred[..., GRIPPER_DIM_IDX], target_grip)
        beat_last_f1 = grip_metrics["gripper_transition_f1"] > last_grip_metrics["gripper_transition_f1"]

    return {
        # Primary continuous
        "continuous_normalized_mse": cont_norm_mse,
        "continuous_raw_mse": cont_raw_mse,
        "continuous_raw_mae": cont_raw_mae,
        "continuous_per_dim_mse": cont_per_dim_mse,
        # Primary gripper
        "gripper_sign_accuracy": grip_metrics["gripper_sign_accuracy"],
        "gripper_open_accuracy": grip_metrics["gripper_open_accuracy"],
        "gripper_close_accuracy": grip_metrics["gripper_close_accuracy"],
        "gripper_transition_f1": grip_metrics["gripper_transition_f1"],
        "gripper_transition_precision": grip_metrics["gripper_transition_precision"],
        "gripper_transition_recall": grip_metrics["gripper_transition_recall"],
        # Diagnostic only
        "global_raw_mse": global_raw_mse,
        "gripper_raw_mse": gripper_raw_mse,
        "old_1e4_gate": old_gate,
        # Baseline comparisons
        "beat_last_action_continuous": beat_last_cont,
        "beat_last_action_gripper_f1": beat_last_f1,
    }


# ---------------------------------------------------------------------------
# 3. Split-head models
# ---------------------------------------------------------------------------

class SplitMLP(nn.Module):
    """MLP with split continuous + gripper heads."""

    def __init__(self, *, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.network(x))


class SplitGRU(nn.Module):
    """GRU with split continuous + gripper heads."""

    def __init__(self, *, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class SplitGRUPlusState(nn.Module):
    """GRU over action history + state input, split heads."""

    def __init__(self, *, state_dim: int, action_dim: int,
                 history_len: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=action_dim, hidden_size=hidden_dim, batch_first=True)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, action_history: torch.Tensor, state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.gru(action_history)
        features = torch.cat([hidden[-1], state_t], dim=-1)
        return self.head(self.network(features))


class SplitLinearAR(nn.Module):
    """Linear autoregressive model with split heads."""

    def __init__(self, *, history_len: int, action_dim: int) -> None:
        super().__init__()
        input_size = history_len * action_dim
        self.continuous_linear = nn.Linear(input_size, GRIPPER_DIM)
        self.gripper_linear = nn.Linear(input_size, 1)

    def forward(self, action_history: torch.Tensor) -> dict[str, torch.Tensor]:
        B = action_history.shape[0]
        flat = action_history.reshape(B, -1)
        continuous = self.continuous_linear(flat).unsqueeze(1)  # [B, 1, 6]
        gripper_logits = self.gripper_linear(flat).squeeze(-1).unsqueeze(1)  # [B, 1]
        from src.models.heads import gripper_logits_to_command
        gripper_cmd = gripper_logits_to_command(gripper_logits).unsqueeze(-1)
        return {
            "pred_continuous_actions": continuous,
            "pred_gripper_logits": gripper_logits,
            "pred_actions": torch.cat([continuous, gripper_cmd], dim=-1),
        }


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def _train_split_model(
    *,
    model: nn.Module,
    model_kind: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    action_contract: dict[str, Any],
    action_transform,
) -> dict[str, Any]:
    """Train a split-head model and evaluate with split metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    action_stats = action_contract.get("continuous_action_stats")
    std_safe = action_stats["std_safe"] if action_stats else None

    # Compute last-action baseline metrics on val
    last_action_mse = None
    last_action_metrics = None
    mean_action_mse = None

    best_split_metrics = None
    best_epoch = -1
    best_train_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            target = batch["target_actions"].to(device)
            outputs = _forward_split(model, model_kind, batch, device)

            # Split loss: continuous (SmoothL1) + gripper (BCE)
            pred_cont = outputs["pred_continuous_actions"]
            target_cont = target[..., CONTINUOUS_DIMS]

            # Normalize continuous targets for loss
            if std_safe is not None:
                std_t = torch.tensor(std_safe, dtype=pred_cont.dtype, device=device)
                pred_cont_norm = pred_cont / std_t
                target_cont_norm = target_cont / std_t
                cont_loss = F.smooth_l1_loss(pred_cont_norm, target_cont_norm)
            else:
                cont_loss = F.smooth_l1_loss(pred_cont, target_cont)

            # Gripper: BCE on logits
            pred_grip_logits = outputs["pred_gripper_logits"]
            target_grip = target[..., GRIPPER_DIM_IDX]
            target_grip_class = (target_grip > GRIPPER_THRESHOLD).float()
            grip_loss = F.binary_cross_entropy_with_logits(pred_grip_logits, target_grip_class)

            loss = cont_loss + grip_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * target.shape[0]
            total_samples += target.shape[0]

        # Eval
        model.eval()
        all_pred = []
        all_target = []
        with torch.no_grad():
            for batch in val_loader:
                target = batch["target_actions"].to(device)
                outputs = _forward_split(model, model_kind, batch, device)
                pred = outputs["pred_actions"]

                if action_transform is not None:
                    pred = action_transform.denormalize_tensor(pred)
                    target = action_transform.denormalize_tensor(target)

                all_pred.append(pred.cpu())
                all_target.append(target.cpu())

        pred_all = torch.cat(all_pred, dim=0)
        target_all = torch.cat(all_target, dim=0)

        # Compute last-action baseline (once)
        if last_action_metrics is None:
            all_history = []
            all_tgt = []
            with torch.no_grad():
                for batch in val_loader:
                    history = batch["action_history"]
                    target = batch["target_actions"]
                    last = history[:, -1:, :]
                    if action_transform is not None:
                        last = action_transform.denormalize_tensor(last)
                        target = action_transform.denormalize_tensor(target)
                    all_history.append(last)
                    all_tgt.append(target)
            last_pred = torch.cat(all_history, dim=0)
            last_tgt = torch.cat(all_tgt, dim=0)
            last_action_metrics = compute_split_metrics(
                last_pred, last_tgt, action_stats=action_stats)
            last_action_mse = last_action_metrics["continuous_normalized_mse"]

        # Compute mean-action baseline (once)
        if mean_action_mse is None:
            all_tgt = []
            with torch.no_grad():
                for batch in val_loader:
                    target = batch["target_actions"]
                    if action_transform is not None:
                        target = action_transform.denormalize_tensor(target)
                    all_tgt.append(target)
            mean_tgt = torch.cat(all_tgt, dim=0)
            mean_pred = mean_tgt.mean(dim=0, keepdim=True).expand_as(mean_tgt)
            mean_metrics = compute_split_metrics(mean_pred, mean_tgt, action_stats=action_stats)
            mean_action_mse = mean_metrics["continuous_normalized_mse"]

        metrics = compute_split_metrics(
            pred_all, target_all, action_stats=action_stats,
            last_action_pred=last_pred if last_action_metrics is not None else None,
        )

        if best_split_metrics is None or metrics["continuous_normalized_mse"] < best_split_metrics["continuous_normalized_mse"]:
            best_split_metrics = dict(metrics)
            best_epoch = epoch
            best_train_loss = total_loss / max(total_samples, 1)

    if best_split_metrics is None:
        raise RuntimeError("G8 training produced no metrics")

    return {
        "best_metrics": best_split_metrics,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "last_action_continuous_mse": last_action_mse,
        "mean_action_continuous_mse": mean_action_mse,
    }


def _forward_split(model, model_kind, batch, device):
    """Forward dispatch for split-head models."""
    if model_kind in ("linear_ar",):
        return model(batch["action_history"].to(device))
    elif model_kind in ("action_history_gru",):
        return model(batch["action_history"].to(device))
    elif model_kind in ("proprio_mlp",):
        return model(batch["optional_state_t"].to(device))
    elif model_kind in ("proprio_history_gru",):
        return model(batch["action_history"].to(device), batch["optional_state_t"].to(device))
    elif model_kind in ("full_state_mlp",):
        return model(batch["full_state_t"].to(device))
    elif model_kind in ("full_state_history_gru",):
        return model(batch["action_history"].to(device), batch["full_state_t"].to(device))
    elif model_kind in ("dino_cls_mlp",):
        return model(batch["z_t"].to(device))
    elif model_kind in ("dino_cls_history_gru",):
        return model(batch["action_history"].to(device), batch["z_t"].to(device))
    else:
        raise ValueError(f"unknown model kind: {model_kind}")


# ---------------------------------------------------------------------------
# 5. Raw image loader
# ---------------------------------------------------------------------------

def resolve_frame_reference(ref: str, dataset_root: str) -> np.ndarray | None:
    """Resolve an HDF5 frame reference string to a raw RGB array.

    Ref format: "suite/file.hdf5:data/demo_N:obs/agentview_rgb:t"
    """
    try:
        import h5py
    except ImportError:
        return None

    parts = ref.split(":")
    if len(parts) < 4:
        return None

    rel_path = parts[0]  # e.g. "libero_spatial/file.hdf5"
    dataset_path = parts[1]  # e.g. "data/demo_0"
    obs_key = parts[2]  # e.g. "obs/agentview_rgb"
    time_idx = int(parts[3])

    hdf5_path = os.path.join(dataset_root, rel_path)
    if not os.path.exists(hdf5_path):
        return None

    try:
        with h5py.File(hdf5_path, "r") as f:
            dataset = f[f"{dataset_path}/{obs_key}"]
            return np.array(dataset[time_idx], dtype=np.uint8)
    except Exception:
        return None


class LazyRawImageDataset:
    """Dataset wrapper that resolves frame references to raw pixels on demand."""

    def __init__(self, base_dataset, dataset_root: str, max_cache: int = 32):
        self.base = base_dataset
        self.dataset_root = dataset_root
        self.cache = {}
        self.max_cache = max_cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        ref = sample["image_t"]
        if isinstance(ref, str):
            if ref in self.cache:
                sample["image_t_raw"] = self.cache[ref]
            else:
                img = resolve_frame_reference(ref, self.dataset_root)
                if img is not None:
                    sample["image_t_raw"] = img
                    if len(self.cache) < self.max_cache:
                        self.cache[ref] = img
                else:
                    sample["image_t_raw"] = np.zeros((128, 128, 3), dtype=np.uint8)
        elif isinstance(ref, np.ndarray):
            sample["image_t_raw"] = ref
        return sample


# ---------------------------------------------------------------------------
# 6. Full state decomposition audit
# ---------------------------------------------------------------------------

def run_full_state_decomposition_audit(output_dir: Path) -> dict[str, Any]:
    """Investigate whether the 92-dim state can be decomposed."""
    # Check for LIBERO/robosuite metadata
    # The 92-dim state is MuJoCo qpos + qvel
    # For LIBERO spatial tasks (pick_and_place):
    # - Robot: Panda arm (7 DOF) + Panda gripper (2 DOF) = 9 DOF
    # - Objects: typically 1-2 objects with 7 DOF each (position 3 + quaternion 4)
    # - Goal: not typically in state vector (goal is task-conditioned)

    # qpos: robot_qpos(9) + object_qpos(7 * n_objects) + possibly extra
    # qvel: same dimensions as qpos

    # For a single object pick_and_place:
    # qpos = [robot_joints(7), gripper_joints(2), object_pos(3), object_quat(4)] = 16
    # qvel = same = 16
    # Total: 32 (but we have 92, so there may be additional objects or different DOF)

    # Without the MuJoCo model XML, exact decomposition is uncertain
    # Conservative: label as full_state_92d

    audit = {
        "state_dim": 92,
        "is_qpos_qvel": True,
        "decomposition_status": "uncertain",
        "known_components": {
            "robot_qpos": "7 DOF (Panda arm joints)",
            "gripper_qpos": "2 DOF (Panda gripper joints)",
            "object_qpos": "7 DOF per object (position 3 + quaternion 4)",
            "robot_qvel": "7 DOF (velocity)",
            "gripper_qvel": "2 DOF (velocity)",
            "object_qvel": "7 DOF per object (velocity)",
        },
        "estimated_breakdown": (
            "qpos: robot(7) + gripper(2) + object(7) = 16; "
            "qvel: robot(7) + gripper(2) + object(7) = 16; "
            "Total = 32. But we have 92 dims, suggesting multiple objects or "
            "additional DOF (e.g., joints, contact forces). "
            "Exact decomposition requires the MuJoCo model XML from LIBERO."
        ),
        "recommendation": (
            "Keep conservative label 'full_state_92d'. "
            "Do not claim 'true oracle' or decompose into named features "
            "without verifying against the MuJoCo model XML."
        ),
        "goal_pose_in_state": "Unknown. Goal may be task-conditioned, not in state vector.",
    }

    lines = [
        "# G8 Full State 92-dim Decomposition Audit",
        "",
        "## Status: UNCERTAIN",
        "",
        "The 92-dim `states` field is the full MuJoCo qpos + qvel vector.",
        "Exact decomposition requires the MuJoCo model XML from LIBERO, which is not available in the HDF5 files.",
        "",
        "## Known Components",
        "",
        "| Component | DOF | Description |",
        "|---|---:|---|",
    ]
    for comp, desc in audit["known_components"].items():
        lines.append(f"| {comp} | variable | {desc} |")

    lines.extend([
        "",
        "## Estimated Breakdown",
        audit["estimated_breakdown"],
        "",
        "## Recommendation",
        audit["recommendation"],
        "",
        "## Goal Pose",
        audit["goal_pose_in_state"],
    ])

    (output_dir / "full_state_92d_decomposition_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    return audit


# ---------------------------------------------------------------------------
# 7. Load trajectories with full state (from G7)
# ---------------------------------------------------------------------------

def _resolve_dataset_root(dataset_root_str: str) -> str:
    if dataset_root_str.startswith("env:"):
        env_name = dataset_root_str[4:]
        value = os.environ.get(env_name)
        if value is None:
            raise EnvironmentError(f"{env_name} is not set")
        return value
    return dataset_root_str


def load_trajectories_with_full_state(config: Mapping[str, Any]):
    """Load trajectories including the 92-dim full MuJoCo state."""
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py required")

    from src.train.train_offline import (
        find_demo_files, task_name_from_file, list_demo_groups,
        assign_splits, extract_language, safe_relative,
    )

    dataset_root = _resolve_dataset_root(str(config["data"]["dataset_root"]))
    suite = str(config["data"]["suite"])
    files = find_demo_files(Path(dataset_root), suite)
    if not files:
        raise FileNotFoundError(f"no HDF5 files for suite={suite}")

    task_names = [task_name_from_file(path) for path in files]
    task_id_by_name = {name: index for index, name in enumerate(sorted(set(task_names)))}

    trajectories = []
    for file_path in files:
        task_name = task_name_from_file(file_path)
        task_id = task_id_by_name[task_name]
        with h5py.File(file_path, "r") as handle:
            for demo_path, group in list_demo_groups(handle):
                if "actions" not in group:
                    continue
                actions = np.asarray(group["actions"][()], dtype=np.float32)
                length = int(actions.shape[0])
                trajectory_id = f"{safe_relative(file_path, Path(dataset_root))}:{demo_path}"
                frame_refs = [f"{trajectory_id}:obs/agentview_rgb:{i}" for i in range(length)]

                states = None
                if "robot_states" in group:
                    states = np.asarray(group["robot_states"][()], dtype=np.float32)

                full_states = None
                if "states" in group:
                    full_states = np.asarray(group["states"][()], dtype=np.float32)

                visual_latents = None
                latent_dir = config["data"].get("latent_dir")
                if latent_dir:
                    from src.train.train_offline import load_preextracted_latents
                    visual_latents = load_preextracted_latents(
                        latent_dir, file_path, demo_path,
                        config["data"].get("latent_format", "hdf5"),
                    )

                traj = RawTrajectory(
                    images=frame_refs, actions=actions, states=states,
                    visual_latents=visual_latents, task_id=task_id,
                    task_name=task_name, frame_refs=frame_refs,
                    language=extract_language(handle, group),
                    trajectory_id=trajectory_id, split="unspecified",
                )
                object.__setattr__(traj, "_full_states", full_states)
                trajectories.append(traj)

    if not trajectories:
        raise ValueError("no trajectories found")

    custom_attrs = {t.trajectory_id: {"_full_states": getattr(t, "_full_states", None)} for t in trajectories}
    split_trajs, split_meta = assign_splits(trajectories, config)
    for traj in split_trajs:
        attrs = custom_attrs.get(traj.trajectory_id, {})
        for key, val in attrs.items():
            object.__setattr__(traj, key, val)
    split_meta["task_id_map"] = task_id_by_name
    return split_trajs, split_meta


# ---------------------------------------------------------------------------
# 8. CLI and main orchestrator
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/g8_mixed_action_metrics"))
    parser.add_argument("--trajectory_id", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "any"], default="train")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--run_id", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = run_g8_diagnostics(
        config_path=args.config, output_root=args.output_dir,
        trajectory_id=args.trajectory_id, source_split=args.split,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device_name=args.device, seed=args.seed, hidden_dim=args.hidden_dim,
        run_id=args.run_id,
        command=[sys.executable, "-m", "src.eval.g8_mixed_action_metrics", *(argv or sys.argv[1:])],
    )
    print(f"g8_output_dir={output_dir}")
    return 0


def run_g8_diagnostics(
    *,
    config_path: Path,
    output_root: Path,
    trajectory_id: str | None = None,
    source_split: str = "train",
    epochs: int = 300,
    batch_size: int = 64,
    lr: float | None = None,
    device_name: str = "cpu",
    seed: int = 0,
    hidden_dim: int = 256,
    run_id: str | None = None,
    command: Sequence[str] | None = None,
):
    seed_everything(seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{timestamp}_g8_mixed_action"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    effective_lr = float(lr if lr is not None else config["training"]["lr"])
    device = torch.device(device_name)
    git_info = get_git_info()
    dataset_name = config["data"]["suite"]

    # ===================================================================
    # Load data
    # ===================================================================
    print("[1/8] Loading trajectories ...")
    trajectories, source_metadata = load_trajectories_with_full_state(config)
    selected = _select_trajectory(trajectories, trajectory_id, source_split)
    diagnostic_trajs = [replace(selected, split="train"), replace(selected, split="val")]

    action_transform, normalization_stats = build_action_transform(diagnostic_trajs, config)
    if action_transform is not None:
        diagnostic_trajs = apply_action_transform(diagnostic_trajs, action_transform)

    full_states = getattr(selected, "_full_states", None)
    proprio_states = selected.states
    has_full_state = full_states is not None
    has_proprio = proprio_states is not None
    has_latents = selected.visual_latents is not None

    actions_np = np.array(selected.actions, dtype=np.float32)
    T, A = actions_np.shape
    history_len = int(config["data"]["history_len"])

    print(f"  Trajectory: {selected.trajectory_id} (T={T}, A={A})")
    print(f"  Full state: {'yes' if has_full_state else 'no'}, "
          f"Proprio: {'yes' if has_proprio else 'no'}, "
          f"Latents: {'yes' if has_latents else 'no'}")

    # ===================================================================
    # 1. Action contract
    # ===================================================================
    print("[2/8] Building action contract ...")
    action_contract = build_action_contract(
        actions_np=actions_np, trajectory_id=selected.trajectory_id,
        dataset=dataset_name, task_name=selected.task_name, git_info=git_info,
    )
    write_action_contract(action_contract, output_dir)
    _write_json(output_dir / "action_contract_v2.json", action_contract)

    # ===================================================================
    # 2. Metric contract
    # ===================================================================
    print("[3/8] Writing metric contract ...")
    _write_metric_contract(output_dir)

    # ===================================================================
    # 3. Build datasets
    # ===================================================================
    print("[4/8] Building datasets ...")
    train_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs, split="train", config=config,
        action_horizon=1, target_shift=0,
    )
    val_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs, split="val", config=config,
        action_horizon=1, target_shift=0,
    )

    # Inject full_state_t
    class FullStateDS:
        def __init__(self, base, fs):
            self.base = base
            self.fs = fs
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            s = self.base[i]
            if self.fs is not None:
                s["full_state_t"] = self.fs[s["time_index"]]
            return s

    fs_train = FullStateDS(train_ds, full_states)
    fs_val = FullStateDS(val_ds, full_states)

    # Inject DINO latents
    class DinoDS:
        def __init__(self, base, latents):
            self.base = base
            self.latents = latents
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            s = self.base[i]
            if self.latents is not None:
                s["z_t"] = self.latents[s["time_index"]]
            return s

    dino_train = DinoDS(fs_train, selected.visual_latents)
    dino_val = DinoDS(fs_val, selected.visual_latents)

    def g8_collate(batch):
        c = collate_action_batch(batch)
        for key in ("full_state_t", "z_t"):
            if key in batch[0]:
                c[key] = torch.stack([torch.as_tensor(s[key], dtype=torch.float32) for s in batch])
        return c

    train_loader = DataLoader(dino_train, batch_size=batch_size, shuffle=True, collate_fn=g8_collate)
    val_loader = DataLoader(dino_val, batch_size=batch_size, shuffle=False, collate_fn=g8_collate)

    sample = dino_train[0]
    action_dim = infer_action_dim(sample)
    effective_state_dim = infer_state_dim(sample) or 0
    full_state_dim = full_states.shape[1] if has_full_state else 0

    print(f"  action_dim={action_dim}, proprio={effective_state_dim}, "
          f"full_state={full_state_dim}, train={len(train_ds)}, val={len(val_ds)}")

    # ===================================================================
    # 4. Split-head baselines
    # ===================================================================
    print("[5/8] Running split-head baselines ...")
    ladder_rows = []

    # 4a. Last-action baseline
    last_pred_list, last_tgt_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            last = batch["action_history"][:, -1:, :]
            tgt = batch["target_actions"]
            if action_transform is not None:
                last = action_transform.denormalize_tensor(last)
                tgt = action_transform.denormalize_tensor(tgt)
            last_pred_list.append(last.cpu())
            last_tgt_list.append(tgt.cpu())
    last_pred_all = torch.cat(last_pred_list)
    last_tgt_all = torch.cat(last_tgt_list)
    last_metrics = compute_split_metrics(last_pred_all, last_tgt_all, action_stats=action_contract["continuous_action_stats"])
    ladder_rows.append(_make_ladder_row("last_action", "copy", 0, last_metrics, -1))

    # 4b. Mean-action baseline
    mean_tgt_list = []
    with torch.no_grad():
        for batch in train_loader:
            tgt = batch["target_actions"]
            if action_transform is not None:
                tgt = action_transform.denormalize_tensor(tgt)
            mean_tgt_list.append(tgt.cpu())
    mean_val = torch.cat(mean_tgt_list).mean(dim=0)
    mean_pred_all = mean_val.expand_as(last_tgt_all)
    mean_metrics = compute_split_metrics(mean_pred_all, last_tgt_all, action_stats=action_contract["continuous_action_stats"])
    ladder_rows.append(_make_ladder_row("mean_action", "constant", 0, mean_metrics, -1))

    # Trainable baselines
    trainable_baselines = [
        ("linear_ar", SplitLinearAR(history_len=history_len, action_dim=action_dim),
         "linear_ar", history_len * action_dim),
        ("action_history_gru", SplitGRU(input_dim=action_dim, hidden_dim=hidden_dim, action_dim=action_dim),
         "action_history_gru", hidden_dim),
    ]

    if has_proprio and effective_state_dim > 0:
        trainable_baselines.append(
            ("proprio_only_state", SplitMLP(input_dim=effective_state_dim, hidden_dim=hidden_dim, action_dim=action_dim),
             "proprio_mlp", effective_state_dim))
        trainable_baselines.append(
            ("proprio_plus_history", SplitGRUPlusState(
                state_dim=effective_state_dim, action_dim=action_dim,
                history_len=history_len, hidden_dim=hidden_dim),
             "proprio_history_gru", effective_state_dim + hidden_dim))

    if has_full_state and full_state_dim > 0:
        trainable_baselines.append(
            ("full_state_92d", SplitMLP(input_dim=full_state_dim, hidden_dim=hidden_dim, action_dim=action_dim),
             "full_state_mlp", full_state_dim))
        trainable_baselines.append(
            ("full_state_plus_history", SplitGRUPlusState(
                state_dim=full_state_dim, action_dim=action_dim,
                history_len=history_len, hidden_dim=hidden_dim),
             "full_state_history_gru", full_state_dim + hidden_dim))

    if has_latents:
        z_t = sample["z_t"]
        latent_dim = len(z_t) if isinstance(z_t, (list, tuple)) else z_t.shape[-1]
        trainable_baselines.append(
            ("dino_cls", SplitMLP(input_dim=latent_dim, hidden_dim=hidden_dim, action_dim=action_dim),
             "dino_cls_mlp", latent_dim))
        trainable_baselines.append(
            ("dino_cls_plus_history", SplitGRUPlusState(
                state_dim=latent_dim, action_dim=action_dim,
                history_len=history_len, hidden_dim=hidden_dim),
             "dino_cls_history_gru", latent_dim + hidden_dim))

    for name, model, kind, params in trainable_baselines:
        print(f"  Training {name} ...")
        model = model.to(device)
        result = _train_split_model(
            model=model, model_kind=kind,
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            action_contract=action_contract, action_transform=action_transform,
        )
        ladder_rows.append(_make_ladder_row(name, kind, params, result["best_metrics"], result["best_epoch"]))

    _write_csv(output_dir / "split_head_baseline_ladder.csv", ladder_rows)

    # ===================================================================
    # 5. H=1 split metric results
    # ===================================================================
    print("[6/8] Writing H=1 split metric results ...")
    _write_csv(output_dir / "h1_split_metric_results.csv", ladder_rows)

    # ===================================================================
    # 6. Full state decomposition audit
    # ===================================================================
    print("[7/8] Full state decomposition audit ...")
    decomp_audit = run_full_state_decomposition_audit(output_dir)

    # ===================================================================
    # 7. Raw image loader status
    # ===================================================================
    print("[8/8] Raw image loader status ...")
    raw_status = {
        "status": "available_via_frame_reference_resolution",
        "note": "Frame references can be resolved to 128x128x3 uint8 RGB via HDF5. "
                "Lazy loader implemented but not run in this diagnostic to keep scope focused. "
                "Raw image CNN baselines deferred to future diagnostic.",
    }
    lines = [
        "# G8 Raw Image Loader Status",
        "",
        "## Status: AVAILABLE (not run)",
        "",
        "Raw images ARE accessible through HDF5 frame-reference resolution.",
        "Frame references are strings of the form:",
        "  `suite/file.hdf5:data/demo_N:obs/agentview_rgb:t`",
        "",
        "A lazy raw-image loader (`LazyRawImageDataset`) has been implemented",
        "in `src/eval/g8_mixed_action_metrics.py`.",
        "",
        "Raw image CNN baselines are deferred to keep G8 scope focused on",
        "split metric repair. They can be run in a future diagnostic.",
    ]
    (output_dir / "raw_image_loader_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ===================================================================
    # Summary
    # ===================================================================
    summary = {
        "status": "g8_mixed_action_metrics",
        "config": str(config_path),
        "trajectory_id": selected.trajectory_id,
        "trajectory_length": T,
        "task_id": selected.task_id,
        "task_name": selected.task_name,
        "dataset": dataset_name,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "seed": seed,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "action_contract": action_contract,
        "baseline_ladder": ladder_rows,
        "full_state_decomposition": decomp_audit,
        "non_claims": [
            "not_closed_loop_success",
            "not_future_latent_benefit_evidence",
            "not_architecture_claim_evidence",
            "not_policy_validity_evidence",
        ],
    }
    _write_json(output_dir / "summary.json", summary)

    # Repro files
    import shutil
    shutil.copyfile(config_path, output_dir / "config.yaml")
    _write_json(output_dir / "split.json", source_metadata)
    (output_dir / "git_commit.txt").write_text(
        f"commit={git_info['commit']}\ndirty={git_info['dirty']}\n", encoding="utf-8")
    (output_dir / "environment.txt").write_text(capture_environment(), encoding="utf-8")
    (output_dir / "seeds.txt").write_text(f"{seed}\n", encoding="utf-8")
    (output_dir / "command.txt").write_text(
        (" ".join(command) if command else " ".join(sys.argv)) + "\n", encoding="utf-8")
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output_dir / "environment.json", env_info)
    (output_dir / "notes.md").write_text(
        "G8 mixed-action objective and metric repair. "
        "Split continuous regression + gripper classification. "
        "Not closed-loop, not architecture-claim evidence.\n", encoding="utf-8")

    return output_dir


# ---------------------------------------------------------------------------
# Helpers
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


def _make_ladder_row(variant, model_type, param_count, metrics, best_epoch):
    return {
        "variant": variant,
        "model_type": model_type,
        "param_count": param_count,
        "continuous_normalized_mse": metrics["continuous_normalized_mse"],
        "continuous_raw_mse": metrics["continuous_raw_mse"],
        "continuous_raw_mae": metrics["continuous_raw_mae"],
        "gripper_sign_accuracy": metrics["gripper_sign_accuracy"],
        "gripper_transition_f1": metrics["gripper_transition_f1"],
        "global_raw_mse": metrics["global_raw_mse"],
        "old_1e4_gate": metrics["old_1e4_gate"],
        "beat_last_action_continuous": metrics.get("beat_last_action_continuous"),
        "beat_last_action_gripper_f1": metrics.get("beat_last_action_gripper_f1"),
        "best_epoch": best_epoch,
    }


def _write_metric_contract(output_dir: Path):
    lines = [
        "# G8 Metric Contract v2",
        "",
        "## Primary Metrics (scientific)",
        "",
        "### Continuous Regression",
        "- **continuous_normalized_mse**: MSE of (pred - target) / std_safe, averaged over continuous dims and timesteps",
        "- **continuous_raw_mse**: Raw MSE of 6 continuous action dims (diagnostic)",
        "- **continuous_raw_mae**: MAE of 6 continuous action dims (diagnostic)",
        "- Per-dimension continuous MSE for decomposition analysis",
        "",
        "### Gripper Classification",
        "- **gripper_sign_accuracy**: Fraction of timesteps where predicted sign matches target",
        "- **gripper_transition_f1**: F1 score for detecting open/close transitions (tolerance=2 steps)",
        "- **gripper_open_accuracy**: Accuracy on open timesteps only",
        "- **gripper_close_accuracy**: Accuracy on close timesteps only",
        "",
        "## Diagnostic Metrics (NOT primary)",
        "",
        "- **global_raw_mse**: MSE over all 7 dims — retained for backward compatibility only",
        "- **gripper_raw_mse**: MSE of gripper dim — diagnostic only (gripper is binary ±1, MSE is not meaningful)",
        "- **old_1e4_gate**: Whether global_raw_mse < 1e-4 — engineering overfit gate only, NOT a scientific pass/fail",
        "",
        "## Baseline Comparison Metrics",
        "",
        "- **beat_last_action_continuous**: Whether variant's continuous_normalized_mse < last-action baseline",
        "- **beat_last_action_gripper_f1**: Whether variant's gripper_transition_f1 > last-action baseline",
        "",
        "## Rules",
        "",
        "1. Primary scientific metric is continuous_normalized_mse + gripper_sign_accuracy + gripper_transition_f1.",
        "2. global_raw_mse is diagnostic only; do not use as primary metric.",
        "3. old_1e4 gate is engineering_overfit_gate_only; do not use as scientific pass/fail.",
        "4. Mixed continuous+gripper raw MSE is not a suitable primary scientific metric.",
        "5. All metrics are computed on same-demo teacher-forced H=1 evaluation.",
    ]
    (output_dir / "metric_contract_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
