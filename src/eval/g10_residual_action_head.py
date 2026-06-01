#!/usr/bin/env python3
"""G10: Residual-action head and action-parameterization repair.

Promotes residual-action prediction from G9 diagnostic to controlled candidate
objective. Under strict causal_next_action_v1, compares direct-action vs
residual-action prediction with matched conditions.

Strict causal contract for all baselines:
- Inputs: observation[t], state[t], action_history[t-k:t-1], task_id
- Target: action[t] (direct) or action[t]-action[t-1] (residual)
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
    _gripper_metrics_from_tensors,
    _find_transitions,
    _write_json,
    GRIPPER_DIM,
    GRIPPER_OPEN_THRESH,
    GRIPPER_CLOSE_THRESH,
)
from src.eval.g8_mixed_action_metrics import (  # noqa: E402
    CONTINUOUS_DIMS,
    GRIPPER_DIM_IDX,
    GRIPPER_THRESHOLD,
    compute_split_metrics,
    build_action_contract,
    SplitMLP,
    SplitGRU,
    SplitGRUPlusState,
    SplitLinearAR,
    _forward_split,
    _train_split_model,
    load_trajectories_with_full_state,
)

from src.eval.g9_residual_action_repair import _write_csv_g9 as _write_csv_safe  # noqa: E402
from src.models.heads import SplitActionGripperHead, gripper_logits_to_command  # noqa: E402
from src.train.train_offline import (  # noqa: E402
    apply_action_transform,
    build_action_transform,
    collate_action_batch,
    infer_action_dim,
    infer_state_dim,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.experiment_io import capture_environment, capture_git_commit  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


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
# 1. Residual-action model heads
# ---------------------------------------------------------------------------

class ResidualSplitGRUPlusState(nn.Module):
    """GRU over action history + state, residual continuous + gripper classification."""

    def __init__(self, *, state_dim: int, action_dim: int,
                 history_len: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.gru = nn.GRU(input_size=action_dim, hidden_size=hidden_dim, batch_first=True)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Residual continuous head (predicts delta from last action)
        self.continuous_residual_head = nn.Linear(hidden_dim, GRIPPER_DIM)
        # Gripper classification head
        self.gripper_head = nn.Linear(hidden_dim, 1)

    def forward(self, action_history: torch.Tensor, state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.gru(action_history)
        features = torch.cat([hidden[-1], state_t], dim=-1)
        features = self.network(features)
        continuous_residual = self.continuous_residual_head(features).unsqueeze(1)  # [B, 1, 6]
        gripper_logits = self.gripper_head(features).squeeze(-1).unsqueeze(1)  # [B, 1]
        gripper_cmd = gripper_logits_to_command(gripper_logits).unsqueeze(-1)
        return {
            "pred_continuous_residual": continuous_residual,
            "pred_continuous_actions": None,  # requires last_action for reconstruction
            "pred_gripper_logits": gripper_logits,
            "pred_actions": None,  # requires last_action for reconstruction
        }


class ResidualSplitGRU(nn.Module):
    """GRU over action history only, residual continuous + gripper."""

    def __init__(self, *, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=action_dim, hidden_size=hidden_dim, batch_first=True)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.continuous_residual_head = nn.Linear(hidden_dim, GRIPPER_DIM)
        self.gripper_head = nn.Linear(hidden_dim, 1)

    def forward(self, action_history: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.gru(action_history)
        features = self.network(hidden[-1])
        continuous_residual = self.continuous_residual_head(features).unsqueeze(1)
        gripper_logits = self.gripper_head(features).squeeze(-1).unsqueeze(1)
        gripper_cmd = gripper_logits_to_command(gripper_logits).unsqueeze(-1)
        return {
            "pred_continuous_residual": continuous_residual,
            "pred_continuous_actions": None,
            "pred_gripper_logits": gripper_logits,
            "pred_actions": None,
        }


class ResidualSplitMLP(nn.Module):
    """MLP from state, residual continuous + gripper."""

    def __init__(self, *, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.continuous_residual_head = nn.Linear(hidden_dim, GRIPPER_DIM)
        self.gripper_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.network(x)
        continuous_residual = self.continuous_residual_head(features).unsqueeze(1)
        gripper_logits = self.gripper_head(features).squeeze(-1).unsqueeze(1)
        gripper_cmd = gripper_logits_to_command(gripper_logits).unsqueeze(-1)
        return {
            "pred_continuous_residual": continuous_residual,
            "pred_continuous_actions": None,
            "pred_gripper_logits": gripper_logits,
            "pred_actions": None,
        }


def reconstruct_from_residual(
    outputs: dict[str, torch.Tensor],
    last_action: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reconstruct predicted action from residual + last_action.

    reconstructed_continuous = last_action[..., CONTINUOUS_DIMS] + predicted_residual
    reconstructed_gripper = from gripper logits (classification, not residual)
    """
    residual = outputs["pred_continuous_residual"]  # [B, 1, 6]
    last_cont = last_action[..., CONTINUOUS_DIMS]  # [B, 1, 6]
    reconstructed_cont = last_cont + residual

    gripper_logits = outputs["pred_gripper_logits"]  # [B, 1]
    gripper_cmd = gripper_logits_to_command(gripper_logits).unsqueeze(-1)  # [B, 1, 1]

    reconstructed_actions = torch.cat([reconstructed_cont, gripper_cmd], dim=-1)

    return {
        "pred_continuous_actions": reconstructed_cont,
        "pred_gripper_logits": gripper_logits,
        "pred_actions": reconstructed_actions,
        "pred_continuous_residual": residual,
    }


# ---------------------------------------------------------------------------
# 2. Residual-action training
# ---------------------------------------------------------------------------

def _forward_residual(model, model_kind, batch, device):
    """Forward dispatch for residual models."""
    if model_kind == "residual_gru":
        return model(batch["action_history"].to(device))
    elif model_kind == "residual_gru_state":
        return model(batch["action_history"].to(device), batch["full_state_t"].to(device))
    elif model_kind == "residual_mlp":
        return model(batch["full_state_t"].to(device))
    else:
        raise ValueError(f"unknown residual model kind: {model_kind}")


def _train_residual_model(
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
    """Train a residual-action model and evaluate with reconstructed metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    action_stats = action_contract["continuous_action_stats"]
    std_safe = action_stats["std_safe"]

    best_recon_metrics = None
    best_epoch = -1
    best_train_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            target_actions = batch["target_actions"].to(device)
            history = batch["action_history"].to(device)
            last_action = history[:, -1:, :]  # [B, 1, A]

            # Compute residual target for continuous dims
            target_cont = target_actions[..., CONTINUOUS_DIMS]
            last_cont = last_action[..., CONTINUOUS_DIMS]
            residual_target = target_cont - last_cont  # [B, 1, 6]

            # Forward
            outputs = _forward_residual(model, model_kind, batch, device)

            # Residual continuous loss (normalized)
            pred_resid = outputs["pred_continuous_residual"]
            std_t = torch.tensor(std_safe, dtype=pred_resid.dtype, device=device)
            resid_loss = F.smooth_l1_loss(pred_resid / std_t, residual_target / std_t)

            # Gripper classification loss
            pred_grip_logits = outputs["pred_gripper_logits"]
            target_grip = target_actions[..., GRIPPER_DIM_IDX]
            target_grip_class = (target_grip > GRIPPER_THRESHOLD).float()
            grip_loss = F.binary_cross_entropy_with_logits(pred_grip_logits, target_grip_class)

            loss = resid_loss + grip_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * target_actions.shape[0]
            total_samples += target_actions.shape[0]

        # Eval: reconstruct actions and compute metrics
        model.eval()
        all_recon_pred = []
        all_target = []
        with torch.no_grad():
            for batch in val_loader:
                target_actions = batch["target_actions"].to(device)
                history = batch["action_history"].to(device)
                last_action = history[:, -1:, :]

                outputs = _forward_residual(model, model_kind, batch, device)
                recon = reconstruct_from_residual(outputs, last_action)

                pred = recon["pred_actions"]
                if action_transform is not None:
                    pred = action_transform.denormalize_tensor(pred)
                    target_actions = action_transform.denormalize_tensor(target_actions)

                all_recon_pred.append(pred.cpu())
                all_target.append(target_actions.cpu())

        pred_all = torch.cat(all_recon_pred)
        target_all = torch.cat(all_target)
        metrics = compute_split_metrics(pred_all, target_all, action_stats=action_stats)

        if best_recon_metrics is None or metrics["continuous_normalized_mse"] < best_recon_metrics["continuous_normalized_mse"]:
            best_recon_metrics = dict(metrics)
            best_epoch = epoch
            best_train_loss = total_loss / max(total_samples, 1)

    if best_recon_metrics is None:
        raise RuntimeError("G10 residual training produced no metrics")

    return {
        "best_metrics": best_recon_metrics,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
    }


# ---------------------------------------------------------------------------
# 3. Orientation-specific repair
# ---------------------------------------------------------------------------

class OrientationWeightedSplitGRUPlusState(nn.Module):
    """GRU + state with dimension-weighted continuous head."""

    def __init__(self, *, state_dim: int, action_dim: int,
                 history_len: int, hidden_dim: int,
                 rotation_weight: float = 5.0) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=action_dim, hidden_size=hidden_dim, batch_first=True)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = SplitActionGripperHead(hidden_dim, 1, action_dim)
        self.rotation_weight = rotation_weight

    def forward(self, action_history, state_t):
        _, hidden = self.gru(action_history)
        features = torch.cat([hidden[-1], state_t], dim=-1)
        return self.head(self.network(features))


class SeparatePosRotHead(nn.Module):
    """GRU + state with separate position and rotation prediction heads."""

    def __init__(self, *, state_dim: int, action_dim: int,
                 history_len: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=action_dim, hidden_size=hidden_dim, batch_first=True)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim), nn.ReLU(),
        )
        # Position head (dims 0-2)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        # Rotation head (dims 3-5)
        self.rot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.gripper_head = nn.Linear(hidden_dim, 1)

    def forward(self, action_history, state_t):
        _, hidden = self.gru(action_history)
        features = self.shared(torch.cat([hidden[-1], state_t], dim=-1))
        pos = self.pos_head(features).unsqueeze(1)  # [B, 1, 3]
        rot = self.rot_head(features).unsqueeze(1)  # [B, 1, 3]
        continuous = torch.cat([pos, rot], dim=-1)  # [B, 1, 6]
        gripper_logits = self.gripper_head(features).squeeze(-1).unsqueeze(1)
        gripper_cmd = gripper_logits_to_command(gripper_logits).unsqueeze(-1)
        return {
            "pred_continuous_actions": continuous,
            "pred_gripper_logits": gripper_logits,
            "pred_actions": torch.cat([continuous, gripper_cmd], dim=-1),
        }


def _train_orientation_variant(
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
    rotation_weight: float = 1.0,
) -> dict[str, Any]:
    """Train with optional dimension-weighted loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    action_stats = action_contract["continuous_action_stats"]
    std_safe = action_stats["std_safe"]

    # Build weight vector: rotation dims get higher weight
    weights = torch.ones(len(CONTINUOUS_DIMS))
    weights[3:6] = rotation_weight  # dims 3-5 are rotation

    best_metrics = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            target = batch["target_actions"].to(device)
            outputs = _forward_split(model, model_kind, batch, device)
            pred_cont = outputs["pred_continuous_actions"]
            target_cont = target[..., CONTINUOUS_DIMS]

            # Weighted normalized MSE
            std_t = torch.tensor(std_safe, dtype=pred_cont.dtype, device=device)
            w = weights.to(device)
            weighted_se = ((pred_cont - target_cont) / std_t).pow(2) * w
            cont_loss = weighted_se.mean()

            # Gripper loss
            pred_grip_logits = outputs["pred_gripper_logits"]
            target_grip = target[..., GRIPPER_DIM_IDX]
            target_grip_class = (target_grip > GRIPPER_THRESHOLD).float()
            grip_loss = F.binary_cross_entropy_with_logits(pred_grip_logits, target_grip_class)

            loss = cont_loss + grip_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for batch in val_loader:
                target = batch["target_actions"].to(device)
                outputs = _forward_split(model, model_kind, batch, device)
                pred = outputs["pred_actions"]
                if action_transform:
                    pred = action_transform.denormalize_tensor(pred)
                    target = action_transform.denormalize_tensor(target)
                all_pred.append(pred.cpu())
                all_tgt.append(target.cpu())
        pred_all = torch.cat(all_pred)
        tgt_all = torch.cat(all_tgt)
        metrics = compute_split_metrics(pred_all, tgt_all, action_stats=action_stats)

        if best_metrics is None or metrics["continuous_normalized_mse"] < best_metrics["continuous_normalized_mse"]:
            best_metrics = dict(metrics)
            best_epoch = epoch

    return {"best_metrics": best_metrics, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# 4. Multi-demo offline validation
# ---------------------------------------------------------------------------

def run_multidemo_validation(
    *,
    trajectories: list[RawTrajectory],
    config: Mapping[str, Any],
    device: torch.device,
    action_contract: dict[str, Any],
    action_transform,
    epochs: int,
    lr: float,
    hidden_dim: int,
    git_info: dict[str, str],
    dataset: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Train on one demo, validate on held-out demos from same task."""
    rows = []

    # Get demos from same task
    task_name = trajectories[0].task_name if trajectories else "unknown"
    same_task = [t for t in trajectories if t.task_name == task_name and t.length > 20]

    if len(same_task) < 2:
        rows.append({
            "variant": "multidemo_offline",
            "note": f"insufficient_demos_for_task_{task_name}",
            "n_demos": len(same_task),
        })
        _write_csv_safe(output_dir / "multidemo_offline_split_metrics.csv", rows)
        return rows

    # Use first demo as train, rest as val
    train_traj = same_task[0]
    val_trajs = same_task[1:min(6, len(same_task))]  # up to 5 val demos

    history_len = int(config["data"]["history_len"])

    for val_idx, val_traj in enumerate(val_trajs):
        # Build train dataset from train_traj
        train_diag = [replace(train_traj, split="train"), replace(train_traj, split="val")]
        if action_transform is not None:
            train_diag = apply_action_transform(train_diag, action_transform)

        train_ds = ShiftedTargetWindowDataset(
            train_diag, split="train", config=config,
            action_horizon=1, target_shift=0,
        )

        # Build val dataset from val_traj
        val_diag = [replace(val_traj, split="val")]
        if action_transform is not None:
            val_diag = apply_action_transform(val_diag, action_transform)

        val_ds = ShiftedTargetWindowDataset(
            val_diag, split="val", config=config,
            action_horizon=1, target_shift=0,
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            continue

        # Inject full_state
        class FSDS:
            def __init__(self, base, fs):
                self.base = base; self.fs = fs
            def __len__(self): return len(self.base)
            def __getitem__(self, i):
                s = self.base[i]
                if self.fs is not None:
                    s["full_state_t"] = self.fs[s["time_index"]]
                return s

        train_fs = FSDS(train_ds, getattr(train_traj, "_full_states", None))
        val_fs = FSDS(val_ds, getattr(val_traj, "_full_states", None))

        def md_collate(batch):
            c = collate_action_batch(batch)
            if "full_state_t" in batch[0]:
                c["full_state_t"] = torch.stack([torch.as_tensor(s["full_state_t"], dtype=torch.float32) for s in batch])
            return c

        train_loader = DataLoader(train_fs, batch_size=64, shuffle=True, collate_fn=md_collate)
        val_loader = DataLoader(val_fs, batch_size=64, shuffle=False, collate_fn=md_collate)

        sample = train_fs[0]
        action_dim = infer_action_dim(sample)
        full_state_dim = train_traj._full_states.shape[1] if hasattr(train_traj, '_full_states') and train_traj._full_states is not None else 0

        # Train direct-action model
        if full_state_dim > 0:
            direct_model = SplitGRUPlusState(
                state_dim=full_state_dim, action_dim=action_dim,
                history_len=history_len, hidden_dim=hidden_dim,
            ).to(device)
            direct_result = _train_split_model(
                model=direct_model, model_kind="full_state_history_gru",
                train_loader=train_loader, val_loader=val_loader,
                device=device, epochs=epochs, lr=lr,
                action_contract=action_contract, action_transform=action_transform,
            )
            direct_mse = direct_result["best_metrics"]["continuous_normalized_mse"]

            # Train residual model
            resid_model = ResidualSplitGRUPlusState(
                state_dim=full_state_dim, action_dim=action_dim,
                history_len=history_len, hidden_dim=hidden_dim,
            ).to(device)
            resid_result = _train_residual_model(
                model=resid_model, model_kind="residual_gru_state",
                train_loader=train_loader, val_loader=val_loader,
                device=device, epochs=epochs, lr=lr,
                action_contract=action_contract, action_transform=action_transform,
            )
            resid_mse = resid_result["best_metrics"]["continuous_normalized_mse"]

            rows.append({
                "variant": "multidemo_offline",
                "val_demo": val_traj.trajectory_id,
                "direct_continuous_normalized_mse": direct_mse,
                "residual_continuous_normalized_mse": resid_mse,
                "improvement_ratio": direct_mse / max(resid_mse, 1e-12),
                "n_train_windows": len(train_ds),
                "n_val_windows": len(val_ds),
            })

    _write_csv_safe(output_dir / "multidemo_offline_split_metrics.csv", rows)

    # Gripper transition audit across val demos
    gripper_rows = []
    for val_traj in val_trajs:
        actions = np.array(val_traj.actions, dtype=np.float32)
        gripper = actions[:, GRIPPER_DIM_IDX]
        gripper_binary = (gripper > 0).astype(int)
        transitions = sum(1 for t in range(1, len(gripper_binary)) if gripper_binary[t] != gripper_binary[t-1])
        gripper_rows.append({
            "demo": val_traj.trajectory_id,
            "length": len(gripper),
            "n_transitions": transitions,
            "fraction_open": float((gripper > GRIPPER_OPEN_THRESH).sum() / len(gripper)),
        })
    _write_csv_safe(output_dir / "multidemo_gripper_transition_metrics.csv", gripper_rows)

    return rows


# ---------------------------------------------------------------------------
# 5. Autoregressive history diagnostic
# ---------------------------------------------------------------------------

def run_autoregressive_diagnostic(
    *,
    model: nn.Module,
    model_kind: str,
    trajectory: RawTrajectory,
    config: Mapping[str, Any],
    device: torch.device,
    action_contract: dict[str, Any],
    action_transform,
    horizon: int = 10,
    git_info: dict[str, str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Roll predicted actions through history without environment interaction."""
    rows = []
    actions_np = np.array(trajectory.actions, dtype=np.float32)
    T = len(actions_np)
    history_len = int(config["data"]["history_len"])
    action_dim = actions_np.shape[1]

    # Get full states if available
    full_states = getattr(trajectory, "_full_states", None)

    model.eval()
    for start_t in range(history_len, T - horizon):
        # Teacher-forced: use ground-truth history
        tf_history = torch.tensor(
            actions_np[start_t - history_len:start_t],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)  # [1, H, A]

        if full_states is not None:
            state_t = torch.tensor(
                full_states[start_t], dtype=torch.float32,
            ).unsqueeze(0).to(device)  # [1, S]

        # Autoregressive: roll predicted actions
        ar_history = tf_history.clone()
        ar_predictions = []
        ar_targets = []

        for h in range(min(horizon, T - start_t)):
            with torch.no_grad():
                if model_kind == "full_state_history_gru":
                    outputs = model(ar_history, state_t)
                elif model_kind == "action_history_gru":
                    outputs = model(ar_history)
                elif model_kind == "residual_gru_state":
                    outputs = model(ar_history, state_t)
                    last_act = ar_history[:, -1:, :]
                    recon = reconstruct_from_residual(outputs, last_act)
                    outputs = recon
                else:
                    outputs = model(ar_history)

            pred = outputs["pred_actions"]  # [1, 1, A]
            ar_predictions.append(pred[0, 0].cpu().numpy())

            target_t = start_t + h
            if target_t < T:
                ar_targets.append(actions_np[target_t])

            # Shift history: drop oldest, append prediction
            ar_history = torch.cat([ar_history[:, 1:, :], pred], dim=1)

        if ar_predictions and ar_targets:
            pred_arr = np.array(ar_predictions[:len(ar_targets)])
            tgt_arr = np.array(ar_targets)

            # Compute metrics
            pred_t = torch.tensor(pred_arr, dtype=torch.float32).unsqueeze(0)
            tgt_t = torch.tensor(tgt_arr, dtype=torch.float32).unsqueeze(0)
            m = compute_split_metrics(pred_t, tgt_t, action_stats=action_contract["continuous_action_stats"])

            rows.append({
                "start_t": start_t,
                "horizon": len(ar_predictions),
                "continuous_normalized_mse": m["continuous_normalized_mse"],
                "gripper_sign_accuracy": m["gripper_sign_accuracy"],
                "global_raw_mse": m["global_raw_mse"],
                "mode": "autoregressive",
            })

    # Also compute teacher-forced for comparison
    tf_rows = []
    for start_t in range(history_len, T - horizon):
        tf_history = torch.tensor(
            actions_np[start_t - history_len:start_t],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        if full_states is not None:
            state_t = torch.tensor(
                full_states[start_t], dtype=torch.float32,
            ).unsqueeze(0).to(device)

        tf_preds = []
        tf_targets = []
        for h in range(min(horizon, T - start_t)):
            # Always use ground-truth history
            with torch.no_grad():
                if model_kind == "full_state_history_gru":
                    outputs = model(tf_history, state_t)
                elif model_kind == "action_history_gru":
                    outputs = model(tf_history)
                else:
                    outputs = model(tf_history)

            pred = outputs["pred_actions"]
            tf_preds.append(pred[0, 0].cpu().numpy())

            target_t = start_t + h
            if target_t < T:
                tf_targets.append(actions_np[target_t])

            # For teacher-forced, shift with ground truth
            gt = torch.tensor(
                actions_np[start_t + h], dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0).to(device)
            tf_history = torch.cat([tf_history[:, 1:, :], gt], dim=1)

        if tf_preds and tf_targets:
            pred_arr = np.array(tf_preds[:len(tf_targets)])
            tgt_arr = np.array(tf_targets)
            pred_t = torch.tensor(pred_arr, dtype=torch.float32).unsqueeze(0)
            tgt_t = torch.tensor(tgt_arr, dtype=torch.float32).unsqueeze(0)
            m = compute_split_metrics(pred_t, tgt_t, action_stats=action_contract["continuous_action_stats"])
            tf_rows.append({
                "start_t": start_t,
                "horizon": len(tf_preds),
                "continuous_normalized_mse": m["continuous_normalized_mse"],
                "gripper_sign_accuracy": m["gripper_sign_accuracy"],
                "mode": "teacher_forced",
            })

    all_rows = rows + tf_rows
    _write_csv_safe(output_dir / "autoregressive_history_diagnostic.csv", all_rows)
    return all_rows


# ---------------------------------------------------------------------------
# 6. CLI and main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/g10_residual_action_head"))
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
    output_dir = run_g10_diagnostics(
        config_path=args.config, output_root=args.output_dir,
        trajectory_id=args.trajectory_id, source_split=args.split,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device_name=args.device, seed=args.seed, hidden_dim=args.hidden_dim,
        run_id=args.run_id,
        command=[sys.executable, "-m", "src.eval.g10_residual_action_head", *(argv or sys.argv[1:])],
    )
    print(f"g10_output_dir={output_dir}")
    return 0


def run_g10_diagnostics(
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
    run_id = run_id or f"{timestamp}_g10_residual_head"
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
    print("[1/9] Loading trajectories ...")
    trajectories, source_metadata = load_trajectories_with_full_state(config)
    selected = _select_trajectory(trajectories, trajectory_id, source_split)
    diagnostic_trajs = [replace(selected, split="train"), replace(selected, split="val")]

    action_transform, normalization_stats = build_action_transform(diagnostic_trajs, config)
    if action_transform is not None:
        diagnostic_trajs = apply_action_transform(diagnostic_trajs, action_transform)

    full_states = getattr(selected, "_full_states", None)
    actions_np = np.array(selected.actions, dtype=np.float32)
    T, A = actions_np.shape
    history_len = int(config["data"]["history_len"])

    print(f"  Trajectory: {selected.trajectory_id} (T={T}, A={A})")

    # Build action contract
    action_contract = build_action_contract(
        actions_np=actions_np, trajectory_id=selected.trajectory_id,
        dataset=dataset_name, task_name=selected.task_name, git_info=git_info,
    )

    # Write residual action contract
    lines = [
        "# G10 Residual Action Contract",
        "",
        "## Target Variants",
        "",
        "### Direct Action",
        "- target = action[t]",
        "- Predicted action = model output",
        "",
        "### Residual Action",
        "- target = action[t] - action[t-1] (continuous dims only)",
        "- Predicted action = action[t-1] + predicted_residual",
        "- action[t-1] is available through action_history (last entry)",
        "- Gripper: classification (sign prediction), NOT residual regression",
        "",
        "## Causal Contract (preserved)",
        "- Inputs: observation[t], state[t], action_history[t-k:t-1], task_id",
        "- target is action[t] (direct) or action[t]-action[t-1] (residual)",
        "- input includes action[t-1] through action history",
        "- input must NOT include action[t], future actions, future observations",
        "",
        "## Reconstruction",
        "- reconstructed_continuous = last_action[..., :6] + predicted_residual",
        "- reconstructed_gripper = sign(predicted_gripper_logits)",
        "- reconstructed_action = cat([reconstructed_continuous, reconstructed_gripper])",
        "",
        "## Metrics",
        "- Residual-space MSE: for monitoring training progress only",
        "- Reconstructed-action metrics: primary scientific metrics",
        "- Both direct and residual models evaluated under same split metric contract",
    ]
    (output_dir / "residual_action_contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ===================================================================
    # Build datasets
    # ===================================================================
    print("[2/9] Building datasets ...")
    train_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs, split="train", config=config,
        action_horizon=1, target_shift=0,
    )
    val_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs, split="val", config=config,
        action_horizon=1, target_shift=0,
    )

    class FullStateDS:
        def __init__(self, base, fs):
            self.base = base; self.fs = fs
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            s = self.base[i]
            if self.fs is not None: s["full_state_t"] = self.fs[s["time_index"]]
            return s

    fs_train = FullStateDS(train_ds, full_states)
    fs_val = FullStateDS(val_ds, full_states)

    def g10_collate(batch):
        c = collate_action_batch(batch)
        if "full_state_t" in batch[0]:
            c["full_state_t"] = torch.stack([torch.as_tensor(s["full_state_t"], dtype=torch.float32) for s in batch])
        return c

    train_loader = DataLoader(fs_train, batch_size=batch_size, shuffle=True, collate_fn=g10_collate)
    val_loader = DataLoader(fs_val, batch_size=batch_size, shuffle=False, collate_fn=g10_collate)

    sample = fs_train[0]
    action_dim = infer_action_dim(sample)
    effective_state_dim = infer_state_dim(sample) or 0
    full_state_dim = full_states.shape[1] if full_states is not None else 0

    print(f"  action_dim={action_dim}, full_state={full_state_dim}, train={len(train_ds)}, val={len(val_ds)}")

    # ===================================================================
    # 2 & 3. Residual-action head baseline ladder + direct vs residual
    # ===================================================================
    print("[3/9] Residual-action head baseline ladder ...")
    comparison_rows = []

    # Define matched pairs: (name, direct_kind, residual_kind, input_setup)
    variants = [
        ("action_history_gru", "action_history_gru", "residual_gru", "history_only"),
        ("full_state_plus_history", "full_state_history_gru", "residual_gru_state", "full_state"),
    ]

    if effective_state_dim > 0:
        variants.append(("proprio_plus_history", "proprio_history_gru", None, "proprio"))

    if hasattr(selected, 'visual_latents') and selected.visual_latents is not None:
        # DINO variants need z_t injection
        pass  # skip for now to keep scope manageable

    for variant_name, direct_kind, residual_kind, input_type in variants:
        print(f"  Training {variant_name} (direct + residual) ...")

        # Direct-action model
        if direct_kind == "action_history_gru":
            direct_model = SplitGRU(input_dim=action_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
        elif direct_kind == "full_state_history_gru":
            direct_model = SplitGRUPlusState(
                state_dim=full_state_dim, action_dim=action_dim,
                history_len=history_len, hidden_dim=hidden_dim,
            ).to(device)
        elif direct_kind == "proprio_history_gru":
            direct_model = SplitGRUPlusState(
                state_dim=effective_state_dim, action_dim=action_dim,
                history_len=history_len, hidden_dim=hidden_dim,
            ).to(device)
        else:
            continue

        direct_result = _train_split_model(
            model=direct_model, model_kind=direct_kind,
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            action_contract=action_contract, action_transform=action_transform,
        )

        # Residual-action model
        if residual_kind is None:
            continue

        if residual_kind == "residual_gru":
            resid_model = ResidualSplitGRU(action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        elif residual_kind == "residual_gru_state":
            resid_model = ResidualSplitGRUPlusState(
                state_dim=full_state_dim, action_dim=action_dim,
                history_len=history_len, hidden_dim=hidden_dim,
            ).to(device)
        else:
            continue

        resid_result = _train_residual_model(
            model=resid_model, model_kind=residual_kind,
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            action_contract=action_contract, action_transform=action_transform,
        )

        dm = direct_result["best_metrics"]
        rm = resid_result["best_metrics"]

        comparison_rows.append({
            "variant": variant_name,
            "direct_continuous_normalized_mse": dm["continuous_normalized_mse"],
            "direct_continuous_raw_mse": dm["continuous_raw_mse"],
            "direct_gripper_sign_accuracy": dm["gripper_sign_accuracy"],
            "residual_continuous_normalized_mse": rm["continuous_normalized_mse"],
            "residual_continuous_raw_mse": rm["continuous_raw_mse"],
            "residual_gripper_sign_accuracy": rm["gripper_sign_accuracy"],
            "improvement_ratio": dm["continuous_normalized_mse"] / max(rm["continuous_normalized_mse"], 1e-12),
            "direct_beat_last_action": dm.get("beat_last_action_continuous", None),
            "residual_beat_last_action": rm.get("beat_last_action_continuous", None),
            "direct_best_epoch": direct_result["best_epoch"],
            "residual_best_epoch": resid_result["best_epoch"],
        })

    _write_csv_safe(output_dir / "direct_vs_residual_comparison.csv", comparison_rows)
    _write_csv_safe(output_dir / "residual_head_baseline_ladder.csv", comparison_rows)

    # ===================================================================
    # 4. Orientation-specific repair
    # ===================================================================
    print("[4/9] Orientation repair ladder ...")
    orient_rows = []

    if full_state_dim > 0:
        # Baseline: standard SplitGRUPlusState (no orientation weighting)
        base_model = SplitGRUPlusState(
            state_dim=full_state_dim, action_dim=action_dim,
            history_len=history_len, hidden_dim=hidden_dim,
        ).to(device)
        base_result = _train_split_model(
            model=base_model, model_kind="full_state_history_gru",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            action_contract=action_contract, action_transform=action_transform,
        )
        orient_rows.append({
            "variant": "baseline_no_weight",
            "rotation_weight": 1.0,
            "continuous_normalized_mse": base_result["best_metrics"]["continuous_normalized_mse"],
            "per_dim_mse": base_result["best_metrics"].get("continuous_per_dim_mse", []),
        })

        # Weighted: rotation_weight=5
        for rw in [2.0, 5.0, 10.0]:
            w_model = OrientationWeightedSplitGRUPlusState(
                state_dim=full_state_dim, action_dim=action_dim,
                history_len=history_len, hidden_dim=hidden_dim,
                rotation_weight=rw,
            ).to(device)
            w_result = _train_orientation_variant(
                model=w_model, model_kind="full_state_history_gru",
                train_loader=train_loader, val_loader=val_loader,
                device=device, epochs=epochs, lr=effective_lr,
                action_contract=action_contract, action_transform=action_transform,
                rotation_weight=rw,
            )
            orient_rows.append({
                "variant": f"rotation_weight_{rw}",
                "rotation_weight": rw,
                "continuous_normalized_mse": w_result["best_metrics"]["continuous_normalized_mse"],
                "per_dim_mse": w_result["best_metrics"].get("continuous_per_dim_mse", []),
            })

        # Separate pos/rot heads
        sep_model = SeparatePosRotHead(
            state_dim=full_state_dim, action_dim=action_dim,
            history_len=history_len, hidden_dim=hidden_dim,
        ).to(device)
        sep_result = _train_orientation_variant(
            model=sep_model, model_kind="full_state_history_gru",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            action_contract=action_contract, action_transform=action_transform,
            rotation_weight=1.0,
        )
        orient_rows.append({
            "variant": "separate_pos_rot_heads",
            "rotation_weight": 1.0,
            "continuous_normalized_mse": sep_result["best_metrics"]["continuous_normalized_mse"],
            "per_dim_mse": sep_result["best_metrics"].get("continuous_per_dim_mse", []),
        })

    _write_csv_safe(output_dir / "orientation_repair_ladder.csv", orient_rows)

    # ===================================================================
    # 5. Multi-demo offline validation
    # ===================================================================
    print("[5/9] Multi-demo offline validation ...")
    multidemo_rows = run_multidemo_validation(
        trajectories=trajectories, config=config, device=device,
        action_contract=action_contract, action_transform=action_transform,
        epochs=min(epochs, 200), lr=effective_lr, hidden_dim=hidden_dim,
        git_info=git_info, dataset=dataset_name, output_dir=output_dir,
    )

    # ===================================================================
    # 6. Autoregressive history diagnostic
    # ===================================================================
    print("[6/9] Autoregressive history diagnostic ...")
    # Train a best model for autoregressive test
    ar_model = SplitGRUPlusState(
        state_dim=full_state_dim, action_dim=action_dim,
        history_len=history_len, hidden_dim=hidden_dim,
    ).to(device)
    _train_split_model(
        model=ar_model, model_kind="full_state_history_gru",
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=epochs, lr=effective_lr,
        action_contract=action_contract, action_transform=action_transform,
    )
    ar_rows = run_autoregressive_diagnostic(
        model=ar_model, model_kind="full_state_history_gru",
        trajectory=selected, config=config, device=device,
        action_contract=action_contract, action_transform=action_transform,
        horizon=10, git_info=git_info, output_dir=output_dir,
    )

    # ===================================================================
    # 7. Raw image status
    # ===================================================================
    print("[7/9] Raw image loader status ...")
    (output_dir / "raw_image_residual_status.md").write_text(
        "# G10 Raw Image Residual Status\n\n"
        "## Status: DEFERRED\n\n"
        "Raw image loader was implemented in G8 (`LazyRawImageDataset`) but not run\n"
        "in G10 to keep scope focused on residual-action head comparison.\n"
        "Raw image CNN baselines can be added in a future diagnostic.\n",
        encoding="utf-8",
    )

    # ===================================================================
    # Summary
    # ===================================================================
    print("[8/9] Writing summary ...")
    best_direct = min((r["direct_continuous_normalized_mse"] for r in comparison_rows), default=float("inf"))
    best_residual = min((r["residual_continuous_normalized_mse"] for r in comparison_rows), default=float("inf"))

    summary = {
        "status": "g10_residual_action_head",
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
        "best_direct_continuous_normalized_mse": best_direct,
        "best_residual_continuous_normalized_mse": best_residual,
        "residual_improvement": best_direct / max(best_residual, 1e-12),
        "comparison": comparison_rows,
        "orientation_ladder": orient_rows,
        "multidemo": multidemo_rows,
        "non_claims": [
            "not_closed_loop_success",
            "not_future_latent_benefit_evidence",
            "not_architecture_claim_evidence",
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
        "G10 residual-action head and action-parameterization repair. "
        "Diagnostic only, not closed-loop or architecture evidence.\n", encoding="utf-8")

    print(f"[9/9] Done. Best direct={best_direct:.6f}, best residual={best_residual:.6f}")
    return output_dir


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


if __name__ == "__main__":
    raise SystemExit(main())
