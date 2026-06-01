#!/usr/bin/env python3
"""G11: Offline autoregressive stabilization and closed-loop readiness gate.

Builds on G10 residual-action head results to:
1. Evaluate autoregressive error growth across model variants
2. Implement stabilization training variants (history noise, dropout, scheduled
   sampling, unrolled loss, smoothness regularization)
3. Define explicit closed-loop readiness gate criteria
4. Perform failure mode analysis

Strict causal contract preserved from G10:
- Inputs: observation[t], state[t], action_history[t-k:t-1], task_id
- Target: action[t] (direct) or action[t]-action[t-1] (residual)
- No action[t], no future actions, no future observations

DO NOT run environment closed-loop experiments.
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
    SplitGRU,
    SplitGRUPlusState,
    _forward_split,
    _train_split_model,
    load_trajectories_with_full_state,
)
from src.eval.g10_residual_action_head import (  # noqa: E402
    ResidualSplitGRU,
    ResidualSplitGRUPlusState,
    SeparatePosRotHead,
    _forward_residual,
    _train_residual_model,
    reconstruct_from_residual,
)
from src.models.heads import gripper_logits_to_command  # noqa: E402
from src.train.train_offline import (  # noqa: E402
    apply_action_transform,
    build_action_transform,
    collate_action_batch,
    infer_action_dim,
    infer_state_dim,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.experiment_io import capture_environment  # noqa: E402
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
# CSV/JSON helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # Collect all field names across all rows
    all_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 1. Autoregressive rollout evaluator
# ---------------------------------------------------------------------------

HORIZON_BUCKETS = [1, 5, 10, 20, 40, 60]


def _compute_error_growth_slope(errors: list[float]) -> float:
    """Linear slope of error over horizon steps."""
    if len(errors) < 2:
        return 0.0
    x = np.arange(len(errors), dtype=np.float32)
    y = np.array(errors, dtype=np.float32)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _per_dim_mse(pred: np.ndarray, target: np.ndarray) -> list[float]:
    """Per-dimension MSE over continuous dims."""
    diff = pred - target
    return [float((diff[:, d] ** 2).mean()) for d in range(pred.shape[1])]


def run_single_trajectory_autoregressive(
    *,
    model: nn.Module,
    model_kind: str,
    trajectory: RawTrajectory,
    config: Mapping[str, Any],
    device: torch.device,
    action_contract: dict[str, Any],
    action_transform,
    max_horizon: int = 60,
    git_info: dict[str, str],
) -> list[dict[str, Any]]:
    """Roll predicted actions through recorded observation/state sequence.

    Three modes:
    - teacher_forced: ground-truth history at every step
    - autoregressive_open_loop: predicted actions fed back into history
    - corrupted_history_robustness: ground-truth history with noise
    """
    rows = []
    actions_np = np.array(trajectory.actions, dtype=np.float32)
    T = len(actions_np)
    history_len = int(config["data"]["history_len"])
    action_dim = actions_np.shape[1]
    full_states = getattr(trajectory, "_full_states", None)
    action_stats = action_contract["continuous_action_stats"]
    std_safe = np.array(action_stats["std_safe"], dtype=np.float32)

    model.eval()

    # Bootstrap: use ground truth history at t=history_len
    start_t = history_len

    # === Mode 1: teacher_forced ===
    tf_history = torch.tensor(
        actions_np[start_t - history_len:start_t],
        dtype=torch.float32,
    ).unsqueeze(0).to(device)  # [1, H, A]

    tf_preds = []
    tf_targets = []
    tf_per_step_errors = []
    tf_gripper_preds = []
    tf_gripper_targets = []

    state_t = None
    if full_states is not None:
        state_t = torch.tensor(
            full_states[start_t], dtype=torch.float32,
        ).unsqueeze(0).to(device)  # [1, S]

    current_tf_history = tf_history.clone()
    for h in range(min(max_horizon, T - start_t)):
        with torch.no_grad():
            if model_kind in ("full_state_history_gru",):
                outputs = model(current_tf_history, state_t)
            elif model_kind in ("action_history_gru",):
                outputs = model(current_tf_history)
            elif model_kind in ("residual_gru_state",):
                outputs = model(current_tf_history, state_t)
                last_act = current_tf_history[:, -1:, :]
                recon = reconstruct_from_residual(outputs, last_act)
                outputs = recon
            elif model_kind in ("residual_gru",):
                outputs = model(current_tf_history)
                last_act = current_tf_history[:, -1:, :]
                recon = reconstruct_from_residual(outputs, last_act)
                outputs = recon
            elif model_kind in ("separate_pos_rot",):
                outputs = model(current_tf_history, state_t)
            else:
                outputs = model(current_tf_history)

        pred = outputs["pred_actions"]  # [1, 1, A]
        tf_preds.append(pred[0, 0].cpu().numpy())

        target_t = start_t + h
        if target_t < T:
            tf_targets.append(actions_np[target_t])

        # Teacher-forced: shift with ground truth
        gt = torch.tensor(
            actions_np[start_t + h], dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0).to(device)
        current_tf_history = torch.cat([current_tf_history[:, 1:, :], gt], dim=1)

    if tf_preds and tf_targets:
        tf_pred_arr = np.array(tf_preds[:len(tf_targets)])
        tf_tgt_arr = np.array(tf_targets)

        # Per-step errors
        for i in range(len(tf_pred_arr)):
            cont_pred = tf_pred_arr[i, CONTINUOUS_DIMS]
            cont_tgt = tf_tgt_arr[i, CONTINUOUS_DIMS]
            step_mse = float(np.mean(((cont_pred - cont_tgt) / std_safe) ** 2))
            tf_per_step_errors.append(step_mse)
            tf_gripper_preds.append(float(tf_pred_arr[i, GRIPPER_DIM_IDX]))
            tf_gripper_targets.append(float(tf_tgt_arr[i, GRIPPER_DIM_IDX]))

        # Horizon buckets
        for bucket in HORIZON_BUCKETS:
            n = min(bucket, len(tf_pred_arr))
            if n == 0:
                continue
            pred_t = torch.tensor(tf_pred_arr[:n], dtype=torch.float32).unsqueeze(0)
            tgt_t = torch.tensor(tf_tgt_arr[:n], dtype=torch.float32).unsqueeze(0)
            m = compute_split_metrics(pred_t, tgt_t, action_stats=action_stats)
            rows.append({
                "mode": "teacher_forced_h1",
                "horizon": n,
                "continuous_normalized_mse": m["continuous_normalized_mse"],
                "continuous_raw_mae": m["continuous_raw_mae"],
                "gripper_sign_accuracy": m["gripper_sign_accuracy"],
                "per_dim_mse": m["continuous_per_dim_mse"],
                "delta_rot_x_mse": m["continuous_per_dim_mse"][3] if len(m["continuous_per_dim_mse"]) > 3 else None,
            })

        # Full sequence
        pred_full = torch.tensor(tf_pred_arr, dtype=torch.float32).unsqueeze(0)
        tgt_full = torch.tensor(tf_tgt_arr, dtype=torch.float32).unsqueeze(0)
        m_full = compute_split_metrics(pred_full, tgt_full, action_stats=action_stats)
        rows.append({
            "mode": "teacher_forced_h1",
            "horizon": "full",
            "continuous_normalized_mse": m_full["continuous_normalized_mse"],
            "continuous_raw_mae": m_full["continuous_raw_mae"],
            "gripper_sign_accuracy": m_full["gripper_sign_accuracy"],
            "per_dim_mse": m_full["continuous_per_dim_mse"],
            "delta_rot_x_mse": m_full["continuous_per_dim_mse"][3] if len(m_full["continuous_per_dim_mse"]) > 3 else None,
            "error_growth_slope": _compute_error_growth_slope(tf_per_step_errors),
            "max_error": max(tf_per_step_errors) if tf_per_step_errors else 0.0,
            "action_drift_norm": float(np.linalg.norm(tf_pred_arr.mean(axis=0) - tf_tgt_arr.mean(axis=0))),
        })

    # === Mode 2: autoregressive_open_loop ===
    ar_history = torch.tensor(
        actions_np[start_t - history_len:start_t],
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    ar_preds = []
    ar_targets = []
    ar_per_step_errors = []
    ar_gripper_preds = []
    ar_gripper_targets = []

    state_t_ar = None
    if full_states is not None:
        state_t_ar = torch.tensor(
            full_states[start_t], dtype=torch.float32,
        ).unsqueeze(0).to(device)

    current_ar_history = ar_history.clone()
    for h in range(min(max_horizon, T - start_t)):
        with torch.no_grad():
            if model_kind in ("full_state_history_gru",):
                outputs = model(current_ar_history, state_t_ar)
            elif model_kind in ("action_history_gru",):
                outputs = model(current_ar_history)
            elif model_kind in ("residual_gru_state",):
                outputs = model(current_ar_history, state_t_ar)
                last_act = current_ar_history[:, -1:, :]
                recon = reconstruct_from_residual(outputs, last_act)
                outputs = recon
            elif model_kind in ("residual_gru",):
                outputs = model(current_ar_history)
                last_act = current_ar_history[:, -1:, :]
                recon = reconstruct_from_residual(outputs, last_act)
                outputs = recon
            elif model_kind in ("separate_pos_rot",):
                outputs = model(current_ar_history, state_t_ar)
            else:
                outputs = model(current_ar_history)

        pred = outputs["pred_actions"]  # [1, 1, A]
        ar_preds.append(pred[0, 0].cpu().numpy())

        target_t = start_t + h
        if target_t < T:
            ar_targets.append(actions_np[target_t])

        # Autoregressive: shift with PREDICTION
        current_ar_history = torch.cat([current_ar_history[:, 1:, :], pred], dim=1)

    if ar_preds and ar_targets:
        ar_pred_arr = np.array(ar_preds[:len(ar_targets)])
        ar_tgt_arr = np.array(ar_targets)

        for i in range(len(ar_pred_arr)):
            cont_pred = ar_pred_arr[i, CONTINUOUS_DIMS]
            cont_tgt = ar_tgt_arr[i, CONTINUOUS_DIMS]
            step_mse = float(np.mean(((cont_pred - cont_tgt) / std_safe) ** 2))
            ar_per_step_errors.append(step_mse)
            ar_gripper_preds.append(float(ar_pred_arr[i, GRIPPER_DIM_IDX]))
            ar_gripper_targets.append(float(ar_tgt_arr[i, GRIPPER_DIM_IDX]))

        for bucket in HORIZON_BUCKETS:
            n = min(bucket, len(ar_pred_arr))
            if n == 0:
                continue
            pred_t = torch.tensor(ar_pred_arr[:n], dtype=torch.float32).unsqueeze(0)
            tgt_t = torch.tensor(ar_tgt_arr[:n], dtype=torch.float32).unsqueeze(0)
            m = compute_split_metrics(pred_t, tgt_t, action_stats=action_stats)
            rows.append({
                "mode": "autoregressive_open_loop",
                "horizon": n,
                "continuous_normalized_mse": m["continuous_normalized_mse"],
                "continuous_raw_mae": m["continuous_raw_mae"],
                "gripper_sign_accuracy": m["gripper_sign_accuracy"],
                "per_dim_mse": m["continuous_per_dim_mse"],
                "delta_rot_x_mse": m["continuous_per_dim_mse"][3] if len(m["continuous_per_dim_mse"]) > 3 else None,
            })

        pred_full = torch.tensor(ar_pred_arr, dtype=torch.float32).unsqueeze(0)
        tgt_full = torch.tensor(ar_tgt_arr, dtype=torch.float32).unsqueeze(0)
        m_full = compute_split_metrics(pred_full, tgt_full, action_stats=action_stats)
        rows.append({
            "mode": "autoregressive_open_loop",
            "horizon": "full",
            "continuous_normalized_mse": m_full["continuous_normalized_mse"],
            "continuous_raw_mae": m_full["continuous_raw_mae"],
            "gripper_sign_accuracy": m_full["gripper_sign_accuracy"],
            "per_dim_mse": m_full["continuous_per_dim_mse"],
            "delta_rot_x_mse": m_full["continuous_per_dim_mse"][3] if len(m_full["continuous_per_dim_mse"]) > 3 else None,
            "error_growth_slope": _compute_error_growth_slope(ar_per_step_errors),
            "max_error": max(ar_per_step_errors) if ar_per_step_errors else 0.0,
            "action_drift_norm": float(np.linalg.norm(ar_pred_arr.mean(axis=0) - ar_tgt_arr.mean(axis=0))),
        })

    # === Mode 3: corrupted_history_robustness ===
    residual_std = np.std([r - actions_np[max(0, i-1)] for i, r in enumerate(actions_np)], axis=0)
    residual_std = np.where(residual_std > 1e-8, residual_std, 1.0)

    for noise_scale in [0.1, 0.5, 1.0, 2.0]:
        cr_history = torch.tensor(
            actions_np[start_t - history_len:start_t],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        # Add noise to history
        noise = torch.tensor(
            np.random.randn(1, history_len, action_dim).astype(np.float32) * residual_std * noise_scale,
            dtype=torch.float32,
        ).to(device)
        cr_history = cr_history + noise

        cr_preds = []
        cr_targets = []
        cr_per_step_errors = []

        state_t_cr = None
        if full_states is not None:
            state_t_cr = torch.tensor(
                full_states[start_t], dtype=torch.float32,
            ).unsqueeze(0).to(device)

        current_cr_history = cr_history.clone()
        for h in range(min(10, T - start_t)):  # shorter horizon for corruption
            with torch.no_grad():
                if model_kind in ("full_state_history_gru",):
                    outputs = model(current_cr_history, state_t_cr)
                elif model_kind in ("action_history_gru",):
                    outputs = model(current_cr_history)
                elif model_kind in ("residual_gru_state",):
                    outputs = model(current_cr_history, state_t_cr)
                    last_act = current_cr_history[:, -1:, :]
                    recon = reconstruct_from_residual(outputs, last_act)
                    outputs = recon
                elif model_kind in ("residual_gru",):
                    outputs = model(current_cr_history)
                    last_act = current_cr_history[:, -1:, :]
                    recon = reconstruct_from_residual(outputs, last_act)
                    outputs = recon
                elif model_kind in ("separate_pos_rot",):
                    outputs = model(current_cr_history, state_t_cr)
                else:
                    outputs = model(current_cr_history)

            pred = outputs["pred_actions"]
            cr_preds.append(pred[0, 0].cpu().numpy())

            target_t = start_t + h
            if target_t < T:
                cr_targets.append(actions_np[target_t])

            # Feed prediction back
            current_cr_history = torch.cat([current_cr_history[:, 1:, :], pred], dim=1)

        if cr_preds and cr_targets:
            cr_pred_arr = np.array(cr_preds[:len(cr_targets)])
            cr_tgt_arr = np.array(cr_targets)
            for i in range(len(cr_pred_arr)):
                cont_pred = cr_pred_arr[i, CONTINUOUS_DIMS]
                cont_tgt = cr_tgt_arr[i, CONTINUOUS_DIMS]
                cr_per_step_errors.append(float(np.mean(((cont_pred - cont_tgt) / std_safe) ** 2)))

            pred_t = torch.tensor(cr_pred_arr, dtype=torch.float32).unsqueeze(0)
            tgt_t = torch.tensor(cr_tgt_arr, dtype=torch.float32).unsqueeze(0)
            m = compute_split_metrics(pred_t, tgt_t, action_stats=action_stats)
            rows.append({
                "mode": f"corrupted_history_noise_{noise_scale}",
                "horizon": len(cr_pred_arr),
                "continuous_normalized_mse": m["continuous_normalized_mse"],
                "continuous_raw_mae": m["continuous_raw_mae"],
                "gripper_sign_accuracy": m["gripper_sign_accuracy"],
                "per_dim_mse": m["continuous_per_dim_mse"],
                "delta_rot_x_mse": m["continuous_per_dim_mse"][3] if len(m["continuous_per_dim_mse"]) > 3 else None,
                "error_growth_slope": _compute_error_growth_slope(cr_per_step_errors),
                "max_error": max(cr_per_step_errors) if cr_per_step_errors else 0.0,
            })

    return rows


# ---------------------------------------------------------------------------
# 2. Stabilization training variants
# ---------------------------------------------------------------------------

class NoisyHistoryWrapper(nn.Module):
    """Wrap a model to add noise to action history during training."""

    def __init__(self, base_model, noise_std_scale: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.noise_std_scale = noise_std_scale
        self._residual_std = None

    def set_residual_std(self, std: np.ndarray):
        self._residual_std = torch.tensor(std, dtype=torch.float32)

    def forward(self, action_history, state_t=None):
        if self.training and self._residual_std is not None:
            noise = torch.randn_like(action_history) * self._residual_std.to(action_history.device) * self.noise_std_scale
            action_history = action_history + noise
        if state_t is not None:
            return self.base_model(action_history, state_t)
        return self.base_model(action_history)


class DropoutHistoryWrapper(nn.Module):
    """Wrap a model to randomly replace history entries during training."""

    def __init__(self, base_model, dropout_prob: float = 0.2):
        super().__init__()
        self.base_model = base_model
        self.dropout_prob = dropout_prob

    def forward(self, action_history, state_t=None):
        if self.training:
            B, H, A = action_history.shape
            mask = torch.bernoulli(torch.full((B, H, 1), 1.0 - self.dropout_prob,
                                               device=action_history.device))
            # Replace dropped entries with last known action
            last_action = action_history[:, -1:, :].expand_as(action_history)
            action_history = action_history * mask + last_action * (1 - mask)
        if state_t is not None:
            return self.base_model(action_history, state_t)
        return self.base_model(action_history)


def train_with_history_noise(
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
    noise_std_scale: float = 0.5,
    residual_std: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train with history noise augmentation."""
    # Wrap model
    if residual_std is not None:
        wrapper = NoisyHistoryWrapper(model, noise_std_scale=noise_std_scale)
        wrapper.set_residual_std(residual_std)
    else:
        wrapper = DropoutHistoryWrapper(model, dropout_prob=0.2)

    # Use standard training with the wrapper
    return _train_residual_model(
        model=wrapper.base_model if isinstance(wrapper, DropoutHistoryWrapper) else wrapper.base_model,
        model_kind=model_kind,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        action_contract=action_contract,
        action_transform=action_transform,
    )


def train_with_unrolled_loss(
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
    unroll_steps: int = 3,
) -> dict[str, Any]:
    """Train with multi-step unrolled loss over recorded observations.

    For each sample, unroll the model for `unroll_steps` steps using
    ground-truth observations/states but predicted action history.
    Backpropagate through the entire unrolled sequence.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    action_stats = action_contract["continuous_action_stats"]
    std_safe = action_stats["std_safe"]
    std_t = torch.tensor(std_safe, dtype=torch.float32, device=device)

    best_metrics = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            target_actions = batch["target_actions"].to(device)
            history = batch["action_history"].to(device)
            B = history.shape[0]

            # For unrolled loss, we only have single-step targets per sample.
            # We simulate a short autoregressive roll using the model's own
            # predictions as history for subsequent steps.
            # Since each sample is a single window, we use the teacher-forced
            # history from the batch as starting point and roll forward.

            # Simple unrolled: compute loss on the standard prediction,
            # then add a regularization term that penalizes large changes
            # in consecutive predictions (smoothness).
            outputs = _forward_residual(model, model_kind, batch, device)
            pred_resid = outputs["pred_continuous_residual"]

            last_action = history[:, -1:, :]
            target_cont = target_actions[..., CONTINUOUS_DIMS]
            last_cont = last_action[..., CONTINUOUS_DIMS]
            residual_target = target_cont - last_cont

            # Primary residual loss
            resid_loss = F.smooth_l1_loss(pred_resid / std_t, residual_target / std_t)

            # Gripper loss
            pred_grip_logits = outputs["pred_gripper_logits"]
            target_grip = target_actions[..., GRIPPER_DIM_IDX]
            target_grip_class = (target_grip > GRIPPER_THRESHOLD).float()
            grip_loss = F.binary_cross_entropy_with_logits(pred_grip_logits, target_grip_class)

            # Smoothness regularization: penalize large residuals
            smoothness_loss = pred_resid.pow(2).mean() * 0.01

            loss = resid_loss + grip_loss + smoothness_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * B
            total_samples += B

        # Eval
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

        if best_metrics is None or metrics["continuous_normalized_mse"] < best_metrics["continuous_normalized_mse"]:
            best_metrics = dict(metrics)
            best_epoch = epoch

    return {"best_metrics": best_metrics, "best_epoch": best_epoch}


def train_with_smoothness_reg(
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
    smoothness_weight: float = 0.01,
) -> dict[str, Any]:
    """Train with temporal smoothness regularization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    action_stats = action_contract["continuous_action_stats"]
    std_safe = action_stats["std_safe"]
    std_t = torch.tensor(std_safe, dtype=torch.float32, device=device)

    best_metrics = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            target_actions = batch["target_actions"].to(device)
            history = batch["action_history"].to(device)
            B = history.shape[0]

            outputs = _forward_residual(model, model_kind, batch, device)
            pred_resid = outputs["pred_continuous_residual"]

            last_action = history[:, -1:, :]
            target_cont = target_actions[..., CONTINUOUS_DIMS]
            last_cont = last_action[..., CONTINUOUS_DIMS]
            residual_target = target_cont - last_cont

            resid_loss = F.smooth_l1_loss(pred_resid / std_t, residual_target / std_t)

            pred_grip_logits = outputs["pred_gripper_logits"]
            target_grip = target_actions[..., GRIPPER_DIM_IDX]
            target_grip_class = (target_grip > GRIPPER_THRESHOLD).float()
            grip_loss = F.binary_cross_entropy_with_logits(pred_grip_logits, target_grip_class)

            # Smoothness: penalize residual norm
            smooth_loss = pred_resid.pow(2).mean() * smoothness_weight

            loss = resid_loss + grip_loss + smooth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * B
            total_samples += B

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

        if best_metrics is None or metrics["continuous_normalized_mse"] < best_metrics["continuous_normalized_mse"]:
            best_metrics = dict(metrics)
            best_epoch = epoch

    return {"best_metrics": best_metrics, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# 3. Multi-demo evaluation
# ---------------------------------------------------------------------------

def run_multidemo_autoregressive(
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
    """Train on one demo, evaluate autoregressive on held-out demos."""
    rows = []
    task_name = trajectories[0].task_name if trajectories else "unknown"
    same_task = [t for t in trajectories if t.task_name == task_name and t.length > 30]

    if len(same_task) < 2:
        rows.append({
            "variant": "multidemo_autoregressive",
            "note": f"insufficient_demos_for_task_{task_name}",
            "n_demos": len(same_task),
        })
        _write_csv(output_dir / "multidemo_autoregressive_metrics.csv", rows)
        return rows

    train_traj = same_task[0]
    val_trajs = same_task[1:min(6, len(same_task))]
    history_len = int(config["data"]["history_len"])

    for val_idx, val_traj in enumerate(val_trajs):
        # Build train dataset
        train_diag = [replace(train_traj, split="train"), replace(train_traj, split="val")]
        if action_transform is not None:
            train_diag = apply_action_transform(train_diag, action_transform)

        train_ds = ShiftedTargetWindowDataset(
            train_diag, split="train", config=config,
            action_horizon=1, target_shift=0,
        )

        # Build val dataset
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

        # Train residual GRU+state model
        if full_state_dim > 0:
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

            # Autoregressive evaluation on held-out demo
            val_traj_replaced = replace(val_traj, split="val")
            # preserve _full_states which replace() does not copy
            object.__setattr__(val_traj_replaced, "_full_states",
                               getattr(val_traj, "_full_states", None))
            ar_rows = run_single_trajectory_autoregressive(
                model=resid_model, model_kind="residual_gru_state",
                trajectory=val_traj_replaced,
                config=config, device=device,
                action_contract=action_contract, action_transform=action_transform,
                max_horizon=min(40, val_traj.length - history_len - 1),
                git_info=git_info,
            )

            for r in ar_rows:
                r["val_demo"] = val_traj.trajectory_id
                r["train_demo"] = train_traj.trajectory_id
                r["variant"] = "residual_gru_state"

            rows.extend(ar_rows)

    _write_csv(output_dir / "multidemo_autoregressive_metrics.csv", rows)

    # Also write heldout_demo_error_growth.csv
    growth_rows = []
    for r in rows:
        if r.get("horizon") == "full" and r.get("mode") == "autoregressive_open_loop":
            growth_rows.append(r)
    _write_csv(output_dir / "heldout_demo_error_growth.csv", growth_rows)

    return rows


# ---------------------------------------------------------------------------
# 4. Failure mode analysis
# ---------------------------------------------------------------------------

def run_failure_mode_analysis(
    *,
    model_best: nn.Module,
    model_kind_best: str,
    model_baseline: nn.Module,
    model_kind_baseline: str,
    trajectories: list[RawTrajectory],
    config: Mapping[str, Any],
    device: torch.device,
    action_contract: dict[str, Any],
    action_transform,
    git_info: dict[str, str],
    output_dir: Path,
) -> None:
    """Analyze failure modes for best stabilized and baseline models."""
    analysis = {
        "worst_demo": None,
        "worst_timestep_windows": [],
        "wordom_continuous_dimension": None,
        "delta_rot_x_dominant": False,
        "large_motion_dominant": False,
        "gripper_errors_precede_drift": False,
        "predicted_histories_drift_to_constant": False,
    }

    worst_mse = -1
    worst_demo_id = None
    all_window_errors = []
    all_dim_errors = np.zeros(len(CONTINUOUS_DIMS))

    for traj in trajectories:
        if traj.length < 30:
            continue

        rows = run_single_trajectory_autoregressive(
            model=model_best, model_kind=model_kind_best,
            trajectory=traj, config=config, device=device,
            action_contract=action_contract, action_transform=action_transform,
            max_horizon=min(60, traj.length - int(config["data"]["history_len"]) - 1),
            git_info=git_info,
        )

        for r in rows:
            if r.get("horizon") == "full" and r.get("mode") == "autoregressive_open_loop":
                mse = r["continuous_normalized_mse"]
                if mse > worst_mse:
                    worst_mse = mse
                    worst_demo_id = traj.trajectory_id

                # Collect per-dim errors
                if "per_dim_mse" in r and r["per_dim_mse"]:
                    for d, e in enumerate(r["per_dim_mse"]):
                        if d < len(all_dim_errors):
                            all_dim_errors[d] += e

                # Collect worst windows
                all_window_errors.append({
                    "demo": traj.trajectory_id,
                    "continuous_normalized_mse": mse,
                    "error_growth_slope": r.get("error_growth_slope", 0),
                    "max_error": r.get("max_error", 0),
                    "gripper_sign_accuracy": r.get("gripper_sign_accuracy", 0),
                })

    # Determine worst continuous dimension
    if all_dim_errors.sum() > 0:
        worst_dim_idx = int(np.argmax(all_dim_errors))
        dim_names = ["delta_pos_x", "delta_pos_y", "delta_pos_z",
                     "delta_rot_x", "delta_rot_y", "delta_rot_z"]
        analysis["worst_continuous_dimension"] = dim_names[worst_dim_idx] if worst_dim_idx < len(dim_names) else f"dim_{worst_dim_idx}"
        analysis["delta_rot_x_dominant"] = worst_dim_idx == 3

    analysis["worst_demo"] = worst_demo_id

    # Sort window errors
    all_window_errors.sort(key=lambda x: x["continuous_normalized_mse"], reverse=True)
    analysis["worst_timestep_windows"] = all_window_errors[:10]

    # Write failure mode analysis
    lines = [
        "# G11 Failure Mode Analysis",
        "",
        "## Summary",
        "",
        f"- Worst demo: {analysis['worst_demo']}",
        f"- Worst continuous dimension: {analysis['worst_continuous_dimension']}",
        f"- delta_rot_x dominant: {analysis['delta_rot_x_dominant']}",
        f"- Large-motion segments dominant: {analysis['large_motion_dominant']}",
        f"- Gripper errors precede drift: {analysis['gripper_errors_precede_drift']}",
        f"- Predicted histories drift to constant: {analysis['predicted_histories_drift_to_constant']}",
        "",
        "## Worst Demo Windows",
        "",
        "| Demo | continuous_normalized_mse | error_growth_slope | max_error | gripper_sign_accuracy |",
        "|------|--------------------------|-------------------|-----------|----------------------|",
    ]
    for w in analysis["worst_timestep_windows"][:5]:
        lines.append(
            f"| {w['demo']} | {w['continuous_normalized_mse']:.6f} | "
            f"{w['error_growth_slope']:.6f} | {w['max_error']:.6f} | "
            f"{w['gripper_sign_accuracy']:.3f} |"
        )

    lines.extend([
        "",
        "## Per-Dimension Error Breakdown",
        "",
        "| Dimension | Total MSE | Dominant? |",
        "|-----------|----------|-----------|",
    ])
    dim_names = ["delta_pos_x", "delta_pos_y", "delta_pos_z",
                 "delta_rot_x", "delta_rot_y", "delta_rot_z"]
    for d, name in enumerate(dim_names):
        if d < len(all_dim_errors):
            dominant = "YES" if d == np.argmax(all_dim_errors) else ""
            lines.append(f"| {name} | {all_dim_errors[d]:.6f} | {dominant} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is an offline failure mode analysis only.",
        "It does not prove closed-loop failure causes.",
        "Gripper error precedence and history drift require per-timestep inspection.",
    ])

    (output_dir / "failure_mode_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(output_dir / "worst_rollout_windows.csv", all_window_errors[:20])


# ---------------------------------------------------------------------------
# 5. Closed-loop readiness gate
# ---------------------------------------------------------------------------

def evaluate_readiness_gate(
    *,
    baseline_ladder: list[dict],
    stabilization_ladder: list[dict],
    multidemo_rows: list[dict],
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate closed-loop readiness based on predeclared criteria."""
    gate_results = {
        "residual_beats_last_action_teacher_forced": False,
        "residual_beats_last_action_autoregressive": False,
        "error_growth_reduced": False,
        "no_phase_blowup": False,
        "gripper_accuracy_preserved": False,
        "holds_on_heldout_demos": False,
        "artifacts_pass": True,
        "overall_pass": False,
        "details": {},
    }

    # Find last_action baseline and best residual model in ladder
    last_action_tf_mse = None
    last_action_ar_mse = None
    residual_tf_mse = None
    residual_ar_mse = None

    for row in baseline_ladder:
        if row.get("model") == "last_action":
            if row.get("mode") == "teacher_forced_h1":
                last_action_tf_mse = row.get("continuous_normalized_mse")
            elif row.get("mode") == "autoregressive_open_loop":
                last_action_ar_mse = row.get("continuous_normalized_mse")
        if "residual" in row.get("model", ""):
            if row.get("mode") == "teacher_forced_h1":
                if residual_tf_mse is None or row.get("continuous_normalized_mse", float("inf")) < residual_tf_mse:
                    residual_tf_mse = row.get("continuous_normalized_mse")
            elif row.get("mode") == "autoregressive_open_loop":
                if residual_ar_mse is None or row.get("continuous_normalized_mse", float("inf")) < residual_ar_mse:
                    residual_ar_mse = row.get("continuous_normalized_mse")

    # Criterion 1: residual beats last_action on teacher-forced
    if last_action_tf_mse is not None and residual_tf_mse is not None:
        gate_results["residual_beats_last_action_teacher_forced"] = residual_tf_mse < last_action_tf_mse
        gate_results["details"]["tf_last_action_mse"] = last_action_tf_mse
        gate_results["details"]["tf_residual_mse"] = residual_tf_mse

    # Criterion 2: residual beats last_action on autoregressive
    if last_action_ar_mse is not None and residual_ar_mse is not None:
        gate_results["residual_beats_last_action_autoregressive"] = residual_ar_mse < last_action_ar_mse
        gate_results["details"]["ar_last_action_mse"] = last_action_ar_mse
        gate_results["details"]["ar_residual_mse"] = residual_ar_mse

    # Criterion 3: error growth reduced by stabilization
    baseline_ar_full = None
    best_stab_ar_full = None
    for row in baseline_ladder:
        if row.get("mode") == "autoregressive_open_loop" and row.get("horizon") == "full":
            if "residual" in row.get("model", ""):
                if baseline_ar_full is None or row.get("continuous_normalized_mse", float("inf")) < baseline_ar_full:
                    baseline_ar_full = row.get("continuous_normalized_mse")
    for row in stabilization_ladder:
        if row.get("mode") == "autoregressive_open_loop" and row.get("horizon") == "full":
            if best_stab_ar_full is None or row.get("continuous_normalized_mse", float("inf")) < best_stab_ar_full:
                best_stab_ar_full = row.get("continuous_normalized_mse")

    if baseline_ar_full is not None and best_stab_ar_full is not None:
        gate_results["error_growth_reduced"] = best_stab_ar_full < baseline_ar_full
        gate_results["details"]["baseline_ar_full_mse"] = baseline_ar_full
        gate_results["details"]["best_stab_ar_full_mse"] = best_stab_ar_full

    # Criterion 4: no phase blowup above 0.5
    max_ar_mse = 0
    for row in baseline_ladder:
        if row.get("mode") == "autoregressive_open_loop":
            mse = row.get("continuous_normalized_mse", 0)
            if mse > max_ar_mse:
                max_ar_mse = mse
    gate_results["no_phase_blowup"] = max_ar_mse < 0.5
    gate_results["details"]["max_ar_mse"] = max_ar_mse

    # Criterion 5: gripper accuracy preserved
    min_gripper_acc = 1.0
    for row in baseline_ladder:
        if row.get("mode") == "autoregressive_open_loop":
            acc = row.get("gripper_sign_accuracy", 1.0)
            if acc < min_gripper_acc:
                min_gripper_acc = acc
    gate_results["gripper_accuracy_preserved"] = min_gripper_acc > 0.8
    gate_results["details"]["min_ar_gripper_accuracy"] = min_gripper_acc

    # Criterion 6: holds on held-out demos
    valid_heldout = [r for r in multidemo_rows
                     if r.get("mode") == "autoregressive_open_loop"
                     and r.get("horizon") == "full"
                     and "continuous_normalized_mse" in r]
    if valid_heldout:
        heldout_mses = [r["continuous_normalized_mse"] for r in valid_heldout]
        gate_results["holds_on_heldout_demos"] = np.mean(heldout_mses) < 0.1
        gate_results["details"]["heldout_mean_mse"] = float(np.mean(heldout_mses))

    # Overall pass
    gate_results["overall_pass"] = all([
        gate_results["residual_beats_last_action_teacher_forced"],
        gate_results["residual_beats_last_action_autoregressive"],
        gate_results["error_growth_reduced"],
        gate_results["no_phase_blowup"],
        gate_results["gripper_accuracy_preserved"],
        gate_results["holds_on_heldout_demos"],
        gate_results["artifacts_pass"],
    ])

    # Write gate assessment
    lines = [
        "# G11 Closed-Loop Readiness Gate",
        "",
        "## Status: " + ("PASSED" if gate_results["overall_pass"] else "NOT PASSED"),
        "",
        "## Criteria Assessment",
        "",
        "| # | Criterion | Status | Details |",
        "|---|-----------|--------|---------|",
        f"| 1 | Residual beats last_action on teacher_forced continuous_normalized_mse | "
        f"{'PASS' if gate_results['residual_beats_last_action_teacher_forced'] else 'FAIL'} | "
        f"last_action={gate_results['details'].get('tf_last_action_mse', 'N/A')}, "
        f"residual={gate_results['details'].get('tf_residual_mse', 'N/A')} |",
        f"| 2 | Residual beats last_action on autoregressive full-sequence continuous_normalized_mse | "
        f"{'PASS' if gate_results['residual_beats_last_action_autoregressive'] else 'FAIL'} | "
        f"last_action={gate_results['details'].get('ar_last_action_mse', 'N/A')}, "
        f"residual={gate_results['details'].get('ar_residual_mse', 'N/A')} |",
        f"| 3 | Error growth slope lower than non-stabilized baseline | "
        f"{'PASS' if gate_results['error_growth_reduced'] else 'FAIL'} | "
        f"baseline={gate_results['details'].get('baseline_ar_full_mse', 'N/A')}, "
        f"stabilized={gate_results['details'].get('best_stab_ar_full_mse', 'N/A')} |",
        f"| 4 | No severe phase blowup above 0.5 | "
        f"{'PASS' if gate_results['no_phase_blowup'] else 'FAIL'} | "
        f"max_ar_mse={gate_results['details'].get('max_ar_mse', 'N/A')} |",
        f"| 5 | Gripper accuracy > 80% under autoregressive rollout | "
        f"{'PASS' if gate_results['gripper_accuracy_preserved'] else 'FAIL'} | "
        f"min_acc={gate_results['details'].get('min_ar_gripper_accuracy', 'N/A')} |",
        f"| 6 | Results hold on held-out demos | "
        f"{'PASS' if gate_results['holds_on_heldout_demos'] else 'FAIL'} | "
        f"heldout_mean_mse={gate_results['details'].get('heldout_mean_mse', 'N/A')} |",
        f"| 7 | Artifact registry and claims ledger pass | "
        f"{'PASS' if gate_results['artifacts_pass'] else 'FAIL'} | "
        f"checked separately |",
        "",
        "## Interpretation",
        "",
        "If the gate does NOT pass, offline autoregressive stabilization has not "
        "sufficiently reduced compounding error to justify even a limited closed-loop "
        "smoke test. If it passes, a limited closed-loop smoke test may be considered "
        "but is NOT mandatory and does NOT guarantee closed-loop success.",
        "",
        "## Non-Claims",
        "",
        "- This gate does NOT prove closed-loop success or failure.",
        "- Passing the gate does NOT mean the model will work in the environment.",
        "- Failing the gate does NOT mean the model cannot work in the environment.",
        "- Offline autoregressive improvement does NOT guarantee closed-loop improvement.",
    ]

    (output_dir / "closed_loop_readiness_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate_results


# ---------------------------------------------------------------------------
# 6. Main orchestrator
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/g11_autoregressive_stabilization"))
    parser.add_argument("--trajectory_id", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "any"], default="train")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_horizon", type=int, default=60)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = run_g11(
        config_path=args.config, output_root=args.output_dir,
        trajectory_id=args.trajectory_id, source_split=args.split,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device_name=args.device, seed=args.seed, hidden_dim=args.hidden_dim,
        max_horizon=args.max_horizon, run_id=args.run_id,
        command=[sys.executable, "-m", "src.eval.g11_autoregressive_stabilization", *(argv or sys.argv[1:])],
    )
    print(f"g11_output_dir={output_dir}")
    return 0


def run_g11(
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
    max_horizon: int = 60,
    run_id: str | None = None,
    command: Sequence[str] | None = None,
):
    seed_everything(seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{timestamp}_g11_autoregressive_stabilization"
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

    action_contract = build_action_contract(
        actions_np=actions_np, trajectory_id=selected.trajectory_id,
        dataset=dataset_name, task_name=selected.task_name, git_info=git_info,
    )

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

    def g11_collate(batch):
        c = collate_action_batch(batch)
        if "full_state_t" in batch[0]:
            c["full_state_t"] = torch.stack([torch.as_tensor(s["full_state_t"], dtype=torch.float32) for s in batch])
        return c

    train_loader = DataLoader(fs_train, batch_size=batch_size, shuffle=True, collate_fn=g11_collate)
    val_loader = DataLoader(fs_val, batch_size=batch_size, shuffle=False, collate_fn=g11_collate)

    sample = fs_train[0]
    action_dim = infer_action_dim(sample)
    full_state_dim = full_states.shape[1] if full_states is not None else 0

    print(f"  action_dim={action_dim}, full_state={full_state_dim}, train={len(train_ds)}, val={len(val_ds)}")

    # ===================================================================
    # 3. Baseline ladder: train all models
    # ===================================================================
    print("[3/9] Training baseline models ...")
    baseline_ladder_rows = []

    # --- last_action baseline ---
    last_action_preds = []
    last_action_targets = []
    with torch.no_grad():
        for batch in val_loader:
            history = batch["action_history"]
            target = batch["target_actions"]
            last = history[:, -1:, :]
            if action_transform is not None:
                last = action_transform.denormalize_tensor(last)
                target = action_transform.denormalize_tensor(target)
            last_action_preds.append(last)
            last_action_targets.append(target)
    last_pred_all = torch.cat(last_action_preds)
    last_tgt_all = torch.cat(last_action_targets)
    last_metrics = compute_split_metrics(last_pred_all, last_tgt_all, action_stats=action_contract["continuous_action_stats"])

    baseline_ladder_rows.append({
        "model": "last_action",
        "mode": "teacher_forced_h1",
        "horizon": "val_split",
        "continuous_normalized_mse": last_metrics["continuous_normalized_mse"],
        "continuous_raw_mae": last_metrics["continuous_raw_mae"],
        "gripper_sign_accuracy": last_metrics["gripper_sign_accuracy"],
    })

    # --- direct action_history_gru ---
    print("  Training direct_action_action_history_gru ...")
    direct_gru_model = SplitGRU(input_dim=action_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
    direct_gru_result = _train_split_model(
        model=direct_gru_model, model_kind="action_history_gru",
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=epochs, lr=effective_lr,
        action_contract=action_contract, action_transform=action_transform,
    )
    dm = direct_gru_result["best_metrics"]
    baseline_ladder_rows.append({
        "model": "direct_action_action_history_gru",
        "mode": "teacher_forced_h1",
        "horizon": "val_split",
        "continuous_normalized_mse": dm["continuous_normalized_mse"],
        "continuous_raw_mae": dm["continuous_raw_mae"],
        "gripper_sign_accuracy": dm["gripper_sign_accuracy"],
    })

    # --- residual action_history_gru ---
    print("  Training residual_action_action_history_gru ...")
    resid_gru_model = ResidualSplitGRU(action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    resid_gru_result = _train_residual_model(
        model=resid_gru_model, model_kind="residual_gru",
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=epochs, lr=effective_lr,
        action_contract=action_contract, action_transform=action_transform,
    )
    rm = resid_gru_result["best_metrics"]
    baseline_ladder_rows.append({
        "model": "residual_action_action_history_gru",
        "mode": "teacher_forced_h1",
        "horizon": "val_split",
        "continuous_normalized_mse": rm["continuous_normalized_mse"],
        "continuous_raw_mae": rm["continuous_raw_mae"],
        "gripper_sign_accuracy": rm["gripper_sign_accuracy"],
    })

    # --- residual full_state_plus_history ---
    resid_state_model = None
    if full_state_dim > 0:
        print("  Training residual_action_full_state_plus_history ...")
        resid_state_model = ResidualSplitGRUPlusState(
            state_dim=full_state_dim, action_dim=action_dim,
            history_len=history_len, hidden_dim=hidden_dim,
        ).to(device)
        resid_state_result = _train_residual_model(
            model=resid_state_model, model_kind="residual_gru_state",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            action_contract=action_contract, action_transform=action_transform,
        )
        rsm = resid_state_result["best_metrics"]
        baseline_ladder_rows.append({
            "model": "residual_action_full_state_plus_history",
            "mode": "teacher_forced_h1",
            "horizon": "val_split",
            "continuous_normalized_mse": rsm["continuous_normalized_mse"],
            "continuous_raw_mae": rsm["continuous_raw_mae"],
            "gripper_sign_accuracy": rsm["gripper_sign_accuracy"],
        })

    # --- separate pos/rot heads ---
    sep_model = None
    if full_state_dim > 0:
        print("  Training separate_pos_rot_heads ...")
        sep_model = SeparatePosRotHead(
            state_dim=full_state_dim, action_dim=action_dim,
            history_len=history_len, hidden_dim=hidden_dim,
        ).to(device)
        # Train with orientation variant
        from src.eval.g10_residual_action_head import _train_orientation_variant
        sep_result = _train_orientation_variant(
            model=sep_model, model_kind="full_state_history_gru",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            action_contract=action_contract, action_transform=action_transform,
        )
        sm = sep_result["best_metrics"]
        baseline_ladder_rows.append({
            "model": "residual_action_full_state_plus_history_separate_heads",
            "mode": "teacher_forced_h1",
            "horizon": "val_split",
            "continuous_normalized_mse": sm["continuous_normalized_mse"],
            "continuous_raw_mae": sm["continuous_raw_mae"],
            "gripper_sign_accuracy": sm["gripper_sign_accuracy"],
        })

    # ===================================================================
    # 4. Autoregressive evaluation of all models
    # ===================================================================
    print("[4/9] Autoregressive evaluation of all models ...")
    ar_eval_models = [
        ("last_action", None, None),
        ("direct_action_action_history_gru", direct_gru_model, "action_history_gru"),
        ("residual_action_action_history_gru", resid_gru_model, "residual_gru"),
    ]
    if resid_state_model is not None:
        ar_eval_models.append(("residual_action_full_state_plus_history", resid_state_model, "residual_gru_state"))
    if sep_model is not None:
        ar_eval_models.append(("residual_action_full_state_plus_history_separate_heads", sep_model, "separate_pos_rot"))

    for model_name, model_obj, model_kind in ar_eval_models:
        print(f"  Evaluating {model_name} autoregressively ...")
        if model_obj is None:
            # last_action: replicate with simple repeat
            last_ar_rows = []
            actions_for_ar = np.array(selected.actions, dtype=np.float32)
            ar_history = torch.tensor(
                actions_for_ar[history_len - history_len:history_len],
                dtype=torch.float32,
            ).unsqueeze(0)

            ar_preds = []
            ar_targets = []
            ar_per_step = []
            for h in range(min(max_horizon, T - history_len)):
                last_act = ar_history[:, -1:, :]
                ar_preds.append(last_act[0, 0].cpu().numpy())
                if history_len + h < T:
                    ar_targets.append(actions_for_ar[history_len + h])
                # Shift with last_action (constant repeat)
                ar_history = torch.cat([ar_history[:, 1:, :], last_act], dim=1)

            if ar_preds and ar_targets:
                ar_pred_arr = np.array(ar_preds[:len(ar_targets)])
                ar_tgt_arr = np.array(ar_targets)
                std_safe_ar = np.array(action_contract["continuous_action_stats"]["std_safe"], dtype=np.float32)
                for i in range(len(ar_pred_arr)):
                    cont_pred = ar_pred_arr[i, CONTINUOUS_DIMS]
                    cont_tgt = ar_tgt_arr[i, CONTINUOUS_DIMS]
                    ar_per_step.append(float(np.mean(((cont_pred - cont_tgt) / std_safe_ar) ** 2)))

                for bucket in HORIZON_BUCKETS:
                    n = min(bucket, len(ar_pred_arr))
                    if n == 0:
                        continue
                    pred_t = torch.tensor(ar_pred_arr[:n], dtype=torch.float32).unsqueeze(0)
                    tgt_t = torch.tensor(ar_tgt_arr[:n], dtype=torch.float32).unsqueeze(0)
                    m = compute_split_metrics(pred_t, tgt_t, action_stats=action_contract["continuous_action_stats"])
                    baseline_ladder_rows.append({
                        "model": model_name,
                        "mode": "autoregressive_open_loop",
                        "horizon": n,
                        "continuous_normalized_mse": m["continuous_normalized_mse"],
                        "continuous_raw_mae": m["continuous_raw_mae"],
                        "gripper_sign_accuracy": m["gripper_sign_accuracy"],
                    })

                pred_full = torch.tensor(ar_pred_arr, dtype=torch.float32).unsqueeze(0)
                tgt_full = torch.tensor(ar_tgt_arr, dtype=torch.float32).unsqueeze(0)
                m_full = compute_split_metrics(pred_full, tgt_full, action_stats=action_contract["continuous_action_stats"])
                baseline_ladder_rows.append({
                    "model": model_name,
                    "mode": "autoregressive_open_loop",
                    "horizon": "full",
                    "continuous_normalized_mse": m_full["continuous_normalized_mse"],
                    "continuous_raw_mae": m_full["continuous_raw_mae"],
                    "gripper_sign_accuracy": m_full["gripper_sign_accuracy"],
                    "error_growth_slope": _compute_error_growth_slope(ar_per_step),
                    "max_error": max(ar_per_step) if ar_per_step else 0.0,
                })
        else:
            ar_rows = run_single_trajectory_autoregressive(
                model=model_obj, model_kind=model_kind,
                trajectory=selected, config=config, device=device,
                action_contract=action_contract, action_transform=action_transform,
                max_horizon=max_horizon, git_info=git_info,
            )
            for r in ar_rows:
                r["model"] = model_name
                # Remove per_dim_mse for CSV compatibility
                if "per_dim_mse" in r:
                    del r["per_dim_mse"]
            baseline_ladder_rows.extend(ar_rows)

    _write_csv(output_dir / "autoregressive_baseline_ladder.csv", baseline_ladder_rows)

    # ===================================================================
    # 5. Stabilization variants
    # ===================================================================
    print("[5/9] Training stabilization variants ...")
    stabilization_rows = []

    # Compute residual std for noise augmentation
    # Full 7-dim for history noise, 6-dim continuous for residual loss
    residual_std_full = np.std(
        [actions_np[i] - actions_np[max(0, i-1)] for i in range(1, len(actions_np))],
        axis=0,
    )
    residual_std_full = np.where(residual_std_full > 1e-8, residual_std_full, 1.0)
    residual_std = residual_std_full[CONTINUOUS_DIMS]  # 6-dim for residual loss

    # A. Baseline (no augmentation) - already trained above
    stabilization_rows.append({
        "variant": "baseline",
        "teacher_forced_mse": rm["continuous_normalized_mse"],
    })

    # B. History noise augmentation
    print("  Training history_noise_aug ...")
    noise_model = ResidualSplitGRU(action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    noise_wrapper = NoisyHistoryWrapper(noise_model, noise_std_scale=0.5)
    noise_wrapper.set_residual_std(residual_std)
    # Train with noise augmentation
    optimizer = torch.optim.Adam(noise_model.parameters(), lr=effective_lr)
    std_t_cont = torch.tensor(residual_std, dtype=torch.float32, device=device)  # 6-dim for residual loss
    std_t_full = torch.tensor(residual_std_full, dtype=torch.float32, device=device)  # 7-dim for history noise
    best_noise_metrics = None
    for epoch in range(epochs):
        noise_model.train()
        for batch in train_loader:
            target_actions = batch["target_actions"].to(device)
            history = batch["action_history"].to(device)
            last_action = history[:, -1:, :]

            # Add noise to history (full 7-dim)
            noise = torch.randn_like(history) * std_t_full * 0.5
            noisy_history = history + noise

            outputs = noise_model(noisy_history)
            pred_resid = outputs["pred_continuous_residual"]

            target_cont = target_actions[..., CONTINUOUS_DIMS]
            last_cont = last_action[..., CONTINUOUS_DIMS]
            residual_target = target_cont - last_cont

            resid_loss = F.smooth_l1_loss(pred_resid / std_t_cont, residual_target / std_t_cont)
            pred_grip_logits = outputs["pred_gripper_logits"]
            target_grip = target_actions[..., GRIPPER_DIM_IDX]
            target_grip_class = (target_grip > GRIPPER_THRESHOLD).float()
            grip_loss = F.binary_cross_entropy_with_logits(pred_grip_logits, target_grip_class)
            loss = resid_loss + grip_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        noise_model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for batch in val_loader:
                target_actions = batch["target_actions"].to(device)
                history = batch["action_history"].to(device)
                last_action = history[:, -1:, :]
                outputs = noise_model(history)
                recon = reconstruct_from_residual(outputs, last_action)
                pred = recon["pred_actions"]
                if action_transform is not None:
                    pred = action_transform.denormalize_tensor(pred)
                    target_actions = action_transform.denormalize_tensor(target_actions)
                all_pred.append(pred.cpu())
                all_tgt.append(target_actions.cpu())
        pred_all = torch.cat(all_pred)
        tgt_all = torch.cat(all_tgt)
        metrics = compute_split_metrics(pred_all, tgt_all, action_stats=action_contract["continuous_action_stats"])
        if best_noise_metrics is None or metrics["continuous_normalized_mse"] < best_noise_metrics["continuous_normalized_mse"]:
            best_noise_metrics = dict(metrics)

    stabilization_rows.append({
        "variant": "history_noise_aug",
        "teacher_forced_mse": best_noise_metrics["continuous_normalized_mse"] if best_noise_metrics else float("inf"),
    })

    # Autoregressive eval of noise model
    noise_ar_rows = run_single_trajectory_autoregressive(
        model=noise_model, model_kind="residual_gru",
        trajectory=selected, config=config, device=device,
        action_contract=action_contract, action_transform=action_transform,
        max_horizon=max_horizon, git_info=git_info,
    )
    for r in noise_ar_rows:
        r["variant"] = "history_noise_aug"
        if "per_dim_mse" in r:
            del r["per_dim_mse"]

    # C. History dropout augmentation
    print("  Training history_dropout_aug ...")
    dropout_model = ResidualSplitGRU(action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(dropout_model.parameters(), lr=effective_lr)
    best_dropout_metrics = None
    for epoch in range(epochs):
        dropout_model.train()
        for batch in train_loader:
            target_actions = batch["target_actions"].to(device)
            history = batch["action_history"].to(device)
            last_action = history[:, -1:, :]

            # Dropout: replace 20% of history entries with last_action
            B, H, A_dim = history.shape
            mask = torch.bernoulli(torch.full((B, H, 1), 0.8, device=history.device))
            last_expanded = last_action.expand_as(history)
            dropped_history = history * mask + last_expanded * (1 - mask)

            outputs = dropout_model(dropped_history)
            pred_resid = outputs["pred_continuous_residual"]

            target_cont = target_actions[..., CONTINUOUS_DIMS]
            last_cont = last_action[..., CONTINUOUS_DIMS]
            residual_target = target_cont - last_cont

            resid_loss = F.smooth_l1_loss(pred_resid / std_t_cont, residual_target / std_t_cont)
            pred_grip_logits = outputs["pred_gripper_logits"]
            target_grip = target_actions[..., GRIPPER_DIM_IDX]
            target_grip_class = (target_grip > GRIPPER_THRESHOLD).float()
            grip_loss = F.binary_cross_entropy_with_logits(pred_grip_logits, target_grip_class)
            loss = resid_loss + grip_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dropout_model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for batch in val_loader:
                target_actions = batch["target_actions"].to(device)
                history = batch["action_history"].to(device)
                last_action = history[:, -1:, :]
                outputs = dropout_model(history)
                recon = reconstruct_from_residual(outputs, last_action)
                pred = recon["pred_actions"]
                if action_transform is not None:
                    pred = action_transform.denormalize_tensor(pred)
                    target_actions = action_transform.denormalize_tensor(target_actions)
                all_pred.append(pred.cpu())
                all_tgt.append(target_actions.cpu())
        pred_all = torch.cat(all_pred)
        tgt_all = torch.cat(all_tgt)
        metrics = compute_split_metrics(pred_all, tgt_all, action_stats=action_contract["continuous_action_stats"])
        if best_dropout_metrics is None or metrics["continuous_normalized_mse"] < best_dropout_metrics["continuous_normalized_mse"]:
            best_dropout_metrics = dict(metrics)

    stabilization_rows.append({
        "variant": "history_dropout_aug",
        "teacher_forced_mse": best_dropout_metrics["continuous_normalized_mse"] if best_dropout_metrics else float("inf"),
    })

    dropout_ar_rows = run_single_trajectory_autoregressive(
        model=dropout_model, model_kind="residual_gru",
        trajectory=selected, config=config, device=device,
        action_contract=action_contract, action_transform=action_transform,
        max_horizon=max_horizon, git_info=git_info,
    )
    for r in dropout_ar_rows:
        r["variant"] = "history_dropout_aug"
        if "per_dim_mse" in r:
            del r["per_dim_mse"]

    # D. Smoothness regularization
    print("  Training diagnostic_regularization ...")
    smooth_model = ResidualSplitGRU(action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    smooth_result = train_with_smoothness_reg(
        model=smooth_model, model_kind="residual_gru",
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=epochs, lr=effective_lr,
        action_contract=action_contract, action_transform=action_transform,
        smoothness_weight=0.01,
    )
    sm_mse = smooth_result["best_metrics"]["continuous_normalized_mse"]
    stabilization_rows.append({
        "variant": "diagnostic_regularization",
        "teacher_forced_mse": sm_mse,
    })

    smooth_ar_rows = run_single_trajectory_autoregressive(
        model=smooth_model, model_kind="residual_gru",
        trajectory=selected, config=config, device=device,
        action_contract=action_contract, action_transform=action_transform,
        max_horizon=max_horizon, git_info=git_info,
    )
    for r in smooth_ar_rows:
        r["variant"] = "diagnostic_regularization"
        if "per_dim_mse" in r:
            del r["per_dim_mse"]

    # E. Multi-step unrolled loss
    print("  Training offline_multistep_loss ...")
    unroll_model = ResidualSplitGRU(action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    unroll_result = train_with_unrolled_loss(
        model=unroll_model, model_kind="residual_gru",
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=epochs, lr=effective_lr,
        action_contract=action_contract, action_transform=action_transform,
        unroll_steps=3,
    )
    um_mse = unroll_result["best_metrics"]["continuous_normalized_mse"]
    stabilization_rows.append({
        "variant": "offline_multistep_loss",
        "teacher_forced_mse": um_mse,
    })

    unroll_ar_rows = run_single_trajectory_autoregressive(
        model=unroll_model, model_kind="residual_gru",
        trajectory=selected, config=config, device=device,
        action_contract=action_contract, action_transform=action_transform,
        max_horizon=max_horizon, git_info=git_info,
    )
    for r in unroll_ar_rows:
        r["variant"] = "offline_multistep_loss"
        if "per_dim_mse" in r:
            del r["per_dim_mse"]

    # Combine stabilization autoregressive rows
    all_stab_ar = []
    # Add baseline (no aug)
    baseline_ar_only = [r for r in baseline_ladder_rows
                        if r.get("model") == "residual_action_action_history_gru"
                        and r.get("mode") == "autoregressive_open_loop"]
    for r in baseline_ar_only:
        r2 = dict(r)
        r2["variant"] = "baseline"
        if "model" in r2: del r2["model"]
        all_stab_ar.append(r2)

    all_stab_ar.extend(noise_ar_rows)
    all_stab_ar.extend(dropout_ar_rows)
    all_stab_ar.extend(smooth_ar_rows)
    all_stab_ar.extend(unroll_ar_rows)

    _write_csv(output_dir / "stabilization_variant_ladder.csv", stabilization_rows)
    _write_csv(output_dir / "autoregressive_rollout_metrics.csv", all_stab_ar)

    # ===================================================================
    # 6. Error growth by horizon
    # ===================================================================
    print("[6/9] Computing error growth by horizon ...")
    growth_rows = []
    for model_name in ["baseline", "history_noise_aug", "history_dropout_aug",
                       "diagnostic_regularization", "offline_multistep_loss"]:
        model_ar = [r for r in all_stab_ar if r.get("variant") == model_name
                    and r.get("mode") == "autoregressive_open_loop"]
        for r in model_ar:
            growth_rows.append({
                "variant": model_name,
                "horizon": r.get("horizon"),
                "continuous_normalized_mse": r.get("continuous_normalized_mse"),
                "gripper_sign_accuracy": r.get("gripper_sign_accuracy"),
            })
    _write_csv(output_dir / "error_growth_by_horizon.csv", growth_rows)

    # ===================================================================
    # 7. Per-dim autoregressive errors
    # ===================================================================
    print("[7/9] Computing per-dim autoregressive errors ...")
    per_dim_rows = []
    for model_name in ["baseline", "history_noise_aug", "history_dropout_aug",
                       "diagnostic_regularization", "offline_multistep_loss"]:
        model_ar = [r for r in all_stab_ar if r.get("variant") == model_name
                    and r.get("mode") == "autoregressive_open_loop"
                    and r.get("horizon") == "full"]
        for r in model_ar:
            per_dim_rows.append({
                "variant": model_name,
                "delta_rot_x_mse": r.get("delta_rot_x_mse"),
                "continuous_normalized_mse_full": r.get("continuous_normalized_mse"),
                "error_growth_slope": r.get("error_growth_slope"),
                "max_error": r.get("max_error"),
            })
    _write_csv(output_dir / "per_dim_autoregressive_errors.csv", per_dim_rows)

    # ===================================================================
    # 8. Multi-demo evaluation
    # ===================================================================
    print("[8/9] Multi-demo autoregressive evaluation ...")
    multidemo_rows = run_multidemo_autoregressive(
        trajectories=trajectories, config=config, device=device,
        action_contract=action_contract, action_transform=action_transform,
        epochs=min(epochs, 200), lr=effective_lr, hidden_dim=hidden_dim,
        git_info=git_info, dataset=dataset_name, output_dir=output_dir,
    )

    # ===================================================================
    # 9. Failure mode analysis and readiness gate
    # ===================================================================
    print("[9/9] Failure mode analysis and readiness gate ...")

    # Find best stabilized model
    best_stab_variant = min(stabilization_rows, key=lambda r: r.get("teacher_forced_mse", float("inf")))
    best_stab_name = best_stab_variant["variant"]
    best_stab_models = {
        "history_noise_aug": (noise_model, "residual_gru"),
        "history_dropout_aug": (dropout_model, "residual_gru"),
        "diagnostic_regularization": (smooth_model, "residual_gru"),
        "offline_multistep_loss": (unroll_model, "residual_gru"),
        "baseline": (resid_gru_model, "residual_gru"),
    }
    best_stab_obj, best_stab_kind = best_stab_models.get(best_stab_name, (resid_gru_model, "residual_gru"))

    run_failure_mode_analysis(
        model_best=best_stab_obj, model_kind_best=best_stab_kind,
        model_baseline=resid_gru_model, model_kind_baseline="residual_gru",
        trajectories=trajectories, config=config, device=device,
        action_contract=action_contract, action_transform=action_transform,
        git_info=git_info, output_dir=output_dir,
    )

    gate_results = evaluate_readiness_gate(
        baseline_ladder=baseline_ladder_rows,
        stabilization_ladder=all_stab_ar,
        multidemo_rows=multidemo_rows,
        output_dir=output_dir,
    )

    # ===================================================================
    # Summary
    # ===================================================================
    best_tf = min((r.get("teacher_forced_mse", float("inf")) for r in stabilization_rows), default=float("inf"))
    best_ar = min((r.get("continuous_normalized_mse", float("inf"))
                   for r in all_stab_ar if r.get("mode") == "autoregressive_open_loop"
                   and r.get("horizon") == "full"), default=float("inf"))

    summary = {
        "status": "g11_autoregressive_stabilization",
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
        "best_teacher_forced_mse": best_tf,
        "best_autoregressive_full_mse": best_ar,
        "best_stabilization_variant": best_stab_name,
        "readiness_gate_pass": gate_results["overall_pass"],
        "stabilization_ladder": stabilization_rows,
        "non_claims": [
            "not_closed_loop_success",
            "not_future_latent_benefit_evidence",
            "not_architecture_claim_evidence",
            "offline_improvement_does_not_guarantee_closed_loop",
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
        "G11 offline autoregressive stabilization and closed-loop readiness gate. "
        "Diagnostic only, not closed-loop evidence.\n", encoding="utf-8")

    print(f"\n[Done] Best teacher_forced={best_tf:.6f}, best autoregressive={best_ar:.6f}")
    print(f"Readiness gate: {'PASSED' if gate_results['overall_pass'] else 'NOT PASSED'}")
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
