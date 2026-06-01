#!/usr/bin/env python3
"""G6: Representation bottleneck diagnostics under causal next-action contract.

Tests whether different input representations can pass H=1 single-demo overfit
under the strict causal_next_action_v1 contract. This isolates representation
bottlenecks from architecture or future-latent modeling issues.

Representations tested:
  - Oracle low-dimensional state (proprio)
  - Raw image CNN
  - DINO CLS only
  - DINO patch mean
  - DINO CLS + patch mean
  - DINO patch tokens + attention pooling
  - Multi-camera DINO features (if available)

Additional diagnostics:
  - Representation-action retrieval metrics
  - Latent dynamics prediction (z_t + a_t -> z_{t+1})
  - Optional same-demo goal-feature planning diagnostic
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from src.data.trajectory_window import (  # noqa: E402
    RawTrajectory,
    TrajectoryWindowDataset,
)
from src.eval.overfit_diagnostics import (  # noqa: E402
    ShiftedTargetWindowDataset,
    _repair_forward,
    _repair_train_epoch,
    _repair_evaluate,
    _repair_loss,
    _repair_metrics_from_tensors,
    _summary_row,
    _write_csv,
    _write_json,
    _save_repair_checkpoint,
    causal_next_action_v1_check,
    run_causal_contract_tests,
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
    infer_current_latent_dim,
    infer_state_dim,
    load_real_libero_trajectories,
    uses_current_latent,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.experiment_io import capture_environment, capture_git_commit  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Git info
# ---------------------------------------------------------------------------

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
# Representation variant models
# ---------------------------------------------------------------------------

GRIPPER_DIM_ = 6  # local alias to avoid confusion with import


class OracleStateSplitMLP(nn.Module):
    """H=1: predict action[t] from proprio/state[t] only (oracle low-dim)."""

    def __init__(self, *, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.action_head(self.network(state_t))


class RawImageCNN(nn.Module):
    """H=1: predict action[t] from raw RGB observation[t] via small CNN."""

    def __init__(self, *, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        # With 128x128 input: Conv1 -> 32x32, Conv2 -> 16x16, Pool -> 4x4
        # Output: 32 * 4 * 4 = 512
        self.network = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        # image: [B, H, W, C] -> [B, C, H, W]
        if image.ndim == 4 and image.shape[-1] in (1, 3):
            img = image.permute(0, 3, 1, 2).float()
        else:
            img = image.float()
        # Normalize to [0,1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        features = self.cnn(img)
        return self.action_head(self.network(features))


class RawImageCNNWithState(nn.Module):
    """H=1: predict action[t] from raw RGB + proprio/state[t]."""

    def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.network = nn.Sequential(
            nn.Linear(512 + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, image: torch.Tensor, state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim == 4 and image.shape[-1] in (1, 3):
            img = image.permute(0, 3, 1, 2).float()
        else:
            img = image.float()
        if img.max() > 1.0:
            img = img / 255.0
        cnn_feat = self.cnn(img)
        features = torch.cat([cnn_feat, state_t], dim=-1)
        return self.action_head(self.network(features))


class DinoVariantMLP(nn.Module):
    """H=1: predict action[t] from arbitrary DINO feature vector + optional state."""

    def __init__(self, *, feature_dim: int, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, features: torch.Tensor, state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.action_head(self.network(torch.cat([features, state_t], dim=-1)))


class DinoAttentionPoolingMLP(nn.Module):
    """H=1: predict action[t] from DINO patch tokens via attention pooling + state."""

    def __init__(self, *, token_dim: int, n_tokens: int, state_dim: int,
                 hidden_dim: int, action_dim: int, attn_heads: int = 4) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        # Simple self-attention pooling
        self.attn = nn.MultiheadAttention(token_dim, num_heads=attn_heads, batch_first=True)
        self.pool_proj = nn.Linear(token_dim, token_dim)
        self.network = nn.Sequential(
            nn.Linear(token_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, tokens: torch.Tensor, state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        # tokens: [B, n_tokens, token_dim]
        attn_out, _ = self.attn(tokens, tokens, tokens)
        pooled = self.pool_proj(attn_out.mean(dim=1))  # [B, token_dim]
        features = torch.cat([pooled, state_t], dim=-1)
        return self.action_head(self.network(features))


class ActionHistoryGRU(nn.Module):
    """H=1: predict action[t] from action history only."""

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


class LatentDynamicsMLP(nn.Module):
    """Predict z_{t+1} from (z_t, a_t) for latent dynamics diagnostic."""

    def __init__(self, *, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([z_t, a_t], dim=-1))


class GoalConditionedDynamicsMLP(nn.Module):
    """Predict next action from (z_t, z_goal) for goal-feature planning diagnostic."""

    def __init__(self, *, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, z_t: torch.Tensor, z_goal: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.action_head(self.network(torch.cat([z_t, z_goal], dim=-1)))


# ---------------------------------------------------------------------------
# DINO feature extraction helpers
# ---------------------------------------------------------------------------

def extract_dino_features(traj: RawTrajectory, variant: str, *, patch_size: int = 14) -> list[list[float]]:
    """Extract DINO features from raw images for a given variant.

    Variants:
      - "cls": CLS token only (standard, already stored as visual_latents)
      - "patch_mean": mean of patch tokens
      - "cls_patch_mean": CLS concatenated with patch mean
    """
    # For CLS, we already have visual_latents
    if variant == "cls":
        if traj.visual_latents is None:
            return []
        return [list(v) for v in traj.visual_latents]

    # For other variants, we need to run DINOv2 on raw images
    if traj.images is None or len(traj.images) == 0:
        return []

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        raise RuntimeError("transformers library required for DINO feature extraction")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    encoder = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    encoder.eval()

    features_list = []
    batch_size = 16
    for start in range(0, len(traj.images), batch_size):
        end = min(start + batch_size, len(traj.images))
        batch_images = []
        for img in traj.images[start:end]:
            if isinstance(img, np.ndarray):
                batch_images.append(torch.from_numpy(img).float())
            elif isinstance(img, (list, tuple)):
                batch_images.append(torch.tensor(img, dtype=torch.float32))
            else:
                # Frame reference string - skip (no raw pixels available)
                return []

        batch_tensor = torch.stack(batch_images).to(device)
        if batch_tensor.max() > 1.0:
            batch_tensor = batch_tensor / 255.0

        inputs = processor(images=batch_tensor, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = encoder(**inputs)

        all_tokens = outputs.last_hidden_state  # [B, 1+n_patches, D]

        if variant == "patch_mean":
            # Mean of patch tokens (excluding CLS)
            feat = all_tokens[:, 1:, :].mean(dim=1)
        elif variant == "cls_patch_mean":
            cls = all_tokens[:, 0, :]
            patch_mean = all_tokens[:, 1:, :].mean(dim=1)
            feat = torch.cat([cls, patch_mean], dim=-1)
        else:
            raise ValueError(f"unknown DINO variant: {variant}")

        features_list.extend(feat.cpu().float().tolist())

    return features_list


# ---------------------------------------------------------------------------
# Representation-action retrieval diagnostics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    latents: np.ndarray,
    actions: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    """Compute representation-action retrieval diagnostics."""
    T, D = latents.shape
    A = actions.shape[1]

    # 1. Latent variance per dimension
    latent_var = latents.var(axis=0)
    latent_var_mean = float(latent_var.mean())
    latent_var_min = float(latent_var.min())
    latent_var_max = float(latent_var.max())

    # 2. Adjacent timestep cosine similarity
    def cosine_sim(a, b):
        dot = np.dot(a, b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(dot / (na * nb))

    adj_cos_sims = [cosine_sim(latents[t], latents[t + 1]) for t in range(T - 1)]
    mean_adj_cos = float(np.mean(adj_cos_sims))
    std_adj_cos = float(np.std(adj_cos_sims))

    # 3. Nearest-neighbor timestep retrieval (finds if latent encodes phase)
    nn_adjacent_correct = 0
    for t in range(T):
        dists = np.linalg.norm(latents - latents[t], axis=1)
        dists[t] = float("inf")
        nn_idx = int(np.argmin(dists))
        if abs(nn_idx - t) <= 1:
            nn_adjacent_correct += 1
    nn_adjacent_acc = nn_adjacent_correct / max(T, 1)

    # 4. Nearest-neighbor action retrieval MSE
    # For each latent, find k nearest latent neighbors and compare their actions
    action_nn_mses = []
    for t in range(T):
        dists = np.linalg.norm(latents - latents[t], axis=1)
        dists[t] = float("inf")
        sorted_idx = np.argsort(dists)
        # Use top-3 nearest neighbors
        k = min(3, T - 1)
        nn_actions = actions[sorted_idx[:k]]
        action_nn_mses.append(float(np.mean((nn_actions - actions[t]) ** 2)))
    action_nn_mse = float(np.mean(action_nn_mses))

    # 5. Correlation between latent distance and action distance
    # Sample pairs to avoid O(T^2)
    n_pairs = min(500, T * (T - 1) // 2)
    rng = np.random.RandomState(42)
    lat_dists = []
    act_dists = []
    for _ in range(n_pairs):
        i, j = rng.choice(T, 2, replace=False)
        lat_dists.append(float(np.linalg.norm(latents[i] - latents[j])))
        act_dists.append(float(np.linalg.norm(actions[i] - actions[j])))
    lat_dists = np.array(lat_dists)
    act_dists = np.array(act_dists)
    if lat_dists.std() > 1e-12 and act_dists.std() > 1e-12:
        latent_action_corr = float(np.corrcoef(lat_dists, act_dists)[0, 1])
    else:
        latent_action_corr = float("nan")

    # 6. PCA top-5 concentration
    centered = latents - latents.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = float((s ** 2).sum())
        top5_var = float((s[:5] ** 2).sum()) if len(s) >= 5 else total_var
        pca_concentration = top5_var / max(total_var, 1e-12)
    except Exception:
        pca_concentration = float("nan")

    return {
        "label": label,
        "n_timesteps": T,
        "latent_dim": D,
        "latent_var_mean": latent_var_mean,
        "latent_var_min": latent_var_min,
        "latent_var_max": latent_var_max,
        "adjacent_cosine_sim_mean": mean_adj_cos,
        "adjacent_cosine_sim_std": std_adj_cos,
        "nn_timestep_retrieval_accuracy": nn_adjacent_acc,
        "nn_action_retrieval_mse": action_nn_mse,
        "latent_action_distance_correlation": latent_action_corr,
        "pca_top5_concentration": pca_concentration,
    }


# ---------------------------------------------------------------------------
# Latent dynamics prediction
# ---------------------------------------------------------------------------

def run_latent_dynamics_diagnostic(
    *,
    trajectory: RawTrajectory,
    device: torch.device,
    epochs: int = 200,
    hidden_dim: int = 128,
    lr: float = 0.001,
    seed: int = 0,
) -> dict[str, Any]:
    """Train z_{t+1} = f(z_t, a_t) and evaluate prediction quality."""
    if trajectory.visual_latents is None:
        return {"error": "no_visual_latents"}

    latents = np.array(trajectory.visual_latents, dtype=np.float32)
    actions = np.array(trajectory.actions, dtype=np.float32)
    T = latents.shape[0]
    latent_dim = latents.shape[1]
    action_dim = actions.shape[1]

    if T < 3:
        return {"error": "trajectory_too_short"}

    # Build pairs (z_t, a_t) -> z_{t+1}
    z_t = torch.tensor(latents[:-1], dtype=torch.float32)
    a_t = torch.tensor(actions[:-1], dtype=torch.float32)
    z_next = torch.tensor(latents[1:], dtype=torch.float32)

    # Split: first 70% train, rest val
    n_train = int(0.7 * (T - 1))
    train_z, train_a, train_target = z_t[:n_train], a_t[:n_train], z_next[:n_train]
    val_z, val_a, val_target = z_t[n_train:], a_t[n_train:], z_next[n_train:]

    model = LatentDynamicsMLP(
        latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mse = float("inf")
    best_val_cos = float("inf")

    for epoch in range(epochs):
        model.train()
        pred = model(train_z.to(device), train_a.to(device))
        loss = F.mse_loss(pred, train_target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_z.to(device), val_a.to(device))
            val_mse = F.mse_loss(val_pred, val_target.to(device)).item()
            val_cos = 1.0 - F.cosine_similarity(
                val_pred, val_target.to(device), dim=-1,
            ).mean().item()
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_val_cos = val_cos

    # Nearest-neighbor next-frame retrieval
    model.eval()
    with torch.no_grad():
        all_pred = model(z_t.to(device), a_t.to(device)).cpu().numpy()
    nn_correct = 0
    for t in range(T - 1):
        dists = np.linalg.norm(z_next - all_pred[t], axis=1)
        nn_idx = int(np.argmin(dists))
        if nn_idx == t:
            nn_correct += 1
    nn_retrieval_acc = nn_correct / max(T - 1, 1)

    return {
        "best_val_mse": best_val_mse,
        "best_val_cosine_error": best_val_cos,
        "nn_next_frame_retrieval_accuracy": nn_retrieval_acc,
        "n_train_pairs": n_train,
        "n_val_pairs": T - 1 - n_train,
        "latent_dim": latent_dim,
        "action_dim": action_dim,
    }


# ---------------------------------------------------------------------------
# Goal-feature planning diagnostic
# ---------------------------------------------------------------------------

def run_goal_feature_planning(
    *,
    trajectory: RawTrajectory,
    device: torch.device,
    epochs: int = 200,
    hidden_dim: int = 128,
    lr: float = 0.001,
    horizon: int = 5,
    n_samples: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """Test whether learned latent dynamics can plan toward a goal latent.

    For each test timestep t:
    1. Set goal = z_{t+h} from the same demo
    2. From z_t, apply zero actions and random actions
    3. Check if the predicted latent trajectory moves closer to goal
    """
    if trajectory.visual_latents is None:
        return {"error": "no_visual_latents"}

    latents = np.array(trajectory.visual_latents, dtype=np.float32)
    actions = np.array(trajectory.actions, dtype=np.float32)
    T = latents.shape[0]
    latent_dim = latents.shape[1]
    action_dim = actions.shape[1]

    if T < horizon + 3:
        return {"error": "trajectory_too_short"}

    # First, train a latent dynamics model
    dynamics = LatentDynamicsMLP(
        latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=lr)

    z_t_all = torch.tensor(latents[:-1], dtype=torch.float32)
    a_t_all = torch.tensor(actions[:-1], dtype=torch.float32)
    z_next_all = torch.tensor(latents[1:], dtype=torch.float32)

    for epoch in range(epochs):
        dynamics.train()
        pred = dynamics(z_t_all.to(device), a_t_all.to(device))
        loss = F.mse_loss(pred, z_next_all.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dynamics.eval()
    rng = np.random.RandomState(seed)

    # Evaluate planning
    improvements = []
    with torch.no_grad():
        for _ in range(n_samples):
            t = rng.randint(0, T - horizon - 1)
            z_init = torch.tensor(latents[t], dtype=torch.float32).unsqueeze(0).to(device)
            z_goal = torch.tensor(latents[t + horizon], dtype=torch.float32).unsqueeze(0).to(device)

            # Baseline: zero actions
            z_zero = z_init.clone()
            for h in range(horizon):
                a_zero = torch.zeros(1, action_dim, device=device)
                z_zero = dynamics(z_zero, a_zero)

            # Baseline: random actions
            z_random = z_init.clone()
            for h in range(horizon):
                a_rand = torch.randn(1, action_dim, device=device) * 0.1
                z_random = dynamics(z_random, a_rand)

            # True actions (teacher-forced)
            z_true = z_init.clone()
            for h in range(horizon):
                a_true = torch.tensor(
                    actions[t + h], dtype=torch.float32,
                ).unsqueeze(0).to(device)
                z_true = dynamics(z_true, a_true)

            dist_init = float(F.cosine_similarity(z_init, z_goal, dim=-1).item())
            dist_zero = float(F.cosine_similarity(z_zero, z_goal, dim=-1).item())
            dist_random = float(F.cosine_similarity(z_random, z_goal, dim=-1).item())
            dist_true = float(F.cosine_similarity(z_true, z_goal, dim=-1).item())

            improvements.append({
                "t": t,
                "cos_sim_zero_vs_goal": dist_zero,
                "cos_sim_random_vs_goal": dist_random,
                "cos_sim_true_vs_goal": dist_true,
                "cos_sim_init_vs_goal": dist_init,
            })

    mean_improvements = {
        "mean_cos_sim_zero_vs_goal": float(np.mean([r["cos_sim_zero_vs_goal"] for r in improvements])),
        "mean_cos_sim_random_vs_goal": float(np.mean([r["cos_sim_random_vs_goal"] for r in improvements])),
        "mean_cos_sim_true_vs_goal": float(np.mean([r["cos_sim_true_vs_goal"] for r in improvements])),
        "mean_cos_sim_init_vs_goal": float(np.mean([r["cos_sim_init_vs_goal"] for r in improvements])),
    }

    # Whether true actions move closer to goal than random/zero
    true_better_than_zero = mean_improvements["mean_cos_sim_true_vs_goal"] > mean_improvements["mean_cos_sim_zero_vs_goal"]
    true_better_than_random = mean_improvements["mean_cos_sim_true_vs_goal"] > mean_improvements["mean_cos_sim_random_vs_goal"]

    return {
        **mean_improvements,
        "true_actions_better_than_zero": true_better_than_zero,
        "true_actions_better_than_random": true_better_than_random,
        "n_samples": n_samples,
        "horizon": horizon,
        "latent_dim": latent_dim,
        "action_dim": action_dim,
    }


# ---------------------------------------------------------------------------
# Training helper (reused from overfit_diagnostics)
# ---------------------------------------------------------------------------

def _train_causal_h1(
    *,
    model: nn.Module,
    model_kind: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    loss_threshold: float,
    action_transform,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Train a model under H=1 causal contract and return metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_metrics = None
    best_epoch = -1
    best_train_mse = float("inf")

    for epoch in range(epochs):
        train_loss, _, _ = _repair_train_epoch(
            model=model, model_kind=model_kind,
            loader=train_loader, device=device,
            optimizer=optimizer, loss_mode="split",
        )
        train_metrics = _repair_evaluate(
            model=model, model_kind=model_kind,
            loader=train_loader, device=device,
            action_transform=action_transform,
        )
        eval_metrics = _repair_evaluate(
            model=model, model_kind=model_kind,
            loader=val_loader, device=device,
            action_transform=action_transform,
        )
        train_metrics["loss"] = train_loss

        if eval_metrics["action_mse"] < (best_metrics["action_mse"] if best_metrics else float("inf")):
            best_metrics = dict(eval_metrics)
            best_epoch = epoch
            best_train_mse = train_metrics["action_mse"]
            if checkpoint_path is not None:
                _save_repair_checkpoint(
                    checkpoint_path, model=model, model_kind=model_kind,
                    best_epoch=best_epoch, best_metrics=best_metrics,
                )

    if best_metrics is None:
        raise RuntimeError("training produced no metrics")

    return {
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "best_train_mse": best_train_mse,
        "passed": best_metrics["action_mse"] <= loss_threshold,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/g6_representation_bottleneck"))
    parser.add_argument("--trajectory_id", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "any"], default="train")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--dynamics_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss_threshold", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--run_id", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = run_g6_diagnostics(
        config_path=args.config,
        output_root=args.output_dir,
        trajectory_id=args.trajectory_id,
        source_split=args.split,
        epochs=args.epochs,
        dynamics_epochs=args.dynamics_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_threshold=args.loss_threshold,
        device_name=args.device,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        run_id=args.run_id,
        command=[sys.executable, "-m", "src.eval.g6_representation_bottleneck", *(argv or sys.argv[1:])],
    )
    print(f"g6_output_dir={output_dir}")
    return 0


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_g6_diagnostics(
    *,
    config_path: Path,
    output_root: Path,
    trajectory_id: str | None = None,
    source_split: str = "train",
    epochs: int = 300,
    dynamics_epochs: int = 200,
    batch_size: int = 64,
    lr: float | None = None,
    loss_threshold: float = 1e-4,
    device_name: str = "cpu",
    seed: int = 0,
    hidden_dim: int = 256,
    run_id: str | None = None,
    command: Sequence[str] | None = None,
) -> Path:
    seed_everything(seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{timestamp}_g6_repr_bottleneck"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    config = load_config(config_path)
    effective_lr = float(lr if lr is not None else config["training"]["lr"])
    device = torch.device(device_name)

    # Load trajectories
    trajectories, source_metadata = load_real_libero_trajectories(config)
    selected = _select_trajectory(trajectories, trajectory_id, source_split)
    diagnostic_trajs = [replace(selected, split="train"), replace(selected, split="val")]

    action_transform, normalization_stats = build_action_transform(diagnostic_trajs, config)
    if action_transform is not None:
        diagnostic_trajs = apply_action_transform(diagnostic_trajs, action_transform)

    # Check what data fields are available
    state_dim = None
    if selected.states is not None and len(selected.states) > 0:
        sample_state = np.array(selected.states[0])
        state_dim = int(sample_state.shape[-1]) if sample_state.ndim >= 1 else None

    has_latents = selected.visual_latents is not None and len(selected.visual_latents) > 0
    has_images = selected.images is not None and len(selected.images) > 0
    # Check if images are raw arrays (not frame reference strings)
    images_are_raw = False
    if has_images:
        sample_img = selected.images[0]
        images_are_raw = isinstance(sample_img, np.ndarray) or (
            isinstance(sample_img, (list, tuple)) and len(sample_img) > 0
            and isinstance(sample_img[0], (list, int, float))
        )

    print(f"Data availability:")
    print(f"  state_dim: {state_dim}")
    print(f"  has_latents: {has_latents}")
    print(f"  has_images: {has_images}")
    print(f"  images_are_raw: {images_are_raw}")
    print(f"  trajectory_length: {selected.length}")
    print(f"  trajectory_id: {selected.trajectory_id}")

    # Build causal H=1 dataset (shift=0, action_horizon=1)
    train_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs, split="train", config=config,
        action_horizon=1, target_shift=0,
    )
    val_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs, split="val", config=config,
        action_horizon=1, target_shift=0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_action_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_action_batch)

    sample = train_ds[0]
    action_dim = infer_action_dim(sample)
    latent_dim = infer_current_latent_dim(sample) if has_latents else 0
    effective_state_dim = infer_state_dim(sample) or 0

    print(f"  action_dim: {action_dim}")
    print(f"  latent_dim: {latent_dim}")
    print(f"  effective_state_dim: {effective_state_dim}")
    print(f"  train_samples: {len(train_ds)}, val_samples: {len(val_ds)}")

    # Run causal contract test
    causal_contract = run_causal_contract_tests(val_ds)
    _write_json(output_dir / "causal_contract_tests.json", causal_contract)

    # Track all results
    all_rows: list[dict[str, Any]] = []
    git_info = get_git_info()

    # ===================================================================
    # 1. Oracle state baseline
    # ===================================================================
    print("[1/7] Oracle state baseline ...")
    if effective_state_dim > 0:
        oracle_model = OracleStateSplitMLP(
            state_dim=effective_state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
        ).to(device)
        oracle_result = _train_causal_h1(
            model=oracle_model, model_kind="proprio",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            loss_threshold=loss_threshold, action_transform=action_transform,
            checkpoint_path=output_dir / "oracle_state_best.pt",
        )
        all_rows.append(_make_g6_row(
            "oracle_state", "proprio", effective_state_dim,
            oracle_result, loss_threshold, git_info,
        ))
        print(f"  Oracle state: eval_mse={oracle_result['best_metrics']['action_mse']:.6e}, "
              f"passed={oracle_result['passed']}")
    else:
        all_rows.append(_make_g6_row(
            "oracle_state", "proprio", 0,
            {"best_metrics": {"action_mse": float("nan")}, "best_epoch": -1, "passed": False,
             "error": "state_dim_not_available"},
            loss_threshold, git_info,
        ))
        print("  Oracle state: SKIPPED (state_dim not available)")

    # ===================================================================
    # 2. Raw image CNN baseline
    # ===================================================================
    print("[2/7] Raw image CNN baseline ...")
    if images_are_raw:
        # Train raw image CNN
        # Need to build a dataset that returns raw images for the CNN
        # We'll use a custom approach since ShiftedTargetWindowDataset returns frame refs
        # For now, build a dataset that includes raw images

        class RawImageDataset:
            """Dataset wrapper that returns raw numpy images for CNN."""
            def __init__(self, base_dataset, trajectory):
                self.base = base_dataset
                self.traj = trajectory
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                sample = self.base[idx]
                t = sample["time_index"]
                sample["image_t_raw"] = self.traj.images[t]
                return sample

        raw_train_ds = RawImageDataset(train_ds, selected)
        raw_val_ds = RawImageDataset(val_ds, selected)

        raw_train_loader = DataLoader(raw_train_ds, batch_size=batch_size, shuffle=True,
                                       collate_fn=collate_action_batch)
        raw_val_loader = DataLoader(raw_val_ds, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_action_batch)

        # Custom training loop for CNN
        cnn_model = RawImageCNN(action_dim=action_dim, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=effective_lr)

        best_mse = float("inf")
        best_epoch = -1

        for epoch in range(epochs):
            cnn_model.train()
            for batch in raw_train_loader:
                img = batch["image_t_raw"]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                elif isinstance(img, (list, tuple)):
                    img = torch.tensor(img, dtype=torch.float32)
                else:
                    img = torch.tensor(np.array(img), dtype=torch.float32)
                img = img.to(device)
                target = batch["target_actions"].to(device)

                outputs = cnn_model(img)
                pred = outputs["pred_actions"] if isinstance(outputs, dict) else outputs
                loss = _repair_loss(outputs, target, mode="split")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Eval
            cnn_model.eval()
            pred_rows = []
            target_rows = []
            with torch.no_grad():
                for batch in raw_val_loader:
                    img = batch["image_t_raw"]
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img).float()
                    elif isinstance(img, (list, tuple)):
                        img = torch.tensor(img, dtype=torch.float32)
                    else:
                        img = torch.tensor(np.array(img), dtype=torch.float32)
                    img = img.to(device)
                    target = batch["target_actions"].to(device)
                    outputs = cnn_model(img)
                    pred = outputs["pred_actions"] if isinstance(outputs, dict) else outputs
                    if action_transform is not None:
                        pred = action_transform.denormalize_tensor(pred)
                        target = action_transform.denormalize_tensor(target)
                    pred_rows.append(pred.cpu())
                    target_rows.append(target.cpu())

            if pred_rows:
                pred_all = torch.cat(pred_rows, dim=0)
                target_all = torch.cat(target_rows, dim=0)
                val_mse = F.mse_loss(pred_all, target_all).item()
                if val_mse < best_mse:
                    best_mse = val_mse
                    best_epoch = epoch

        all_rows.append(_make_g6_row(
            "raw_image_cnn", "cnn_128", 512,
            {"best_metrics": {"action_mse": best_mse}, "best_epoch": best_epoch,
             "passed": best_mse <= loss_threshold},
            loss_threshold, git_info,
        ))
        print(f"  Raw image CNN: eval_mse={best_mse:.6e}, passed={best_mse <= loss_threshold}")
    else:
        all_rows.append(_make_g6_row(
            "raw_image_cnn", "cnn_128", 0,
            {"best_metrics": {"action_mse": float("nan")}, "best_epoch": -1, "passed": False,
             "error": "raw_images_not_available"},
            loss_threshold, git_info,
        ))
        print("  Raw image CNN: SKIPPED (raw images not available)")

    # ===================================================================
    # 3. DINO feature variant ladder
    # ===================================================================
    print("[3/7] DINO feature variant ladder ...")
    if has_latents:
        latents_np = np.array(selected.visual_latents, dtype=np.float32)
        actions_np = np.array(selected.actions, dtype=np.float32)

        # For each DINO variant, build features and train
        dino_variants = {
            "dino_cls": latents_np,  # [T, 384]
        }

        # If we have raw images, compute additional variants
        if images_are_raw:
            # Extract DINO patch features from raw images
            try:
                patch_mean_feats = extract_dino_features(selected, "patch_mean")
                if patch_mean_feats:
                    dino_variants["dino_patch_mean"] = np.array(patch_mean_feats, dtype=np.float32)

                cls_patch_feats = extract_dino_features(selected, "cls_patch_mean")
                if cls_patch_feats:
                    dino_variants["dino_cls_patch_mean"] = np.array(cls_patch_feats, dtype=np.float32)
            except Exception as e:
                print(f"  Warning: Could not extract DINO variants: {e}")

        for variant_name, feat_np in dino_variants.items():
            feat_dim = feat_np.shape[1]

            # Build a dataset that provides these features as z_t
            class DinoFeatureDataset:
                def __init__(self, base_dataset, features):
                    self.base = base_dataset
                    self.features = features
                def __len__(self):
                    return len(self.base)
                def __getitem__(self, idx):
                    sample = self.base[idx]
                    t = sample["time_index"]
                    sample["z_t"] = self.features[t]
                    return sample

            feat_train_ds = DinoFeatureDataset(train_ds, feat_np)
            feat_val_ds = DinoFeatureDataset(val_ds, feat_np)

            feat_train_loader = DataLoader(feat_train_ds, batch_size=batch_size, shuffle=True,
                                            collate_fn=collate_action_batch)
            feat_val_loader = DataLoader(feat_val_ds, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_action_batch)

            # Train DINO variant MLP
            dino_model = DinoVariantMLP(
                feature_dim=feat_dim, state_dim=effective_state_dim,
                hidden_dim=hidden_dim, action_dim=action_dim,
            ).to(device)

            result = _train_causal_h1(
                model=dino_model, model_kind="dino_proprio" if effective_state_dim > 0 else "latent_mlp",
                train_loader=feat_train_loader, val_loader=feat_val_loader,
                device=device, epochs=epochs, lr=effective_lr,
                loss_threshold=loss_threshold, action_transform=action_transform,
                checkpoint_path=output_dir / f"{variant_name}_best.pt",
            )
            all_rows.append(_make_g6_row(
                variant_name, f"mlp_{feat_dim}", feat_dim,
                result, loss_threshold, git_info,
            ))
            print(f"  {variant_name}: eval_mse={result['best_metrics']['action_mse']:.6e}, "
                  f"passed={result['passed']}")
    else:
        all_rows.append(_make_g6_row(
            "dino_cls", "mlp_384", 0,
            {"best_metrics": {"action_mse": float("nan")}, "best_epoch": -1, "passed": False,
             "error": "no_latents"},
            loss_threshold, git_info,
        ))
        print("  DINO variants: SKIPPED (no latents)")

    # ===================================================================
    # 4. Action history baseline (re-reference)
    # ===================================================================
    print("[4/7] Action history GRU baseline ...")
    history_len = int(config["data"]["history_len"])
    action_hist_model = ActionHistoryGRU(
        history_len=history_len, action_dim=action_dim, hidden_dim=hidden_dim,
    ).to(device)
    action_hist_result = _train_causal_h1(
        model=action_hist_model, model_kind="action_history_gru",
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=epochs, lr=effective_lr,
        loss_threshold=loss_threshold, action_transform=action_transform,
        checkpoint_path=output_dir / "action_history_gru_best.pt",
    )
    all_rows.append(_make_g6_row(
        "action_history_gru", "gru", hidden_dim,
        action_hist_result, loss_threshold, git_info,
    ))
    print(f"  Action history GRU: eval_mse={action_hist_result['best_metrics']['action_mse']:.6e}, "
          f"passed={action_hist_result['passed']}")

    # ===================================================================
    # 5. Representation-action retrieval diagnostics
    # ===================================================================
    print("[5/7] Representation-action retrieval diagnostics ...")
    retrieval_results = []
    actions_np = np.array(selected.actions, dtype=np.float32)

    if has_latents:
        retrieval_results.append(compute_retrieval_metrics(
            latents_np, actions_np, label="dino_cls",
        ))

    # Compute retrieval for oracle state if available
    if selected.states is not None:
        states_np = np.array(selected.states, dtype=np.float32)
        retrieval_results.append(compute_retrieval_metrics(
            states_np, actions_np, label="oracle_state",
        ))

    # Write retrieval report
    _write_retrieval_report(output_dir / "representation_action_retrieval_report.md", retrieval_results)

    # Write retrieval CSV
    retrieval_csv_path = output_dir / "representation_retrieval_metrics.csv"
    if retrieval_results:
        keys = list(retrieval_results[0].keys())
        with retrieval_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(retrieval_results)

    # ===================================================================
    # 6. Latent dynamics prediction
    # ===================================================================
    print("[6/7] Latent dynamics prediction ...")
    dynamics_rows = []
    if has_latents:
        dynamics_result = run_latent_dynamics_diagnostic(
            trajectory=selected, device=device,
            epochs=dynamics_epochs, hidden_dim=hidden_dim,
            lr=effective_lr, seed=seed,
        )
        dynamics_result["variant"] = "dino_cls"
        dynamics_result["git_commit"] = git_info["commit"]
        dynamics_result["git_dirty"] = git_info["dirty"]
        dynamics_result["dataset"] = source_metadata.get("suite", "unknown")
        dynamics_result["trajectory_id"] = selected.trajectory_id
        dynamics_result["task_id"] = selected.task_id
        dynamics_result["task_name"] = selected.task_name
        dynamics_rows.append(dynamics_result)
        print(f"  DINO CLS dynamics: val_mse={dynamics_result.get('best_val_mse', 'N/A')}, "
              f"val_cos_error={dynamics_result.get('best_val_cosine_error', 'N/A')}")

    # Also run dynamics for oracle state if available
    if selected.states is not None and state_dim is not None:
        # Build a trajectory-like object with states as "latents"
        state_traj = replace(selected, visual_latents=[list(s) for s in selected.states])
        state_dynamics = run_latent_dynamics_diagnostic(
            trajectory=state_traj, device=device,
            epochs=dynamics_epochs, hidden_dim=hidden_dim,
            lr=effective_lr, seed=seed,
        )
        state_dynamics["variant"] = "oracle_state"
        state_dynamics["git_commit"] = git_info["commit"]
        state_dynamics["git_dirty"] = git_info["dirty"]
        state_dynamics["dataset"] = source_metadata.get("suite", "unknown")
        state_dynamics["trajectory_id"] = selected.trajectory_id
        state_dynamics["task_id"] = selected.task_id
        state_dynamics["task_name"] = selected.task_name
        dynamics_rows.append(state_dynamics)
        print(f"  Oracle state dynamics: val_mse={state_dynamics.get('best_val_mse', 'N/A')}, "
              f"val_cos_error={state_dynamics.get('best_val_cosine_error', 'N/A')}")

    _write_csv(output_dir / "latent_dynamics_prediction.csv", dynamics_rows)

    # ===================================================================
    # 7. Goal-feature planning (optional, cheap variant)
    # ===================================================================
    print("[7/7] Goal-feature planning diagnostic ...")
    planning_rows = []
    if has_latents:
        planning_result = run_goal_feature_planning(
            trajectory=selected, device=device,
            epochs=dynamics_epochs, hidden_dim=hidden_dim,
            lr=effective_lr, horizon=5, n_samples=50, seed=seed,
        )
        planning_result["variant"] = "dino_cls"
        planning_result["git_commit"] = git_info["commit"]
        planning_result["git_dirty"] = git_info["dirty"]
        planning_result["dataset"] = source_metadata.get("suite", "unknown")
        planning_result["trajectory_id"] = selected.trajectory_id
        planning_result["task_id"] = selected.task_id
        planning_result["task_name"] = selected.task_name
        planning_rows.append(planning_result)
        print(f"  Goal planning: true_better_than_zero={planning_result.get('true_actions_better_than_zero', 'N/A')}, "
              f"true_better_than_random={planning_result.get('true_actions_better_than_random', 'N/A')}")

    if planning_rows:
        _write_csv(output_dir / "goal_feature_planning_diagnostic.csv", planning_rows)

    # ===================================================================
    # Write all output CSVs
    # ===================================================================
    _write_csv(output_dir / "oracle_state_baseline.csv",
               [r for r in all_rows if r["variant"] == "oracle_state"])
    _write_csv(output_dir / "raw_image_cnn_overfit.csv",
               [r for r in all_rows if r["variant"] == "raw_image_cnn"])
    _write_csv(output_dir / "dino_feature_variant_ladder.csv",
               [r for r in all_rows if "dino" in r["variant"]])

    # Write comprehensive results CSV
    _write_csv(output_dir / "g6_all_results.csv", all_rows)

    # Write summary
    summary = {
        "status": "g6_representation_bottleneck",
        "config": str(config_path),
        "trajectory_id": selected.trajectory_id,
        "trajectory_length": selected.length,
        "task_id": selected.task_id,
        "task_name": selected.task_name,
        "dataset": source_metadata.get("suite", "unknown"),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "seed": seed,
        "epochs": epochs,
        "dynamics_epochs": dynamics_epochs,
        "loss_threshold": loss_threshold,
        "hidden_dim": hidden_dim,
        "state_dim_available": state_dim is not None,
        "state_dim": state_dim,
        "has_latents": has_latents,
        "has_raw_images": images_are_raw,
        "causal_contract_pass": causal_contract.get("pass", False),
        "results": all_rows,
        "retrieval_metrics": retrieval_results,
        "dynamics_results": dynamics_rows,
        "planning_results": planning_rows,
        "non_claims": [
            "not_closed_loop_success",
            "not_future_latent_benefit_evidence",
            "not_architecture_claim_evidence",
            "not_policy_validity_evidence",
        ],
    }
    _write_json(output_dir / "summary.json", summary)

    # Write repro files
    _write_repro_files(
        output_dir=output_dir,
        config_path=config_path,
        command=command,
        source_metadata=source_metadata,
        normalization_stats=normalization_stats,
        seed=seed,
        git_info=git_info,
    )

    # Write diagnostic report
    _write_diagnostic_report(output_dir / "diagnostic_report.md", summary, all_rows, retrieval_results)

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


def _make_g6_row(
    variant: str,
    model_type: str,
    feature_dim: int,
    result: dict[str, Any],
    threshold: float,
    git_info: dict[str, str],
) -> dict[str, Any]:
    metrics = result.get("best_metrics", {})
    return {
        "variant": variant,
        "model_type": model_type,
        "feature_dim": feature_dim,
        "eval_mse": metrics.get("action_mse", float("nan")),
        "continuous_mse": metrics.get("continuous_mse", float("nan")),
        "gripper_mse": metrics.get("gripper_mse", float("nan")),
        "gripper_sign_accuracy": metrics.get("gripper_sign_accuracy", float("nan")),
        "best_epoch": result.get("best_epoch", -1),
        "passed": result.get("passed", False),
        "error": result.get("error", ""),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "threshold": threshold,
    }


def _write_retrieval_report(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Representation-Action Retrieval Report",
        "",
        "G6 diagnostic: whether each representation encodes control-relevant state vs. demo phase.",
        "",
        "## Metrics",
        "",
        "| Label | Var Mean | Adj Cos Sim | NN Timestep Acc | NN Action MSE | Lat-Act Corr | PCA Top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['latent_var_mean']:.6e} | "
            f"{r['adjacent_cosine_sim_mean']:.6f} | "
            f"{r['nn_timestep_retrieval_accuracy']:.4f} | "
            f"{r['nn_action_retrieval_mse']:.6e} | "
            f"{r['latent_action_distance_correlation']:.4f} | "
            f"{r['pca_top5_concentration']:.4f} |"
        )

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "- **High NN timestep accuracy** (close to 1.0): representation mainly encodes demo phase/time,",
        "  not fine-grained control state. This would mean the representation collapses to a phase indicator.",
        "- **Low NN action MSE**: representation captures action-relevant information well.",
        "- **High latent-action distance correlation**: latent distance tracks action distance,",
        "  suggesting the representation encodes control-relevant variation.",
        "- **High PCA concentration**: most variance in few dimensions, possibly indicating",
        "  low effective dimensionality.",
        "",
        "These metrics diagnose representation quality for control, not policy quality.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_diagnostic_report(
    path: Path,
    summary: dict[str, Any],
    all_rows: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
) -> None:
    lines = [
        "# G6 Representation Bottleneck Diagnostic",
        "",
        f"Dataset: {summary['dataset']}",
        f"Trajectory: {summary['trajectory_id']} (length={summary['trajectory_length']})",
        f"Task: {summary['task_name']} (id={summary['task_id']})",
        f"Git commit: {summary['git_commit']}",
        f"Git dirty: {summary['git_dirty']}",
        f"Loss threshold: {summary['loss_threshold']}",
        "",
        "## Causal Contract",
        f"- Pass: {summary['causal_contract_pass']}",
        "",
        "## H=1 Baseline Results",
        "",
        "| Variant | Model Type | Feature Dim | Eval MSE | Passed |",
        "|---|---|---:|---:|---|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['variant']} | {row['model_type']} | {row['feature_dim']} | "
            f"{row['eval_mse']:.6e} | {row['passed']} |"
        )

    passed = [r for r in all_rows if r["passed"]]
    failed = [r for r in all_rows if not r["passed"] and not r.get("error")]

    lines.extend([
        "",
        "## Pass/Fail Summary",
        f"- Passed: {len(passed)}",
        f"- Failed: {len(failed)}",
        f"- Skipped: {len([r for r in all_rows if r.get('error')])}",
        "",
    ])

    if passed:
        lines.append("**Passing variants:**")
        for r in passed:
            lines.append(f"- {r['variant']}: eval_mse={r['eval_mse']:.6e}")
        lines.append("")

    if failed:
        lines.append("**Failing variants:**")
        for r in failed:
            lines.append(f"- {r['variant']}: eval_mse={r['eval_mse']:.6e}")
        lines.append("")

    lines.extend([
        "## Representation-Action Retrieval",
        "See `representation_action_retrieval_report.md` for detailed analysis.",
        "",
        "## Latent Dynamics Prediction",
        "See `latent_dynamics_prediction.csv` for details.",
        "",
        "## Goal-Feature Planning",
        "See `goal_feature_planning_diagnostic.csv` for details.",
        "",
        "## Interpretation Boundaries",
        "",
        "- This is a single-demo H=1 overfit diagnostic under strict causal contract.",
        "- It does not measure closed-loop success, generalization, or policy validity.",
        "- Future-latent benefit/harm is not claimed from this diagnostic.",
        "- WAM-GRU architecture validity is not claimed from this diagnostic.",
        "- DINOv2 suitability is not claimed from this diagnostic.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_repro_files(
    *,
    output_dir: Path,
    config_path: Path,
    command: Sequence[str] | None,
    source_metadata: dict[str, Any],
    normalization_stats: dict[str, Any],
    seed: int,
    git_info: dict[str, str],
) -> None:
    import shutil
    shutil.copyfile(config_path, output_dir / "config.yaml")
    _write_json(output_dir / "split.json", source_metadata)
    _write_json(output_dir / "normalization_stats.json", normalization_stats)
    (output_dir / "git_commit.txt").write_text(
        f"commit={git_info['commit']}\n"
        f"dirty={git_info['dirty']}\n",
        encoding="utf-8",
    )
    (output_dir / "environment.txt").write_text(capture_environment(), encoding="utf-8")
    (output_dir / "seeds.txt").write_text(f"{seed}\n", encoding="utf-8")
    (output_dir / "command.txt").write_text(
        (" ".join(command) if command else " ".join(sys.argv)) + "\n",
        encoding="utf-8",
    )
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output_dir / "environment.json", env_info)
    (output_dir / "notes.md").write_text(
        "G6 representation bottleneck diagnostic. "
        "Single-demo H=1 overfit under strict causal contract. "
        "Not closed-loop, not architecture-claim evidence.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
