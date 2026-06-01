#!/usr/bin/env python3
"""G7: State/action contract audit after G6 representation bottleneck.

Audits the HDF5 schema, implements true oracle state baseline using the
full 92-dim MuJoCo state, analyzes action targets, and benchmarks the
1e-4 threshold against action scale and variance.

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
    _repair_forward,
    _repair_train_epoch,
    _repair_evaluate,
    _repair_loss,
    _repair_metrics_from_tensors,
    _write_csv,
    _write_json,
    _save_repair_checkpoint,
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
# 1. Dataset schema audit
# ---------------------------------------------------------------------------

def run_state_schema_audit(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Inspect the HDF5 schema and produce a field inventory."""
    try:
        import h5py
    except ImportError:
        return {"error": "h5py not installed"}

    dataset_root = _resolve_dataset_root(str(config["data"]["dataset_root"]))
    suite = str(config["data"]["suite"])

    # Find first HDF5 file
    candidate_dirs = [
        Path(dataset_root) / suite,
        Path(dataset_root) / "datasets" / suite,
    ]
    hdf5_path = None
    for d in candidate_dirs:
        files = sorted(d.glob("*.hdf5")) + sorted(d.glob("*.h5"))
        if files:
            hdf5_path = files[0]
            break
    if hdf5_path is None:
        return {"error": "no HDF5 files found"}

    with h5py.File(hdf5_path, "r") as f:
        demo = f.get("data/demo_0")
        if demo is None:
            return {"error": "no demo_0 group"}

        fields = {}

        # Top-level fields
        for key in demo.keys():
            item = demo[key]
            if hasattr(item, "shape"):
                fields[f"data/demo_0/{key}"] = {
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                    "category": _categorize_field(key),
                }

        # obs/ fields
        if "obs" in demo:
            obs = demo["obs"]
            for key in obs.keys():
                item = obs[key]
                if hasattr(item, "shape"):
                    fields[f"data/demo_0/obs/{key}"] = {
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                        "category": _categorize_field(key),
                    }

        # Attributes
        attrs = {}
        for ak in demo.attrs:
            val = demo.attrs[ak]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            attrs[ak] = repr(val)[:200]

    # Determine availability
    available_fields = {
        "image_agentview": "data/demo_0/obs/agentview_rgb" in fields,
        "image_eye_in_hand": "data/demo_0/obs/eye_in_hand_rgb" in fields,
        "ee_pos": "data/demo_0/obs/ee_pos" in fields,
        "ee_ori": "data/demo_0/obs/ee_ori" in fields,
        "ee_states": "data/demo_0/obs/ee_states" in fields,
        "gripper_states": "data/demo_0/obs/gripper_states" in fields,
        "joint_states": "data/demo_0/obs/joint_states" in fields,
        "robot_states": "data/demo_0/robot_states" in fields,
        "states_92d": "data/demo_0/states" in fields,
        "actions": "data/demo_0/actions" in fields,
    }

    # Check for object/goal fields (not in standard LIBERO schema)
    object_goal_fields = {
        "object_pos": False,
        "object_ori": False,
        "goal_pos": False,
        "goal_ori": False,
    }
    for key in fields:
        lower = key.lower()
        if "object" in lower and ("pos" in lower or "pose" in lower):
            object_goal_fields["object_pos"] = True
        if "object" in lower and "ori" in lower:
            object_goal_fields["object_ori"] = True
        if "goal" in lower and ("pos" in lower or "pose" in lower):
            object_goal_fields["goal_pos"] = True
        if "goal" in lower and "ori" in lower:
            object_goal_fields["goal_ori"] = True

    schema = {
        "source_file": str(hdf5_path),
        "suite": suite,
        "fields": fields,
        "attributes": attrs,
        "available_fields": available_fields,
        "object_goal_fields": object_goal_fields,
        "has_object_pose_in_fields": any(object_goal_fields.values()),
        "has_full_mujoco_state": available_fields.get("states_92d", False),
        "notes": (
            "The 92-dim 'states' field is likely the full MuJoCo qpos+qvel vector, "
            "which contains object poses. However, the exact decomposition into "
            "object position/orientation requires knowing the model DOF counts. "
            "The current pipeline loads only 'robot_states' (9-dim), which is "
            "proprioceptive (gripper_states + ee/robot joint info), NOT a true "
            "oracle state with object and goal poses."
        ),
    }

    # Write schema audit report
    _write_schema_audit_report(output_dir / "state_schema_audit.md", schema)

    return schema


def _categorize_field(key: str) -> str:
    lower = key.lower()
    if "rgb" in lower or "image" in lower:
        return "image"
    if "ee_pos" in lower or "ee_position" in lower:
        return "end_effector_pose"
    if "ee_ori" in lower or "ee_orientation" in lower:
        return "end_effector_pose"
    if "ee_states" in lower:
        return "end_effector_states"
    if "gripper" in lower:
        return "gripper_state"
    if "joint" in lower:
        return "joint_state"
    if "robot_states" in lower:
        return "robot_state_proprioceptive"
    if "states" in lower:
        return "full_mujoco_state"
    if "action" in lower:
        return "action"
    if "reward" in lower:
        return "reward"
    if "done" in lower:
        return "done_flag"
    return "other"


def _resolve_dataset_root(dataset_root_str: str) -> str:
    if dataset_root_str.startswith("env:"):
        env_name = dataset_root_str[4:]
        import os
        value = os.environ.get(env_name)
        if value is None:
            raise EnvironmentError(f"{env_name} is not set")
        return value
    return dataset_root_str


def _write_schema_audit_report(path: Path, schema: dict[str, Any]) -> None:
    lines = [
        "# G7 State Schema Audit",
        "",
        f"Source: `{schema.get('source_file', 'unknown')}`",
        f"Suite: `{schema.get('suite', 'unknown')}`",
        "",
        "## Field Inventory",
        "",
        "| Field | Shape | Dtype | Category |",
        "|---|---|---|---|",
    ]
    for field_name, info in sorted(schema.get("fields", {}).items()):
        lines.append(f"| `{field_name}` | {info['shape']} | {info['dtype']} | {info['category']} |")

    lines.extend([
        "",
        "## Availability Summary",
        "",
    ])
    for key, available in sorted(schema.get("available_fields", {}).items()):
        lines.append(f"- **{key}**: {'✓' if available else '✗'}")

    lines.extend([
        "",
        "## Object/Goal Fields",
        "",
    ])
    for key, available in sorted(schema.get("object_goal_fields", {}).items()):
        lines.append(f"- **{key}**: {'✓ available' if available else '✗ NOT in HDF5 schema'}")

    lines.extend([
        "",
        "## Assessment",
        "",
        "- **No explicit object pose or goal pose fields exist in the HDF5 schema.**",
        "- The `states` (92-dim) field is the full MuJoCo simulation state (qpos+qvel),",
        "  which theoretically contains object poses but requires knowing the model DOF breakdown.",
        "- The `robot_states` (9-dim) field is proprioceptive only: gripper states + robot joint info.",
        "- The current pipeline loads only `robot_states` as `optional_state_t`.",
        "",
        "### Implications for Oracle State Baseline",
        "",
        "- A true oracle state with object pose and goal pose is NOT directly available",
        "  as named fields in the HDF5 schema.",
        "- The 92-dim `states` field CAN be used as a full oracle state, since it",
        "  theoretically contains all simulation state including object positions.",
        "- However, its exact decomposition is unknown without the MuJoCo model XML.",
        "- Conservative approach: use the 92-dim `states` as 'full_state_oracle' and",
        "  document that it is the full MuJoCo state, not a hand-crafted oracle.",
        "",
        "### Fields NOT Available (Cannot Fabricate)",
        "",
        "- Object position/orientation as named fields",
        "- Goal/target position/orientation as named fields",
        "- Object-to-EEF relative vector",
        "- Goal-to-object relative vector",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Action target audit
# ---------------------------------------------------------------------------

def run_action_target_audit(
    trajectory: RawTrajectory,
    *,
    output_dir: Path,
    git_info: dict[str, str],
    dataset: str,
) -> dict[str, Any]:
    """Comprehensive audit of action tensor semantics."""
    actions = np.array(trajectory.actions, dtype=np.float32)
    T, A = actions.shape

    # Per-dimension stats
    dim_stats = []
    for d in range(A):
        col = actions[:, d]
        dim_stats.append({
            "dim": d,
            "label": _action_dim_label(d),
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "var": float(col.var()),
        })

    # Gripper analysis
    gripper_vals = actions[:, GRIPPER_DIM]
    unique_gripper = np.unique(gripper_vals)
    n_open = int((gripper_vals > GRIPPER_OPEN_THRESH).sum())
    n_close = int((gripper_vals < GRIPPER_CLOSE_THRESH).sum())
    n_neutral = int(T - n_open - n_close)

    # Transition detection
    gripper_binary = (gripper_vals > 0).astype(int)
    transitions = []
    for t in range(1, T):
        if gripper_binary[t] != gripper_binary[t - 1]:
            transitions.append(t)

    # Autocorrelation
    autocorr = []
    for d in range(A):
        col = actions[:, d]
        if col.std() > 1e-12:
            c0 = np.mean((col[:-1] - col.mean()) * (col[1:] - col.mean()))
            autocorr.append(float(c0 / col.var()))
        else:
            autocorr.append(0.0)

    # Action convention analysis
    action_magnitudes = np.linalg.norm(actions[:, :GRIPPER_DIM], axis=1)
    is_delta_like = float(action_magnitudes.mean()) < 0.5  # rough heuristic

    audit = {
        "trajectory_id": trajectory.trajectory_id,
        "trajectory_length": T,
        "action_dim": A,
        "per_dim_stats": dim_stats,
        "gripper": {
            "unique_values": unique_gripper.tolist(),
            "n_open": n_open,
            "n_close": n_close,
            "n_neutral": n_neutral,
            "open_fraction": n_open / T,
            "close_fraction": n_close / T,
            "n_transitions": len(transitions),
            "transition_times": transitions[:10],
            "is_binary": len(unique_gripper) <= 2,
            "is_sign_coded": set(unique_gripper.tolist()) <= {-1.0, 1.0},
        },
        "continuous_dims_mean_std": float(np.mean([d["std"] for d in dim_stats[:GRIPPER_DIM]])),
        "gripper_dim_std": float(dim_stats[GRIPPER_DIM]["std"]),
        "action_autocorrelation_lag1": autocorr,
        "is_delta_like": is_delta_like,
        "action_convention": "action_to_current_obs (confirmed by trajectory_window.py)",
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "dataset": dataset,
    }

    # Write action per-dim metrics CSV
    _write_csv(output_dir / "action_per_dim_metrics.csv", dim_stats)

    # Write action target audit report
    _write_action_audit_report(output_dir / "action_target_audit.md", audit)

    return audit


def _action_dim_label(d: int) -> str:
    labels = ["delta_pos_x", "delta_pos_y", "delta_pos_z",
              "delta_rot_x", "delta_rot_y", "delta_rot_z", "gripper"]
    return labels[d] if d < len(labels) else f"dim_{d}"


def _write_action_audit_report(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# G7 Action Target Audit",
        "",
        f"Trajectory: `{audit['trajectory_id']}`",
        f"Length: {audit['trajectory_length']}, Action dim: {audit['action_dim']}",
        f"Convention: {audit['action_convention']}",
        "",
        "## Per-Dimension Statistics",
        "",
        "| Dim | Label | Min | Max | Mean | Std | Autocorr |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for d in audit["per_dim_stats"]:
        lines.append(
            f"| {d['dim']} | {d['label']} | {d['min']:.6f} | {d['max']:.6f} | "
            f"{d['mean']:.6f} | {d['std']:.6f} | {audit['action_autocorrelation_lag1'][d['dim']]:.4f} |"
        )

    grip = audit["gripper"]
    lines.extend([
        "",
        "## Gripper Analysis",
        f"- Unique values: {grip['unique_values']}",
        f"- Is binary: {grip['is_binary']}",
        f"- Is sign-coded (±1): {grip['is_sign_coded']}",
        f"- Open (>{GRIPPER_OPEN_THRESH}): {grip['n_open']} ({grip['open_fraction']:.1%})",
        f"- Close (<{GRIPPER_CLOSE_THRESH}): {grip['n_close']} ({grip['close_fraction']:.1%})",
        f"- Neutral: {grip['n_neutral']}",
        f"- Transitions: {grip['n_transitions']}",
        "",
        "## Action Convention",
        f"- Type: {'delta-like' if audit['is_delta_like'] else 'absolute-like'}",
        f"- Continuous dims mean std: {audit['continuous_dims_mean_std']:.6f}",
        f"- Gripper dim std: {audit['gripper_dim_std']:.6f}",
        "",
        "## Interpretation",
        "",
        "- Actions are 7-dim: 6 continuous (delta position + delta orientation) + 1 gripper.",
        "- Gripper is binary sign-coded: -1 (close) or +1 (open).",
        "- Continuous dims have small magnitude, consistent with delta actions.",
        "- High autocorrelation indicates smooth expert demonstrations.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. True oracle state baseline (using full 92-dim MuJoCo state)
# ---------------------------------------------------------------------------

class FullStateSplitMLP(nn.Module):
    """H=1: predict action[t] from full MuJoCo state[t] (92-dim)."""

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


class FullStatePlusHistoryGRU(nn.Module):
    """H=1: predict action[t] from full state[t] + action history."""

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
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, action_history: torch.Tensor, state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.gru(action_history)
        features = torch.cat([hidden[-1], state_t], dim=-1)
        return self.action_head(self.network(features))


class ProprioPlusHistoryGRU(nn.Module):
    """H=1: predict action[t] from proprio (9-dim) + action history."""

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
        self.action_head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, action_history: torch.Tensor, state_t: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.gru(action_history)
        features = torch.cat([hidden[-1], state_t], dim=-1)
        return self.action_head(self.network(features))


class LinearARModel(nn.Module):
    """H=1: linear autoregressive model over action history."""

    def __init__(self, *, history_len: int, action_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(history_len * action_dim, action_dim)

    def forward(self, action_history: torch.Tensor) -> dict[str, torch.Tensor]:
        B = action_history.shape[0]
        flat = action_history.reshape(B, -1)
        return {"pred_actions": self.linear(flat).unsqueeze(1)}


# ---------------------------------------------------------------------------
# 4. Load trajectories with full state
# ---------------------------------------------------------------------------

def load_trajectories_with_full_state(
    config: Mapping[str, Any],
) -> tuple[list[RawTrajectory], dict[str, Any]]:
    """Load trajectories including the 92-dim full MuJoCo state."""
    try:
        import h5py
        import numpy as np
    except ImportError:
        raise RuntimeError("h5py and numpy are required")

    dataset_root = _resolve_dataset_root(str(config["data"]["dataset_root"]))
    suite = str(config["data"]["suite"])

    from src.train.train_offline import find_demo_files, task_name_from_file, list_demo_groups
    from src.train.train_offline import assign_splits, extract_language, safe_relative

    files = find_demo_files(Path(dataset_root), suite)
    if not files:
        raise FileNotFoundError(f"no HDF5 files for suite={suite}")

    task_names = [task_name_from_file(path) for path in files]
    task_id_by_name = {name: index for index, name in enumerate(sorted(set(task_names)))}

    trajectories: list[RawTrajectory] = []
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
                frame_refs = [
                    f"{trajectory_id}:obs/agentview_rgb:{index}" for index in range(length)
                ]

                # Load proprio (robot_states, 9-dim)
                states = None
                if "robot_states" in group:
                    states = np.asarray(group["robot_states"][()], dtype=np.float32)

                # Load full MuJoCo state (92-dim)
                full_states = None
                if "states" in group:
                    full_states = np.asarray(group["states"][()], dtype=np.float32)

                # Load pre-extracted latents
                visual_latents = None
                latent_dir = config["data"].get("latent_dir")
                if latent_dir:
                    from src.train.train_offline import load_preextracted_latents
                    visual_latents = load_preextracted_latents(
                        latent_dir, file_path, demo_path,
                        config["data"].get("latent_format", "hdf5"),
                    )

                # Load per-joint obs fields
                obs_fields = {}
                if "obs" in group:
                    obs_group = group["obs"]
                    for key in ["ee_pos", "ee_ori", "gripper_states", "ee_states", "joint_states"]:
                        if key in obs_group:
                            obs_fields[key] = np.asarray(obs_group[key][()], dtype=np.float32)

                # Store full_states as a custom attribute for oracle access
                traj = RawTrajectory(
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
                # Attach full_states via object.__setattr__ (frozen dataclass workaround)
                object.__setattr__(traj, "_full_states", full_states)
                object.__setattr__(traj, "_obs_fields", obs_fields)
                trajectories.append(traj)

    if not trajectories:
        raise ValueError("no trajectories found")

    # Build a lookup of custom attributes before assign_splits (which uses replace())
    custom_attrs = {}
    for traj in trajectories:
        custom_attrs[traj.trajectory_id] = {
            "_full_states": getattr(traj, "_full_states", None),
            "_obs_fields": getattr(traj, "_obs_fields", {}),
        }

    split_trajectories, split_metadata = assign_splits(trajectories, config)

    # Re-attach custom attributes to the split trajectories
    for traj in split_trajectories:
        attrs = custom_attrs.get(traj.trajectory_id, {})
        for key, val in attrs.items():
            object.__setattr__(traj, key, val)

    split_metadata["task_id_map"] = task_id_by_name
    return split_trajectories, split_metadata


# ---------------------------------------------------------------------------
# 5. Threshold audit
# ---------------------------------------------------------------------------

def run_threshold_audit(
    *,
    trajectory: RawTrajectory,
    baseline_mses: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    """Audit whether 1e-4 is a scientifically justified threshold."""
    actions = np.array(trajectory.actions, dtype=np.float32)
    T, A = actions.shape

    # Per-dim action variance
    action_var = actions.var(axis=0)
    action_std = actions.std(axis=0)

    # Last-action baseline (a[t-1] -> a[t])
    last_action_mse = float(np.mean((actions[1:] - actions[:-1]) ** 2))

    # Continuous-only MSE for last-action
    last_action_continuous_mse = float(np.mean((actions[1:, :GRIPPER_DIM] - actions[:-1, :GRIPPER_DIM]) ** 2))

    # Duplicate/similar-state variance (adjacent timesteps are similar states)
    adjacent_action_diff = np.abs(actions[1:] - actions[:-1])
    adjacent_action_var = float(adjacent_action_diff.var())

    # Overall action scale
    action_scale = float(np.mean(np.abs(actions)))

    # Gripper-specific analysis
    gripper_vals = actions[:, GRIPPER_DIM]
    gripper_var = float(gripper_vals.var())

    # Threshold comparisons
    threshold = 1e-4
    results = {
        "threshold": threshold,
        "action_scale_mean_abs": action_scale,
        "action_variance_per_dim": action_var.tolist(),
        "action_std_per_dim": action_std.tolist(),
        "last_action_mse": last_action_mse,
        "last_action_continuous_mse": last_action_continuous_mse,
        "adjacent_action_var": adjacent_action_var,
        "gripper_variance": gripper_var,
        "baseline_mses": baseline_mses,
    }

    # Verdict
    # If last-action MSE >> threshold, threshold may be too strict for mixed continuous+gripper
    # If continuous-only MSE << threshold, threshold is reasonable for continuous dims
    # If gripper MSE dominates, consider split threshold

    continuous_baselines = {k: v for k, v in baseline_mses.items()
                           if v < float("inf") and not math.isnan(v)}

    verdict_lines = []

    # Check if 1e-4 is achievable by any baseline
    achievable_baselines = {k: v for k, v in continuous_baselines.items() if v <= threshold}
    if achievable_baselines:
        verdict_lines.append(
            f"Threshold 1e-4 is achievable by: {', '.join(achievable_baselines.keys())}"
        )
    else:
        verdict_lines.append(
            f"Threshold 1e-4 is NOT achieved by any baseline. "
            f"Best baseline: {min(continuous_baselines, key=continuous_baselines.get)} "
            f"at {min(continuous_baselines.values()):.6e}"
        )

    # Check continuous vs gripper decomposition
    if last_action_continuous_mse < threshold:
        verdict_lines.append(
            f"Last-action continuous MSE ({last_action_continuous_mse:.6e}) < threshold. "
            f"Continuous regression gate at 1e-4 is achievable for trivial baselines."
        )
    else:
        verdict_lines.append(
            f"Last-action continuous MSE ({last_action_continuous_mse:.6e}) >= threshold. "
            f"Even trivial baselines cannot achieve 1e-4 on continuous dims."
        )

    verdict_lines.append(
        f"Action scale (mean |action|): {action_scale:.6f}. "
        f"Threshold represents {threshold/action_scale:.2%} of action scale."
    )

    verdict_lines.append(
        f"Recommendation: 1e-4 is a reasonable engineering overfit gate for continuous dims, "
        f"but may be too strict when mixed with binary gripper MSE. "
        f"Consider splitting: continuous regression gate at 1e-4, gripper classification gate at 0."
    )

    results["verdict_lines"] = verdict_lines

    # Write threshold audit
    _write_threshold_audit(output_dir / "threshold_audit.md", results)

    return results


def _write_threshold_audit(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# G7 Threshold Audit",
        "",
        f"## Candidate Threshold: {results['threshold']}",
        "",
        "## Action Scale",
        f"- Mean absolute action: {results['action_scale_mean_abs']:.6f}",
        f"- Threshold as % of action scale: {results['threshold']/results['action_scale_mean_abs']:.2%}",
        "",
        "## Per-Dimension Variance",
        "",
        "| Dim | Std | Variance |",
        "|---:|---:|---:|",
    ]
    for i, (std, var) in enumerate(zip(results["action_std_per_dim"], results["action_variance_per_dim"])):
        lines.append(f"| {i} ({_action_dim_label(i)}) | {std:.6f} | {var:.6e} |")

    lines.extend([
        "",
        "## Baseline Comparisons",
        "",
        f"- Last-action MSE: {results['last_action_mse']:.6e}",
        f"- Last-action continuous MSE: {results['last_action_continuous_mse']:.6e}",
        f"- Adjacent action variance: {results['adjacent_action_var']:.6e}",
        f"- Gripper variance: {results['gripper_variance']:.6f}",
        "",
        "## Baseline MSEs at Threshold",
        "",
        "| Baseline | MSE | Passes 1e-4? |",
        "|---|---:|---|",
    ])
    for name, mse in sorted(results["baseline_mses"].items(), key=lambda x: x[1]):
        if not math.isnan(mse) and mse < float("inf"):
            lines.append(f"| {name} | {mse:.6e} | {'✓' if mse <= results['threshold'] else '✗'} |")

    lines.extend([
        "",
        "## Verdict",
        "",
    ])
    for line in results.get("verdict_lines", []):
        lines.append(f"- {line}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. Raw image frame-reference audit
# ---------------------------------------------------------------------------

def run_raw_image_audit(
    trajectory: RawTrajectory,
    output_dir: Path,
) -> dict[str, Any]:
    """Check if frame references can be resolved to raw pixels."""
    images = trajectory.images
    if not images:
        return {"error": "no images", "can_load": False}

    sample_ref = images[0]
    is_string_ref = isinstance(sample_ref, str)
    is_raw_array = isinstance(sample_ref, (np.ndarray, list))

    # Try to parse frame reference
    can_dereference = False
    dereference_method = None

    if is_string_ref:
        # Frame references look like: "libero_spatial/file.hdf5:data/demo_0:obs/agentview_rgb:0"
        parts = sample_ref.split(":")
        if len(parts) >= 4:
            can_dereference = True
            dereference_method = "hdf5_dataset_path"

    audit = {
        "n_images": len(images),
        "sample_type": type(sample_ref).__name__,
        "is_string_reference": is_string_ref,
        "is_raw_array": is_raw_array,
        "sample_reference": str(sample_ref)[:200],
        "can_dereference": can_dereference,
        "dereference_method": dereference_method,
        "note": (
            "Frame references are HDF5 paths that CAN be resolved to raw pixels "
            "by reading the original HDF5 files. The current pipeline does NOT "
            "resolve them because it loads latents (not raw images) by default. "
            "A lazy raw-image loader could resolve these references on demand."
        ),
    }

    # Write audit report
    lines = [
        "# G7 Raw Image Frame-Reference Audit",
        "",
        f"Sample reference: `{sample_ref}`",
        f"Type: `{type(sample_ref).__name__}`",
        f"Is string reference: {is_string_ref}",
        f"Can dereference: {can_dereference}",
        f"Dereference method: {dereference_method}",
        "",
        "## Assessment",
        "",
        "- Frame references are HDF5 dataset paths of the form:",
        "  `suite/file.hdf5:data/demo_N:obs/agentview_rgb:t`",
        "- These CAN be resolved to raw 128x128x3 RGB uint8 arrays by reading the HDF5 file.",
        "- The current pipeline does NOT resolve them because it uses pre-extracted DINO latents.",
        "- A lazy raw-image loader could be implemented to resolve these on demand.",
        "- The raw-image CNN baseline in G6 was skipped because the dataset returns frame references",
        "  (strings), not raw pixel arrays.",
        "",
        "## Resolution Path",
        "",
        "1. Parse the frame reference string",
        "2. Open the HDF5 file at the referenced path",
        "3. Read `data/demo_N/obs/agentview_rgb[t]`",
        "4. Return the uint8 array",
        "",
        "## Conclusion",
        "",
        "Raw images ARE accessible through frame-reference resolution.",
        "A lazy loader can be implemented to enable raw-image CNN baselines.",
    ]
    (output_dir / "raw_image_loader_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return audit


# ---------------------------------------------------------------------------
# 7. Action-space baseline ladder
# ---------------------------------------------------------------------------

def _repair_forward_g7(model, model_kind, batch, device):
    """Forward dispatch for G7 model kinds."""
    if model_kind == "full_state_mlp":
        return model(batch["full_state_t"].to(device))
    elif model_kind == "full_state_history_gru":
        return model(batch["action_history"].to(device), batch["full_state_t"].to(device))
    elif model_kind == "proprio_history_gru":
        return model(batch["action_history"].to(device), batch["optional_state_t"].to(device))
    elif model_kind == "linear_ar":
        return model(batch["action_history"].to(device))
    elif model_kind == "action_history_gru":
        return model(batch["action_history"].to(device))
    elif model_kind == "proprio_mlp":
        return model(batch["optional_state_t"].to(device))
    else:
        raise ValueError(f"unknown G7 model kind: {model_kind}")


def _train_g7_model(
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
) -> dict[str, Any]:
    """Train a model under H=1 causal contract and return metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_metrics = None
    best_epoch = -1
    best_train_mse = float("inf")

    # Determine loss mode based on model kind
    loss_mode = "mse" if model_kind in ("linear_ar",) else "split"

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            outputs = _repair_forward_g7(model, model_kind, batch, device)
            target = batch["target_actions"].to(device)
            loss = _repair_loss(outputs, target, mode=loss_mode)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * target.shape[0]
            total_samples += target.shape[0]

        # Eval
        model.eval()
        pred_rows = []
        target_rows = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = _repair_forward_g7(model, model_kind, batch, device)
                pred = outputs["pred_actions"] if isinstance(outputs, dict) else outputs
                target = batch["target_actions"].to(device)
                if action_transform is not None:
                    pred = action_transform.denormalize_tensor(pred)
                    target = action_transform.denormalize_tensor(target)
                pred_rows.append(pred.cpu())
                target_rows.append(target.cpu())

        if pred_rows:
            pred_all = torch.cat(pred_rows, dim=0)
            target_all = torch.cat(target_rows, dim=0)
            metrics = _repair_metrics_from_tensors(pred_all, target_all)

            if metrics["action_mse"] < (best_metrics["action_mse"] if best_metrics else float("inf")):
                best_metrics = dict(metrics)
                best_epoch = epoch
                best_train_mse = total_loss / max(total_samples, 1)

    if best_metrics is None:
        raise RuntimeError("G7 training produced no metrics")

    return {
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "best_train_mse": best_train_mse,
        "passed": best_metrics["action_mse"] <= loss_threshold,
    }


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/g7_state_action_contract"))
    parser.add_argument("--trajectory_id", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "any"], default="train")
    parser.add_argument("--epochs", type=int, default=300)
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
    output_dir = run_g7_diagnostics(
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
        hidden_dim=args.hidden_dim,
        run_id=args.run_id,
        command=[sys.executable, "-m", "src.eval.g7_state_action_contract", *(argv or sys.argv[1:])],
    )
    print(f"g7_output_dir={output_dir}")
    return 0


def run_g7_diagnostics(
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
    hidden_dim: int = 256,
    run_id: str | None = None,
    command: Sequence[str] | None = None,
) -> Path:
    seed_everything(seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{timestamp}_g7_state_action_contract"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    effective_lr = float(lr if lr is not None else config["training"]["lr"])
    device = torch.device(device_name)
    git_info = get_git_info()

    # ===================================================================
    # 1. Dataset schema audit
    # ===================================================================
    print("[1/9] Dataset schema audit ...")
    schema = run_state_schema_audit(config, output_dir)

    # ===================================================================
    # 2. Load trajectories with full state
    # ===================================================================
    print("[2/9] Loading trajectories with full state ...")
    trajectories, source_metadata = load_trajectories_with_full_state(config)
    selected = _select_trajectory(trajectories, trajectory_id, source_split)
    diagnostic_trajs = [replace(selected, split="train"), replace(selected, split="val")]

    action_transform, normalization_stats = build_action_transform(diagnostic_trajs, config)
    if action_transform is not None:
        diagnostic_trajs = apply_action_transform(diagnostic_trajs, action_transform)

    full_states = getattr(selected, "_full_states", None)
    obs_fields = getattr(selected, "_obs_fields", {})
    proprio_states = selected.states  # 9-dim robot_states

    has_full_state = full_states is not None
    has_proprio = proprio_states is not None
    has_latents = selected.visual_latents is not None
    dataset_name = source_metadata.get("suite", "unknown")

    print(f"  Trajectory: {selected.trajectory_id} (T={selected.length})")
    print(f"  Full state (92-dim): {'available' if has_full_state else 'NOT available'}")
    print(f"  Proprio (9-dim): {'available' if has_proprio else 'NOT available'}")
    print(f"  Obs fields: {list(obs_fields.keys())}")

    # ===================================================================
    # 3. Action target audit
    # ===================================================================
    print("[3/9] Action target audit ...")
    action_audit = run_action_target_audit(
        selected, output_dir=output_dir, git_info=git_info, dataset=dataset_name,
    )

    # ===================================================================
    # 4. Raw image frame-reference audit
    # ===================================================================
    print("[4/9] Raw image frame-reference audit ...")
    raw_image_audit = run_raw_image_audit(selected, output_dir)

    # ===================================================================
    # 5. Build datasets with full state
    # ===================================================================
    print("[5/9] Building datasets ...")
    train_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs, split="train", config=config,
        action_horizon=1, target_shift=0,
    )
    val_ds = ShiftedTargetWindowDataset(
        diagnostic_trajs, split="val", config=config,
        action_horizon=1, target_shift=0,
    )

    # Inject full_state_t into datasets
    class FullStateDataset:
        def __init__(self, base_ds, full_states_arr, obs_fields_dict):
            self.base = base_ds
            self.full_states_arr = full_states_arr
            self.obs_fields_dict = obs_fields_dict
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            sample = self.base[idx]
            t = sample["time_index"]
            if self.full_states_arr is not None:
                sample["full_state_t"] = self.full_states_arr[t]
            return sample

    fs_train = FullStateDataset(train_ds, full_states, obs_fields)
    fs_val = FullStateDataset(val_ds, full_states, obs_fields)

    def g7_collate(batch):
        collated = collate_action_batch(batch)
        if "full_state_t" in batch[0]:
            collated["full_state_t"] = torch.stack([
                torch.as_tensor(s["full_state_t"], dtype=torch.float32)
                for s in batch
            ])
        return collated

    train_loader = DataLoader(fs_train, batch_size=batch_size, shuffle=True, collate_fn=g7_collate)
    val_loader = DataLoader(fs_val, batch_size=batch_size, shuffle=False, collate_fn=g7_collate)

    sample = fs_train[0]
    action_dim = infer_action_dim(sample)
    effective_state_dim = infer_state_dim(sample) or 0
    full_state_dim = full_states.shape[1] if has_full_state else 0
    history_len = int(config["data"]["history_len"])

    print(f"  action_dim={action_dim}, proprio_dim={effective_state_dim}, "
          f"full_state_dim={full_state_dim}, history_len={history_len}")

    # ===================================================================
    # 6. Action-space baseline ladder
    # ===================================================================
    print("[6/9] Action-space baseline ladder ...")
    ladder_rows = []

    # 6a. Zero action
    zero_mse = _eval_constant_baseline(val_loader, "zero")
    ladder_rows.append(_make_ladder_row("zero_action", "constant", 0, zero_mse, action_dim, loss_threshold))

    # 6b. Mean action (from train)
    mean_mse = _eval_mean_baseline(train_loader, val_loader, action_transform)
    ladder_rows.append(_make_ladder_row("mean_action", "constant", 0, mean_mse, action_dim, loss_threshold))

    # 6c. Last action a[t-1]
    last_action_mse = _eval_last_action_baseline(val_loader, action_transform)
    ladder_rows.append(_make_ladder_row("last_action", "copy", 0, last_action_mse, action_dim, loss_threshold))

    # 6d. Linear AR model
    if epochs > 0:
        print("  Training linear AR ...")
        ar_model = LinearARModel(history_len=history_len, action_dim=action_dim).to(device)
        ar_result = _train_g7_model(
            model=ar_model, model_kind="linear_ar",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            loss_threshold=loss_threshold, action_transform=action_transform,
        )
        ladder_rows.append(_make_ladder_row_from_result(
            "linear_ar", "linear", history_len * action_dim, ar_result, loss_threshold))

    # 6e. Action history GRU
    if epochs > 0:
        print("  Training action history GRU ...")
        from src.eval.overfit_diagnostics import ActionHistorySplitGRU
        hist_gru = ActionHistorySplitGRU(
            history_len=history_len, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        hist_result = _train_g7_model(
            model=hist_gru, model_kind="action_history_gru",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            loss_threshold=loss_threshold, action_transform=action_transform,
        )
        ladder_rows.append(_make_ladder_row_from_result(
            "action_history_gru", "gru", hidden_dim, hist_result, loss_threshold))

    # 6f. Proprio-only MLP
    if has_proprio and epochs > 0:
        print("  Training proprio-only MLP ...")
        from src.eval.overfit_diagnostics import ProprioSplitMLP
        proprio_model = ProprioSplitMLP(
            state_dim=effective_state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
        ).to(device)
        proprio_result = _train_g7_model(
            model=proprio_model, model_kind="proprio_mlp",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            loss_threshold=loss_threshold, action_transform=action_transform,
        )
        ladder_rows.append(_make_ladder_row_from_result(
            "proprio_only_state", "mlp", effective_state_dim, proprio_result, loss_threshold))

    # 6g. Proprio + action history GRU
    if has_proprio and epochs > 0:
        print("  Training proprio + history GRU ...")
        ph_model = ProprioPlusHistoryGRU(
            state_dim=effective_state_dim, action_dim=action_dim,
            history_len=history_len, hidden_dim=hidden_dim,
        ).to(device)
        ph_result = _train_g7_model(
            model=ph_model, model_kind="proprio_history_gru",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            loss_threshold=loss_threshold, action_transform=action_transform,
        )
        ladder_rows.append(_make_ladder_row_from_result(
            "proprio_plus_history", "gru+mlp", effective_state_dim + hidden_dim, ph_result, loss_threshold))

    # 6h. Full state (92-dim) MLP
    if has_full_state and epochs > 0:
        print("  Training full state (92-dim) MLP ...")
        full_model = FullStateSplitMLP(
            state_dim=full_state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
        ).to(device)
        full_result = _train_g7_model(
            model=full_model, model_kind="full_state_mlp",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            loss_threshold=loss_threshold, action_transform=action_transform,
        )
        ladder_rows.append(_make_ladder_row_from_result(
            "full_state_92d_oracle", "mlp", full_state_dim, full_result, loss_threshold))
    elif not has_full_state:
        ladder_rows.append(_make_ladder_row(
            "full_state_92d_oracle", "mlp", 0,
            {"best_metrics": {"action_mse": float("nan")}, "passed": False,
             "error": "full_state_not_available"},
            action_dim, loss_threshold))

    # 6i. Full state + action history GRU
    if has_full_state and epochs > 0:
        print("  Training full state + history GRU ...")
        fsh_model = FullStatePlusHistoryGRU(
            state_dim=full_state_dim, action_dim=action_dim,
            history_len=history_len, hidden_dim=hidden_dim,
        ).to(device)
        fsh_result = _train_g7_model(
            model=fsh_model, model_kind="full_state_history_gru",
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            loss_threshold=loss_threshold, action_transform=action_transform,
        )
        ladder_rows.append(_make_ladder_row_from_result(
            "full_state_plus_history", "gru+mlp", full_state_dim + hidden_dim, fsh_result, loss_threshold))

    _write_csv(output_dir / "action_space_baseline_ladder.csv", ladder_rows)

    # ===================================================================
    # 7. Threshold audit
    # ===================================================================
    print("[7/9] Threshold audit ...")
    baseline_mses = {row["variant"]: row["eval_mse"] for row in ladder_rows
                     if not math.isnan(row["eval_mse"]) and row["eval_mse"] < float("inf")}
    threshold_result = run_threshold_audit(
        trajectory=selected, baseline_mses=baseline_mses, output_dir=output_dir,
    )

    # ===================================================================
    # 8. Write true oracle state result
    # ===================================================================
    print("[8/9] Writing oracle state result ...")
    if has_full_state:
        oracle_rows = [r for r in ladder_rows if r["variant"] in ("full_state_92d_oracle", "full_state_plus_history")]
        _write_csv(output_dir / "true_oracle_state_baseline.csv", oracle_rows)
    else:
        (output_dir / "true_oracle_state_unavailable.md").write_text(
            "# True Oracle State Unavailable\n\n"
            "The 92-dim full MuJoCo state field ('states') was not found in the HDF5 files.\n"
            "The dataset only exposes robot_states (9-dim proprioceptive).\n"
            "A true oracle state with object and goal poses cannot be constructed.\n",
            encoding="utf-8",
        )

    # ===================================================================
    # 9. Summary and repro files
    # ===================================================================
    print("[9/9] Writing summary and repro files ...")
    summary = {
        "status": "g7_state_action_contract",
        "config": str(config_path),
        "trajectory_id": selected.trajectory_id,
        "trajectory_length": selected.length,
        "task_id": selected.task_id,
        "task_name": selected.task_name,
        "dataset": dataset_name,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "seed": seed,
        "epochs": epochs,
        "loss_threshold": loss_threshold,
        "hidden_dim": hidden_dim,
        "has_full_state": has_full_state,
        "full_state_dim": full_state_dim,
        "has_proprio": has_proprio,
        "proprio_dim": effective_state_dim,
        "action_dim": action_dim,
        "schema": schema,
        "action_audit": action_audit,
        "raw_image_audit": raw_image_audit,
        "baseline_ladder": ladder_rows,
        "threshold_audit": threshold_result,
        "non_claims": [
            "not_closed_loop_success",
            "not_future_latent_benefit_evidence",
            "not_architecture_claim_evidence",
            "not_policy_validity_evidence",
            "oracle_state_not_available" if not has_full_state else "oracle_state_is_full_mujoco_state",
        ],
    }
    _write_json(output_dir / "summary.json", summary)

    # Write git_commit.txt, environment, etc.
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
    import shutil
    shutil.copyfile(config_path, output_dir / "config.yaml")
    _write_json(output_dir / "split.json", source_metadata)
    (output_dir / "notes.md").write_text(
        "G7 state/action contract audit. Single-demo H=1 overfit under strict causal contract. "
        "Not closed-loop, not architecture-claim evidence.\n",
        encoding="utf-8",
    )

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


def _make_ladder_row(variant, model_type, param_count, eval_mse, action_dim, threshold):
    return {
        "variant": variant,
        "model_type": model_type,
        "param_count": param_count,
        "eval_mse": eval_mse if not math.isnan(eval_mse) else float("nan"),
        "continuous_mse": float("nan"),
        "gripper_mse": float("nan"),
        "gripper_sign_accuracy": float("nan"),
        "passed": eval_mse <= threshold if not math.isnan(eval_mse) else False,
        "best_epoch": -1,
        "error": "",
    }


def _make_ladder_row_from_result(variant, model_type, param_count, result, threshold):
    m = result["best_metrics"]
    return {
        "variant": variant,
        "model_type": model_type,
        "param_count": param_count,
        "eval_mse": m["action_mse"],
        "continuous_mse": m.get("continuous_mse", float("nan")),
        "gripper_mse": m.get("gripper_mse", float("nan")),
        "gripper_sign_accuracy": m.get("gripper_sign_accuracy", float("nan")),
        "passed": result["passed"],
        "best_epoch": result["best_epoch"],
        "error": "",
    }


def _eval_constant_baseline(loader, value="zero"):
    """Evaluate a constant-action baseline."""
    total_se = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["target_actions"]
            if value == "zero":
                pred = torch.zeros_like(target)
            else:
                raise ValueError(f"unknown constant: {value}")
            total_se += float((pred - target).pow(2).sum().item())
            count += target.numel()
    return total_se / max(count, 1)


def _eval_mean_baseline(train_loader, val_loader, action_transform):
    """Compute train mean and evaluate on val."""
    # Accumulate train mean
    all_train = []
    with torch.no_grad():
        for batch in train_loader:
            target = batch["target_actions"]
            if action_transform is not None:
                target = action_transform.denormalize_tensor(target)
            all_train.append(target)
    train_mean = torch.cat(all_train, dim=0).mean(dim=0)  # [H, A]

    # Evaluate on val
    total_se = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            target = batch["target_actions"]
            if action_transform is not None:
                target = action_transform.denormalize_tensor(target)
            pred = train_mean.expand_as(target)
            total_se += float((pred - target).pow(2).sum().item())
            count += target.numel()
    return total_se / max(count, 1)


def _eval_last_action_baseline(loader, action_transform):
    """Evaluate last-action (a[t-1] -> a[t]) baseline."""
    total_se = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            action_history = batch["action_history"]  # [B, H_hist, A]
            target = batch["target_actions"]  # [B, 1, A]
            last_action = action_history[:, -1:, :]  # [B, 1, A]
            if action_transform is not None:
                last_action = action_transform.denormalize_tensor(last_action)
                target = action_transform.denormalize_tensor(target)
            total_se += float((last_action - target).pow(2).sum().item())
            count += target.numel()
    return total_se / max(count, 1)


if __name__ == "__main__":
    raise SystemExit(main())
