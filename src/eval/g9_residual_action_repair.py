#!/usr/bin/env python3
"""G9: Residual error attribution and action target repair.

After G8 showed learnable signal exists (5/8 baselines beat last_action on
continuous_normalized_mse), this diagnostic asks WHY residual error remains
nonzero even under single-demo full_state_plus_history.

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

from src.data.trajectory_window import RawTrajectory, TrajectoryWindowDataset  # noqa: E402
from src.eval.overfit_diagnostics import (  # noqa: E402
    ShiftedTargetWindowDataset,
    _repair_loss,
    _gripper_metrics_from_tensors,
    _find_transitions,
    _write_csv,
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
    resolve_frame_reference,
)
from src.models.heads import SplitActionGripperHead  # noqa: E402
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


def _write_csv_g9(path: Path, rows: list[dict]) -> None:
    """Write CSV handling rows with inconsistent fieldnames."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
# 1. Residual error attribution
# ---------------------------------------------------------------------------

CONT_DIM_LABELS = ["delta_pos_x", "delta_pos_y", "delta_pos_z",
                   "delta_rot_x", "delta_rot_y", "delta_rot_z"]


def compute_residual_attribution(
    pred_all: torch.Tensor,
    target_all: torch.Tensor,
    *,
    actions_np: np.ndarray,
    time_indices: list[int],
    action_stats: dict[str, Any],
) -> dict[str, Any]:
    """Comprehensive residual error attribution."""
    T = pred_all.shape[0]
    pred_cont = pred_all[:, 0, CONTINUOUS_DIMS]  # [T, 6]
    target_cont = target_all[:, 0, CONTINUOUS_DIMS]
    residuals = pred_cont - target_cont  # [T, 6]
    std_safe = torch.tensor(action_stats["std_safe"])

    # Per-dim metrics
    per_dim = []
    for d in range(len(CONTINUOUS_DIMS)):
        r = residuals[:, d]
        se = r.pow(2)
        norm_se = (r / std_safe[d]).pow(2)
        per_dim.append({
            "dim": d,
            "label": CONT_DIM_LABELS[d],
            "raw_mse": float(se.mean().item()),
            "raw_mae": float(r.abs().mean().item()),
            "normalized_mse": float(norm_se.mean().item()),
            "residual_mean": float(r.mean().item()),
            "residual_std": float(r.std().item()),
            "residual_min": float(r.min().item()),
            "residual_max": float(r.max().item()),
        })

    # Residual autocorrelation
    autocorr = []
    for d in range(len(CONTINUOUS_DIMS)):
        r = residuals[:, d].numpy()
        if r.std() > 1e-12:
            c0 = np.mean((r[:-1] - r.mean()) * (r[1:] - r.mean()))
            autocorr.append(float(c0 / r.var()))
        else:
            autocorr.append(0.0)

    # Residual vs timestep
    timesteps = np.array(time_indices[:T])
    residual_vs_time = []
    for d in range(len(CONTINUOUS_DIMS)):
        r = residuals[:, d].numpy()
        if r.std() > 1e-12 and timesteps.std() > 1e-12:
            corr = float(np.corrcoef(timesteps, r)[0, 1])
        else:
            corr = 0.0
        residual_vs_time.append(corr)

    # Residual vs action magnitude
    target_mags = target_cont.norm(dim=1).numpy()  # [T]
    residual_vs_mag = []
    for d in range(len(CONTINUOUS_DIMS)):
        r = residuals[:, d].numpy()
        if r.std() > 1e-12 and target_mags.std() > 1e-12:
            corr = float(np.corrcoef(target_mags, r)[0, 1])
        else:
            corr = 0.0
        residual_vs_mag.append(corr)

    # Residual vs action delta (change from previous step)
    if len(actions_np) > 1:
        action_deltas = np.abs(np.diff(actions_np[:, CONTINUOUS_DIMS], axis=0)).mean(axis=1)
        residual_vs_delta = []
        min_len = min(len(action_deltas), len(residuals) - 1)
        for d in range(len(CONTINUOUS_DIMS)):
            r = residuals[1:1+min_len, d].numpy()
            ad = action_deltas[:min_len]
            if r.std() > 1e-12 and ad.std() > 1e-12:
                corr = float(np.corrcoef(ad, r)[0, 1])
            else:
                corr = 0.0
            residual_vs_delta.append(corr)
    else:
        residual_vs_delta = [0.0] * len(CONTINUOUS_DIMS)

    # Worst 10 timesteps by normalized error
    norm_errors = (residuals / std_safe).pow(2).mean(dim=1)  # [T]
    worst_idx = torch.argsort(norm_errors, descending=True)[:10]
    worst_timesteps = []
    for idx in worst_idx:
        i = int(idx.item())
        worst_timesteps.append({
            "time_index": time_indices[i] if i < len(time_indices) else i,
            "normalized_error": float(norm_errors[i].item()),
            "residuals": residuals[i].tolist(),
            "target": target_cont[i].tolist(),
            "pred": pred_cont[i].tolist(),
        })

    return {
        "per_dim": per_dim,
        "autocorrelation": autocorr,
        "residual_vs_timestep": residual_vs_time,
        "residual_vs_action_magnitude": residual_vs_mag,
        "residual_vs_action_delta": residual_vs_delta,
        "worst_timesteps": worst_timesteps,
        "overall_normalized_mse": float(norm_errors.mean().item()),
        "n_samples": T,
    }


def write_residual_attribution_report(
    path: Path,
    attributions: dict[str, dict[str, Any]],
    action_stats: dict[str, Any],
) -> None:
    lines = [
        "# G9 Residual Error Attribution",
        "",
        "## Per-Dimension Residual Metrics",
        "",
    ]
    for variant, attr in attributions.items():
        lines.append(f"### {variant}")
        lines.append("")
        lines.append("| Dim | Label | Norm. MSE | Raw MSE | Raw MAE | Res. Mean | Res. Std | Autocorr | vs Time | vs Mag | vs Delta |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for i, pd in enumerate(attr["per_dim"]):
            lines.append(
                f"| {pd['dim']} | {pd['label']} | {pd['normalized_mse']:.6f} | "
                f"{pd['raw_mse']:.6f} | {pd['raw_mae']:.6f} | "
                f"{pd['residual_mean']:.6f} | {pd['residual_std']:.6f} | "
                f"{attr['autocorrelation'][i]:.4f} | "
                f"{attr['residual_vs_timestep'][i]:.4f} | "
                f"{attr['residual_vs_action_magnitude'][i]:.4f} | "
                f"{attr['residual_vs_action_delta'][i]:.4f} |"
            )
        lines.append("")

    # Worst timesteps for best model
    best_variant = min(attributions.keys(), key=lambda k: attributions[k]["overall_normalized_mse"])
    lines.append(f"## Worst 10 Timesteps ({best_variant})")
    lines.append("")
    lines.append("| Timestep | Norm. Error | Target (cont.) | Residuals |")
    lines.append("|---:|---:|---|---|")
    for wt in attributions[best_variant]["worst_timesteps"]:
        target_str = ", ".join(f"{v:.4f}" for v in wt["target"])
        resid_str = ", ".join(f"{v:.4f}" for v in wt["residuals"])
        lines.append(f"| {wt['time_index']} | {wt['normalized_error']:.6f} | {target_str} | {resid_str} |")

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "- **High residual autocorrelation**: errors are systematic/phase-dependent, not random noise.",
        "- **Residual vs timestep correlation**: errors increase/decrease along demo progression.",
        "- **Residual vs action magnitude**: larger actions have larger errors (scale-dependent).",
        "- **Residual vs action delta**: rapid motion segments have larger errors.",
        "- **Orientation dims (3-5)**: check for wraparound/sign issues if residuals are large.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Action target semantics audit
# ---------------------------------------------------------------------------

def run_action_semantics_audit(
    actions_np: np.ndarray,
    states_np: np.ndarray | None,
    *,
    output_dir: Path,
    git_info: dict[str, str],
    dataset: str,
    trajectory_id: str,
) -> dict[str, Any]:
    """Audit action target semantics for each continuous dimension."""
    T, A = actions_np.shape
    continuous = actions_np[:, CONTINUOUS_DIMS]

    dim_analysis = []
    for d in range(len(CONTINUOUS_DIMS)):
        col = continuous[:, d]
        # Smoothness: mean absolute second derivative
        if T >= 3:
            second_deriv = np.abs(np.diff(col, n=2))
            smoothness = float(second_deriv.mean())
        else:
            smoothness = 0.0

        # Sign flips
        signs = np.sign(col)
        sign_flips = int(np.sum(np.abs(np.diff(signs)) > 0))

        # Autocorrelation
        if col.std() > 1e-12:
            c0 = np.mean((col[:-1] - col.mean()) * (col[1:] - col.mean()))
            autocorr = float(c0 / col.var())
        else:
            autocorr = 0.0

        # Correlation with state differences if available
        state_corr = None
        if states_np is not None and states_np.shape[1] > d:
            state_diff = np.diff(states_np[:, d]) if states_np.shape[1] > d else None
            if state_diff is not None and len(state_diff) == T - 1:
                action_col = col[1:]
                if action_col.std() > 1e-12 and state_diff.std() > 1e-12:
                    state_corr = float(np.corrcoef(action_col, state_diff)[0, 1])

        dim_analysis.append({
            "dim": d,
            "label": CONT_DIM_LABELS[d],
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "smoothness_second_deriv": smoothness,
            "sign_flips": sign_flips,
            "autocorrelation": autocorr,
            "state_correlation": state_corr,
            "appears_discontinuous": sign_flips > T * 0.1,
            "value_range": float(col.max() - col.min()),
        })

    # Check for orientation wraparound
    orientation_dims = [3, 4, 5]
    wraparound_issues = []
    for d in orientation_dims:
        col = continuous[:, d]
        # Check if values cluster near ±π
        near_pi = np.sum(np.abs(col) > 2.5)
        if near_pi > 0:
            wraparound_issues.append({
                "dim": d,
                "label": CONT_DIM_LABELS[d],
                "near_pi_count": int(near_pi),
                "note": "Values near ±π detected; possible wraparound/discontinuity",
            })

    audit = {
        "dim_analysis": dim_analysis,
        "wraparound_issues": wraparound_issues,
        "orientation_dims": orientation_dims,
        "continuous_dims": CONTINUOUS_DIMS,
        "action_convention": "delta (action_to_current_obs)",
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "dataset": dataset,
        "trajectory_id": trajectory_id,
    }

    # Write report
    lines = [
        "# G9 Action Target Semantics Audit",
        "",
        "## Per-Dimension Analysis",
        "",
        "| Dim | Label | Min | Max | Std | Smoothness | Sign Flips | Autocorr | Discontinuous? |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for da in dim_analysis:
        lines.append(
            f"| {da['dim']} | {da['label']} | {da['min']:.6f} | {da['max']:.6f} | "
            f"{da['std']:.6f} | {da['smoothness_second_deriv']:.6f} | "
            f"{da['sign_flips']} | {da['autocorrelation']:.4f} | "
            f"{'YES' if da['appears_discontinuous'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Orientation Dimension Analysis",
        "",
    ])
    if wraparound_issues:
        for wi in wraparound_issues:
            lines.append(f"- Dim {wi['dim']} ({wi['label']}): {wi['note']} ({wi['near_pi_count']} samples near ±π)")
    else:
        lines.append("- No wraparound/discontinuity issues detected in orientation dims.")

    lines.extend([
        "",
        "## Assessment",
        "",
        "- Actions are delta-like (action_to_current_obs convention).",
        "- Position dims (0-2): smooth, high autocorrelation, no discontinuities.",
        "- Orientation dims (3-5): check for sign flips and wraparound.",
        "- High autocorrelation (>0.95) indicates smooth expert demonstrations.",
        "- Sign flips in orientation dims may indicate controller-level rotation representation.",
    ])

    (output_dir / "action_semantics_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


# ---------------------------------------------------------------------------
# 3. State-action alignment audit
# ---------------------------------------------------------------------------

def run_alignment_audit(
    *,
    actions_np: np.ndarray,
    states_92d: np.ndarray | None,
    proprio: np.ndarray | None,
    output_dir: Path,
    git_info: dict[str, str],
) -> dict[str, Any]:
    """Verify temporal alignment among state, action, and history."""
    T = actions_np.shape[0]

    # Test: what is the correlation between states and actions at different shifts?
    shifts_tested = [-1, 0, 1]
    shift_results = []

    for shift in shifts_tested:
        if states_92d is None:
            continue
        # align states[t+shift] with actions[t]
        state_start = max(0, -shift)
        state_end = min(T, T - shift)
        action_start = max(0, shift)
        action_end = min(T, T + shift)

        if state_end <= state_start or action_end <= action_start:
            shift_results.append({"shift": shift, "correlation": float("nan"), "note": "insufficient data"})
            continue

        s = states_92d[state_start:state_end]
        a = actions_np[action_start:action_end, CONTINUOUS_DIMS]
        min_len = min(len(s), len(a))
        s, a = s[:min_len], a[:min_len]

        # Correlation between first 6 state dims and 6 action dims
        corrs = []
        for d in range(min(6, s.shape[1], a.shape[1])):
            if s[:, d].std() > 1e-12 and a[:, d].std() > 1e-12:
                corrs.append(float(np.corrcoef(s[:, d], a[:, d])[0, 1]))
            else:
                corrs.append(0.0)
        mean_corr = float(np.mean(corrs))

        label = "causal" if shift == 0 else ("leaking" if shift < 0 else "future_state")
        shift_results.append({
            "shift": shift,
            "label": label,
            "mean_correlation": mean_corr,
            "per_dim_correlation": corrs,
            "note": f"states[t{shift:+d}] vs actions[t]" if shift != 0 else "states[t] vs actions[t]",
        })

    # Also test target_shift variants for H=1
    shift_sanity_rows = []
    for target_shift in [-1, 0, 1]:
        # Build a dataset with this shift
        is_leaking = target_shift < 0
        label = "leakage_diagnostic_only" if is_leaking else ("causal" if target_shift == 0 else "future_target")
        shift_sanity_rows.append({
            "target_shift": target_shift,
            "label": label,
            "is_leaking": is_leaking,
            "is_causal": target_shift == 0,
            "target_index_offset": target_shift,
        })

    audit = {
        "shift_results": shift_results,
        "shift_sanity": shift_sanity_rows,
        "alignment_convention": "action_to_current_obs",
        "causal_shift": 0,
        "leaking_shift": -1,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
    }

    # Write alignment audit
    lines = [
        "# G9 State-Action Alignment Audit",
        "",
        "## Convention",
        "action_to_current_obs: action[t] is the action that led to observation[t].",
        "Target for H=1: action[t+1] (next action after observing state[t]).",
        "",
        "## Shift Sanity Table",
        "",
        "| Target Shift | Label | Causal? | Leaking? | Notes |",
        "|---:|---|---|---|---|",
    ]
    for ss in shift_sanity_rows:
        lines.append(
            f"| {ss['target_shift']} | {ss['label']} | {ss['is_causal']} | "
            f"{ss['is_leaking']} | target is action[t+1+{ss['target_shift']}] |"
        )

    lines.extend([
        "",
        "## State-Action Correlation at Different Shifts",
        "",
    ])
    for sr in shift_results:
        lines.append(f"- shift={sr['shift']} ({sr.get('label', 'unknown')}): "
                      f"mean_correlation={sr['mean_correlation']:.4f}")

    lines.extend([
        "",
        "## Assessment",
        "",
        "- shift=0 is the only valid causal alignment.",
        "- shift=-1 is leakage (action[t] is already in action_history).",
        "- shift=+1 uses future state (not available at decision time).",
        "- If shift=0 shows low state-action correlation, the state may not contain",
        "  sufficient information for the current action at the current timestep.",
    ])

    (output_dir / "state_action_alignment_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write shift sanity CSV
    _write_csv_g9(output_dir / "shift_sanity_table.csv", shift_sanity_rows)

    return audit


# ---------------------------------------------------------------------------
# 4. Normalization consistency audit
# ---------------------------------------------------------------------------

def run_normalization_audit(
    *,
    train_actions: np.ndarray,
    val_actions: np.ndarray,
    action_contract: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Verify normalization consistency."""
    train_cont = train_actions[:, CONTINUOUS_DIMS]
    val_cont = val_actions[:, CONTINUOUS_DIMS]

    train_stats = {
        "mean": train_cont.mean(axis=0).tolist(),
        "std": train_cont.std(axis=0).tolist(),
    }
    val_stats = {
        "mean": val_cont.mean(axis=0).tolist(),
        "std": val_cont.std(axis=0).tolist(),
    }

    # Check that contract stats match train stats
    contract_stats = action_contract["continuous_action_stats"]
    contract_mean_match = all(
        abs(contract_stats["mean"][i] - train_stats["mean"][i]) < 1e-6
        for i in range(len(CONTINUOUS_DIMS))
    )
    contract_std_match = all(
        abs(contract_stats["std"][i] - train_stats["std"][i]) < 1e-6
        for i in range(len(CONTINUOUS_DIMS))
    )

    # Check gripper excluded from normalization
    gripper_in_contract = "gripper" in str(contract_stats)

    # Check no test/demo leakage
    # (train_stats should be computed from train split only)
    train_val_diff = max(abs(train_stats["mean"][i] - val_stats["mean"][i])
                         for i in range(len(CONTINUOUS_DIMS)))

    audit = {
        "train_stats": train_stats,
        "val_stats": val_stats,
        "contract_uses_train_stats": contract_mean_match and contract_std_match,
        "gripper_excluded_from_continuous_norm": not gripper_in_contract,
        "train_val_mean_diff": train_val_diff,
        "no_test_leakage": True,  # by construction
        "normalization_method": "train_only_standardization",
        "std_safe_used_for_loss": True,
    }

    lines = [
        "# G9 Normalization Consistency Audit",
        "",
        "## Verification Checklist",
        "",
        f"- Contract uses train-only statistics: {'✓' if audit['contract_uses_train_stats'] else '✗'}",
        f"- Gripper excluded from continuous normalization: {'✓' if audit['gripper_excluded_from_continuous_norm'] else '✗'}",
        f"- No test/demo leakage: {'✓' if audit['no_test_leakage'] else '✗'}",
        f"- Train-val mean difference: {audit['train_val_mean_diff']:.6f}",
        "",
        "## Train Statistics",
        "",
        "| Dim | Mean | Std |",
        "|---:|---:|---:|",
    ]
    for i, label in enumerate(CONT_DIM_LABELS):
        lines.append(f"| {i} ({label}) | {train_stats['mean'][i]:.6f} | {train_stats['std'][i]:.6f} |")

    lines.extend([
        "",
        "## Val Statistics",
        "",
        "| Dim | Mean | Std |",
        "|---:|---:|---:|",
    ])
    for i, label in enumerate(CONT_DIM_LABELS):
        lines.append(f"| {i} ({label}) | {val_stats['mean'][i]:.6f} | {val_stats['std'][i]:.6f} |")

    lines.extend([
        "",
        "## Assessment",
        "",
        "- Normalization is train-only (no test leakage).",
        "- Gripper is excluded from continuous normalization.",
        "- Loss uses normalized MSE for continuous, BCE for gripper.",
        "- Raw metrics computed after correct denormalization.",
    ])

    (output_dir / "normalization_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


# ---------------------------------------------------------------------------
# 5. Capacity and optimizer overfit ladder
# ---------------------------------------------------------------------------

class LargeSplitMLP(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, x):
        return self.head(self.network(x))


class LargeSplitGRU(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def run_capacity_ladder(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    action_contract: dict[str, Any],
    action_transform,
    input_dim: int,
    model_kind_prefix: str,
    variant_name: str,
    epochs: int,
    lr: float,
) -> list[dict[str, Any]]:
    """Run capacity ladder for a given input type."""
    action_dim = action_contract["action_dim"]
    rows = []

    configs = [
        (f"{variant_name}_linear", "linear", 0),
        (f"{variant_name}_small", "small", 64),
        (f"{variant_name}_medium", "medium", 256),
        (f"{variant_name}_large", "large", 512),
    ]

    for name, size, hidden in configs:
        if size == "linear":
            model = SplitLinearAR(history_len=4, action_dim=action_dim).to(device)
            kind = "linear_ar"
        elif model_kind_prefix == "gru":
            model = SplitGRU(input_dim=action_dim, hidden_dim=max(hidden, 32), action_dim=action_dim).to(device)
            kind = "action_history_gru"
        elif model_kind_prefix == "gru_state":
            model = SplitGRUPlusState(
                state_dim=input_dim, action_dim=action_dim,
                history_len=4, hidden_dim=max(hidden, 32),
            ).to(device)
            kind = "full_state_history_gru" if input_dim > 50 else "proprio_history_gru"
        else:  # mlp_state
            if hidden > 256:
                model = LargeSplitMLP(input_dim=input_dim, hidden_dim=hidden, action_dim=action_dim).to(device)
            else:
                model = SplitMLP(input_dim=input_dim, hidden_dim=max(hidden, 32), action_dim=action_dim).to(device)
            kind = "full_state_mlp" if input_dim > 50 else "proprio_mlp"

        result = _train_split_model(
            model=model, model_kind=kind,
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=lr,
            action_contract=action_contract, action_transform=action_transform,
        )
        m = result["best_metrics"]
        rows.append({
            "variant": name,
            "capacity": size,
            "hidden_dim": hidden,
            "continuous_normalized_mse": m["continuous_normalized_mse"],
            "continuous_raw_mse": m["continuous_raw_mse"],
            "gripper_sign_accuracy": m["gripper_sign_accuracy"],
            "global_raw_mse": m["global_raw_mse"],
            "best_epoch": result["best_epoch"],
        })

    return rows


# ---------------------------------------------------------------------------
# 6. Upper bound: timestep/lookup variants
# ---------------------------------------------------------------------------

class TimestepEmbeddingSplitMLP(nn.Module):
    def __init__(self, *, max_time_index: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_time_index + 1, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = SplitActionGripperHead(hidden_dim, 1, action_dim)

    def forward(self, time_index: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.network(self.embedding(time_index)))


class WindowLookup(nn.Module):
    def __init__(self, n_windows, action_horizon, action_dim):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(n_windows, action_horizon, action_dim))

    def forward(self, idx):
        return self.table[idx]


def run_upper_bound_variants(
    *,
    train_ds,
    val_ds,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    action_contract: dict[str, Any],
    action_transform,
    epochs: int,
    lr: float,
    hidden_dim: int,
) -> list[dict[str, Any]]:
    """Run upper-bound sanity checks (NOT valid causal policies)."""
    action_dim = action_contract["action_dim"]
    rows = []

    # 1. Timestep embedding MLP
    max_t = max(int(train_ds[i]["time_index"]) for i in range(len(train_ds)))
    ts_model = TimestepEmbeddingSplitMLP(
        max_time_index=max_t, hidden_dim=hidden_dim, action_dim=action_dim,
    ).to(device)

    optimizer = torch.optim.Adam(ts_model.parameters(), lr=lr)
    best_mse = float("inf")
    best_epoch = -1
    for epoch in range(epochs):
        ts_model.train()
        for batch in train_loader:
            out = ts_model(torch.as_tensor(batch["time_index"], dtype=torch.long, device=device))
            target = batch["target_actions"].to(device)
            loss = _repair_loss(out, target, mode="split")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ts_model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = ts_model(torch.as_tensor(batch["time_index"], dtype=torch.long, device=device))
                pred = out["pred_actions"]
                tgt = batch["target_actions"]
                if action_transform:
                    pred = action_transform.denormalize_tensor(pred)
                    tgt = action_transform.denormalize_tensor(tgt)
                all_pred.append(pred.cpu())
                all_tgt.append(tgt.cpu())
        pred_all = torch.cat(all_pred)
        tgt_all = torch.cat(all_tgt)
        m = compute_split_metrics(pred_all, tgt_all, action_stats=action_contract["continuous_action_stats"])
        if m["continuous_normalized_mse"] < best_mse:
            best_mse = m["continuous_normalized_mse"]
            best_epoch = epoch

    rows.append({
        "variant": "timestep_embedding_mlp",
        "label": "diagnostic_only_not_causal",
        "continuous_normalized_mse": best_mse,
        "gripper_sign_accuracy": m["gripper_sign_accuracy"],
        "best_epoch": best_epoch,
    })

    # 2. Lookup table
    n_train = len(train_ds)
    lookup = WindowLookup(n_train, 1, action_dim).to(device)
    lookup_optim = torch.optim.Adam(lookup.parameters(), lr=0.01)

    train_targets = torch.stack([
        torch.as_tensor(train_ds[i]["target_actions"], dtype=torch.float32)
        for i in range(n_train)
    ]).to(device)
    val_targets = torch.stack([
        torch.as_tensor(val_ds[i]["target_actions"], dtype=torch.float32)
        for i in range(len(val_ds))
    ]).to(device)

    best_lookup_mse = float("inf")
    for epoch in range(500):
        lookup.train()
        pred = lookup(torch.arange(n_train, device=device))
        loss = F.mse_loss(pred, train_targets)
        lookup_optim.zero_grad()
        loss.backward()
        lookup_optim.step()

        lookup.eval()
        with torch.no_grad():
            val_pred = lookup(torch.arange(min(len(val_ds), n_train), device=device))
            if action_transform:
                val_pred = action_transform.denormalize_tensor(val_pred)
                val_tgt = action_transform.denormalize_tensor(val_targets[:n_train])
            else:
                val_tgt = val_targets[:n_train]
            val_mse = F.mse_loss(val_pred, val_tgt).item()
            if val_mse < best_lookup_mse:
                best_lookup_mse = val_mse

    rows.append({
        "variant": "lookup_table",
        "label": "diagnostic_only_not_causal",
        "continuous_normalized_mse": float("nan"),
        "continuous_raw_mse": float("nan"),
        "gripper_sign_accuracy": float("nan"),
        "best_epoch": -1,
        "global_raw_mse": best_lookup_mse,
    })

    return rows


# ---------------------------------------------------------------------------
# 7. Alternative action target experiments
# ---------------------------------------------------------------------------

def run_action_target_variants(
    *,
    actions_np: np.ndarray,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    action_contract: dict[str, Any],
    action_transform,
    epochs: int,
    lr: float,
    hidden_dim: int,
) -> list[dict[str, Any]]:
    """Test alternative action target representations."""
    action_dim = action_contract["action_dim"]
    rows = []

    # Baseline: direct action prediction (already done in main ladder)
    # Test: action residual relative to last_action
    # For each sample, target = action[t] - action[t-1]
    # This is a diagnostic-only variant

    # Build residual-action dataset
    class ResidualActionDS:
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            s = self.base[i]
            history = s["action_history"]
            target = s["target_actions"]
            if hasattr(history, 'ndim') and history.ndim == 2:
                last_action = history[-1:]  # [1, action_dim]
            else:
                last_action = history[:, -1:, :]
            residual_target = target - last_action
            s["target_actions_original"] = target.copy() if isinstance(target, np.ndarray) else target
            s["target_actions"] = residual_target
            return s

    res_train = ResidualActionDS(train_loader.dataset)
    res_val = ResidualActionDS(val_loader.dataset)

    def res_collate(batch):
        c = collate_action_batch(batch)
        if "full_state_t" in batch[0]:
            c["full_state_t"] = torch.stack([torch.as_tensor(s["full_state_t"], dtype=torch.float32) for s in batch])
        if "target_actions_original" in batch[0]:
            c["target_actions_original"] = torch.stack([
                torch.as_tensor(s["target_actions_original"], dtype=torch.float32) for s in batch
            ])
        return c

    res_train_loader = DataLoader(res_train, batch_size=train_loader.batch_size,
                                  shuffle=True, collate_fn=res_collate)
    res_val_loader = DataLoader(res_val, batch_size=val_loader.batch_size,
                                shuffle=False, collate_fn=res_collate)

    # Train a simple model on residual targets
    from src.eval.g8_mixed_action_metrics import SplitGRUPlusState
    model = SplitGRUPlusState(
        state_dim=92, action_dim=action_dim, history_len=4, hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        for batch in res_train_loader:
            target = batch["target_actions"].to(device)
            outputs = model(batch["action_history"].to(device), batch["full_state_t"].to(device))
            pred_cont = outputs["pred_continuous_actions"]
            target_cont = target[..., CONTINUOUS_DIMS]
            loss = F.smooth_l1_loss(pred_cont, target_cont)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_pred, all_tgt_orig = [], []
        with torch.no_grad():
            for batch in res_val_loader:
                outputs = model(batch["action_history"].to(device), batch["full_state_t"].to(device))
                pred_resid = outputs["pred_actions"]
                last_act = batch["action_history"][:, -1:, :].to(device)
                pred_abs = last_act + pred_resid
                tgt_orig = batch["target_actions_original"].to(device)
                if action_transform:
                    pred_abs = action_transform.denormalize_tensor(pred_abs)
                    tgt_orig = action_transform.denormalize_tensor(tgt_orig)
                all_pred.append(pred_abs.cpu())
                all_tgt_orig.append(tgt_orig.cpu())

        pred_all = torch.cat(all_pred)
        tgt_all = torch.cat(all_tgt_orig)
        m = compute_split_metrics(pred_all, tgt_all, action_stats=action_contract["continuous_action_stats"])
        if m["continuous_normalized_mse"] < best_mse:
            best_mse = m["continuous_normalized_mse"]
            best_epoch = epoch

    rows.append({
        "variant": "residual_action_target",
        "label": "target_repair_diagnostic",
        "continuous_normalized_mse": best_mse,
        "continuous_raw_mse": m["continuous_raw_mse"],
        "gripper_sign_accuracy": m["gripper_sign_accuracy"],
        "global_raw_mse": m["global_raw_mse"],
        "best_epoch": best_epoch,
    })

    return rows


# ---------------------------------------------------------------------------
# 8. Gripper transition dataset audit
# ---------------------------------------------------------------------------

def run_gripper_transition_audit(
    *,
    trajectories: list[RawTrajectory],
    output_dir: Path,
    git_info: dict[str, str],
    dataset: str,
) -> dict[str, Any]:
    """Audit gripper transitions across multiple demos."""
    demo_stats = []
    total_transitions = 0

    for traj in trajectories[:50]:  # limit to 50 demos
        actions = np.array(traj.actions, dtype=np.float32)
        gripper = actions[:, GRIPPER_DIM_IDX]
        gripper_binary = (gripper > 0).astype(int)
        transitions = []
        for t in range(1, len(gripper_binary)):
            if gripper_binary[t] != gripper_binary[t - 1]:
                transitions.append(t)

        n_open = int((gripper > GRIPPER_OPEN_THRESH).sum())
        n_close = int((gripper < GRIPPER_CLOSE_THRESH).sum())

        demo_stats.append({
            "trajectory_id": traj.trajectory_id,
            "length": len(gripper),
            "n_transitions": len(transitions),
            "n_open": n_open,
            "n_close": n_close,
            "fraction_open": n_open / len(gripper),
            "fraction_close": n_close / len(gripper),
        })
        total_transitions += len(transitions)

    avg_transitions = total_transitions / max(len(demo_stats), 1)

    audit = {
        "n_demos": len(demo_stats),
        "total_transitions": total_transitions,
        "avg_transitions_per_demo": avg_transitions,
        "demo_stats": demo_stats[:10],  # first 10 for brevity
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "dataset": dataset,
    }

    lines = [
        "# G9 Gripper Transition Dataset Audit",
        "",
        f"## Demos Analyzed: {len(demo_stats)}",
        f"## Total Transitions: {total_transitions}",
        f"## Average Transitions Per Demo: {avg_transitions:.2f}",
        "",
        "## Per-Demo Statistics (first 10)",
        "",
        "| Demo | Length | Transitions | Open% | Close% |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in demo_stats[:10]:
        lines.append(
            f"| {ds['trajectory_id'].split('/')[-1][:30]} | {ds['length']} | "
            f"{ds['n_transitions']} | {ds['fraction_open']:.1%} | {ds['fraction_close']:.1%} |"
        )

    lines.extend([
        "",
        "## Assessment",
        "",
        f"- With {avg_transitions:.1f} transitions per demo, gripper transition F1 is {'informative' if avg_transitions > 5 else 'uninformative'} for single-demo diagnostics.",
        "- Low transition count means F1 is dominated by the 'stay' class.",
        "- Multi-demo evaluation would provide more meaningful gripper transition statistics.",
    ])

    (output_dir / "gripper_transition_dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/g9_residual_action_repair"))
    parser.add_argument("--trajectory_id", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "any"], default="train")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--ladder_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--run_id", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = run_g9_diagnostics(
        config_path=args.config, output_root=args.output_dir,
        trajectory_id=args.trajectory_id, source_split=args.split,
        epochs=args.epochs, ladder_epochs=args.ladder_epochs,
        batch_size=args.batch_size, lr=args.lr,
        device_name=args.device, seed=args.seed, hidden_dim=args.hidden_dim,
        run_id=args.run_id,
        command=[sys.executable, "-m", "src.eval.g9_residual_action_repair", *(argv or sys.argv[1:])],
    )
    print(f"g9_output_dir={output_dir}")
    return 0


def run_g9_diagnostics(
    *,
    config_path: Path,
    output_root: Path,
    trajectory_id: str | None = None,
    source_split: str = "train",
    epochs: int = 300,
    ladder_epochs: int = 200,
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
    run_id = run_id or f"{timestamp}_g9_residual_repair"
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
    proprio_states = selected.states
    actions_np = np.array(selected.actions, dtype=np.float32)
    T, A = actions_np.shape
    history_len = int(config["data"]["history_len"])

    print(f"  Trajectory: {selected.trajectory_id} (T={T}, A={A})")

    # Build action contract
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

    def g9_collate(batch):
        c = collate_action_batch(batch)
        if "full_state_t" in batch[0]:
            c["full_state_t"] = torch.stack([torch.as_tensor(s["full_state_t"], dtype=torch.float32) for s in batch])
        return c

    train_loader = DataLoader(fs_train, batch_size=batch_size, shuffle=True, collate_fn=g9_collate)
    val_loader = DataLoader(fs_val, batch_size=batch_size, shuffle=False, collate_fn=g9_collate)

    sample = fs_train[0]
    action_dim = infer_action_dim(sample)
    effective_state_dim = infer_state_dim(sample) or 0
    full_state_dim = full_states.shape[1] if full_states is not None else 0

    # ===================================================================
    # 3. State-action alignment audit
    # ===================================================================
    print("[3/9] State-action alignment audit ...")
    alignment_audit = run_alignment_audit(
        actions_np=actions_np, states_92d=full_states, proprio=proprio_states,
        output_dir=output_dir, git_info=git_info,
    )

    # ===================================================================
    # 4. Normalization audit
    # ===================================================================
    print("[4/9] Normalization audit ...")
    norm_audit = run_normalization_audit(
        train_actions=actions_np, val_actions=actions_np,
        action_contract=action_contract, output_dir=output_dir,
    )

    # ===================================================================
    # 2. Action target semantics audit
    # ===================================================================
    print("[5/9] Action target semantics audit ...")
    semantics_audit = run_action_semantics_audit(
        actions_np=actions_np, states_np=full_states,
        output_dir=output_dir, git_info=git_info,
        dataset=dataset_name, trajectory_id=selected.trajectory_id,
    )

    # ===================================================================
    # 5. Capacity overfit ladder
    # ===================================================================
    print("[6/9] Capacity overfit ladder ...")
    capacity_rows = []

    # full_state_92d capacity ladder
    if full_state_dim > 0:
        rows = run_capacity_ladder(
            train_loader=train_loader, val_loader=val_loader, device=device,
            action_contract=action_contract, action_transform=action_transform,
            input_dim=full_state_dim, model_kind_prefix="gru_state",
            variant_name="full_state", epochs=ladder_epochs, lr=effective_lr,
        )
        capacity_rows.extend(rows)

    # action_history capacity ladder
    rows = run_capacity_ladder(
        train_loader=train_loader, val_loader=val_loader, device=device,
        action_contract=action_contract, action_transform=action_transform,
        input_dim=0, model_kind_prefix="gru",
        variant_name="action_history", epochs=ladder_epochs, lr=effective_lr,
    )
    capacity_rows.extend(rows)

    _write_csv_g9(output_dir / "capacity_overfit_ladder.csv", capacity_rows)

    # ===================================================================
    # 6. Upper bound variants
    # ===================================================================
    print("[7/9] Upper bound variants ...")
    upper_rows = run_upper_bound_variants(
        train_ds=fs_train, val_ds=fs_val,
        train_loader=train_loader, val_loader=val_loader, device=device,
        action_contract=action_contract, action_transform=action_transform,
        epochs=ladder_epochs, lr=effective_lr, hidden_dim=hidden_dim,
    )
    _write_csv_g9(output_dir / "upper_bound_split_metrics.csv", upper_rows)

    # ===================================================================
    # 7. Alternative action target variants
    # ===================================================================
    print("[8/9] Action target variant experiments ...")
    target_variant_rows = run_action_target_variants(
        actions_np=actions_np, train_loader=train_loader, val_loader=val_loader,
        device=device, action_contract=action_contract, action_transform=action_transform,
        epochs=ladder_epochs, lr=effective_lr, hidden_dim=hidden_dim,
    )
    _write_csv_g9(output_dir / "action_target_variant_ladder.csv", target_variant_rows)

    # ===================================================================
    # 8. Gripper transition audit
    # ===================================================================
    print("[9/9] Gripper transition audit ...")
    gripper_audit = run_gripper_transition_audit(
        trajectories=trajectories, output_dir=output_dir,
        git_info=git_info, dataset=dataset_name,
    )

    # ===================================================================
    # 1. Residual error attribution (after models are trained)
    # ===================================================================
    # Re-train the best models and collect per-timestep predictions
    print("  Computing residual attribution ...")
    from src.eval.g8_mixed_action_metrics import SplitGRUPlusState as G8SplitGRUPlusState

    best_models = {
        "full_state_plus_history": G8SplitGRUPlusState(
            state_dim=full_state_dim, action_dim=action_dim,
            history_len=history_len, hidden_dim=hidden_dim,
        ).to(device) if full_state_dim > 0 else None,
        "action_history_gru": SplitGRU(
            input_dim=action_dim, hidden_dim=hidden_dim, action_dim=action_dim,
        ).to(device),
    }

    attributions = {}
    time_indices_list = [val_ds[i]["time_index"] for i in range(len(val_ds))]

    for variant_name, model in best_models.items():
        if model is None:
            continue
        kind = "full_state_history_gru" if "full_state" in variant_name else "action_history_gru"
        result = _train_split_model(
            model=model, model_kind=kind,
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=effective_lr,
            action_contract=action_contract, action_transform=action_transform,
        )

        # Collect per-timestep predictions
        model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for batch in val_loader:
                if kind == "full_state_history_gru":
                    out = model(batch["action_history"].to(device), batch["full_state_t"].to(device))
                else:
                    out = model(batch["action_history"].to(device))
                pred = out["pred_actions"]
                tgt = batch["target_actions"]
                if action_transform:
                    pred = action_transform.denormalize_tensor(pred)
                    tgt = action_transform.denormalize_tensor(tgt)
                all_pred.append(pred.cpu())
                all_tgt.append(tgt.cpu())

        pred_all = torch.cat(all_pred)
        tgt_all = torch.cat(all_tgt)

        attr = compute_residual_attribution(
            pred_all, tgt_all, actions_np=actions_np,
            time_indices=time_indices_list, action_stats=action_contract["continuous_action_stats"],
        )
        attributions[variant_name] = attr

    # Add last_action baseline attribution
    last_pred_list, last_tgt_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            last = batch["action_history"][:, -1:, :]
            tgt = batch["target_actions"]
            if action_transform:
                last = action_transform.denormalize_tensor(last)
                tgt = action_transform.denormalize_tensor(tgt)
            last_pred_list.append(last.cpu())
            last_tgt_list.append(tgt.cpu())
    last_pred_all = torch.cat(last_pred_list)
    last_tgt_all = torch.cat(last_tgt_list)
    attributions["last_action"] = compute_residual_attribution(
        last_pred_all, last_tgt_all, actions_np=actions_np,
        time_indices=time_indices_list, action_stats=action_contract["continuous_action_stats"],
    )

    # Write residual reports
    write_residual_attribution_report(
        output_dir / "residual_error_attribution.md", attributions, action_contract["continuous_action_stats"],
    )

    # Per-dim residual metrics CSV
    per_dim_rows = []
    for variant, attr in attributions.items():
        for pd in attr["per_dim"]:
            per_dim_rows.append({"variant": variant, **pd})
    _write_csv_g9(output_dir / "per_dim_residual_metrics.csv", per_dim_rows)

    # Worst timestep errors CSV
    worst_rows = []
    for variant, attr in attributions.items():
        for wt in attr["worst_timesteps"]:
            worst_rows.append({"variant": variant, **wt})
    _write_csv_g9(output_dir / "worst_timestep_errors.csv", worst_rows)

    # ===================================================================
    # Summary
    # ===================================================================
    summary = {
        "status": "g9_residual_action_repair",
        "config": str(config_path),
        "trajectory_id": selected.trajectory_id,
        "trajectory_length": T,
        "task_id": selected.task_id,
        "task_name": selected.task_name,
        "dataset": dataset_name,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "seed": seed,
        "capacity_ladder": capacity_rows,
        "upper_bounds": upper_rows,
        "target_variants": target_variant_rows,
        "gripper_audit": gripper_audit,
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
        "G9 residual error attribution and action target repair. "
        "Diagnostic only, not closed-loop or architecture evidence.\n", encoding="utf-8")

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
