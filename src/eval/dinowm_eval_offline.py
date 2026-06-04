#!/usr/bin/env python3
"""Multi-horizon offline evaluation for DINOwM Transformer.

Evaluates one-step and multi-step patch latent metrics at configurable
horizons (H=1,2,4) on held-out data. Supports action ablation modes
(real, zeros, shuffle) and rollout modes (teacher_forced, autoregressive).

Usage:
    python src/eval/dinowm_eval_offline.py \\
        --run_dir results/runs/dinowm_transformer_baseline_real \\
        --horizons 1 2 4 --action_mode real

    python src/eval/dinowm_eval_offline.py \\
        --run_dir results/runs/dinowm_transformer_baseline_real \\
        --horizons 1 2 4 --action_mode shuffle --shuffle_seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.patch_latent_dataset import create_dinowm_transition_dataset  # noqa: E402
from src.models.dinowm_transformer import DINOwMTransformer  # noqa: E402
from src.train.metrics import (  # noqa: E402
    patch_cosine_error,
    patch_mean_cosine_error,
    patch_mse,
)
from src.utils.seed import seed_everything  # # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path, help="Training run directory.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path.")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Override patch latent cache dir.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4], help="Future horizons to evaluate.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=None, help="Max eval steps per horizon.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--split", choices=["val", "train", "both"], default="val")
    parser.add_argument(
        "--action_mode", choices=["real", "zeros", "shuffle"], default="real",
        help="Future action mode: real (GT future candidate actions), zeros "
             "(OOD ablation: zero future actions, not a true no-action baseline), "
             "shuffle (batch/time permutation of future actions, paired eval)."
    )
    parser.add_argument(
        "--shuffle_seeds", type=int, nargs="+", default=[0],
        help="Seeds for action shuffling (paired eval: same sample_ids across seeds)."
    )
    parser.add_argument(
        "--rollout_mode", choices=["teacher_forced", "autoregressive"], default="autoregressive",
        help="Rollout mode for H > model_horizon: teacher_forced (GT context, no compounding error) "
             "or autoregressive (model predictions as context, compounding error)."
    )
    return parser.parse_args(argv)


def patch_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate PatchLatentTransitionDataset samples with metadata."""
    return {
        "z_context": torch.stack([s["z_context"] for s in samples], dim=0),
        "actions": torch.stack([s["actions"] for s in samples], dim=0),
        "future_actions": torch.stack([s["future_actions"] for s in samples], dim=0),
        "z_target": torch.stack([s["z_target"] for s in samples], dim=0),
        "metadata": [s.get("metadata", {}) for s in samples],
    }


def load_model(run_dir: Path, checkpoint_path: Path | None, device: torch.device) -> tuple[DINOwMTransformer, dict]:
    """Load trained model and config."""
    ckpt_path = checkpoint_path or (run_dir / "best.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

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
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    return model, config


def _apply_action_mode(
    actions: torch.Tensor,
    *,
    action_mode: str,
    shuffle_rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Transform actions according to the action mode."""
    if action_mode == "real":
        return actions
    elif action_mode == "zeros":
        return torch.zeros_like(actions)
    elif action_mode == "shuffle":
        # Paired action-ablation: break sample/action alignment while keeping
        # the future-action distribution. Batch permutation matters for H=1;
        # temporal permutation additionally breaks within-chunk ordering.
        B, H, A = actions.shape
        batch_perm = torch.randperm(B, generator=shuffle_rng, device=actions.device)
        shuffled = actions[batch_perm].clone()
        for b in range(B):
            time_perm = torch.randperm(H, generator=shuffle_rng, device=actions.device)
            shuffled[b] = shuffled[b, time_perm]
        return shuffled
    else:
        raise ValueError(f"Unknown action_mode: {action_mode!r}")


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    """Return a deterministic generator on the tensor device."""
    if device.type == "cpu":
        return torch.Generator().manual_seed(seed)
    return torch.Generator(device=device).manual_seed(seed)


def _future_action_chunk(
    future_actions: torch.Tensor,
    *,
    start: int,
    length: int,
) -> torch.Tensor:
    """Return [B, length, A] future actions, padding with the last action."""
    B, H, A = future_actions.shape
    end = start + length
    if start < H:
        chunk = future_actions[:, start:min(end, H)]
    else:
        chunk = future_actions[:, -1:]
    if chunk.shape[1] < length:
        pad = chunk[:, -1:].expand(B, length - chunk.shape[1], A)
        chunk = torch.cat([chunk, pad], dim=1)
    return chunk


@torch.no_grad()
def eval_one_horizon(
    model: DINOwMTransformer | None,
    loader: DataLoader,
    *,
    eval_horizon: int,
    model_horizon: int,
    device: torch.device,
    action_mode: str = "real",
    shuffle_seed: int = 0,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Evaluate metrics at a specific horizon.

    If eval_horizon > model_horizon, autoregressively chain predictions.
    If eval_horizon <= model_horizon, use first eval_horizon steps of prediction.
    """
    patch_cosine_errs: list[float] = []
    patch_mse_vals: list[float] = []
    patch_mean_cos_errs: list[float] = []
    per_sample_records: list[dict[str, Any]] = []
    total_fallback_samples = 0
    total_samples_seen = 0
    steps = 0

    for batch in loader:
        if max_steps is not None and steps >= max_steps:
            break

        z_context = batch["z_context"].to(device)  # [B, T_ctx, P, D]
        action_history = batch["actions"].to(device)  # [B, T_ctx, A]
        future_actions_raw = batch["future_actions"].to(device)  # [B, H_eval, A]
        z_target_full = batch["z_target"].to(device)  # [B, H_model, P, D]
        metadata_list = batch.get("metadata", [{}] * z_context.shape[0])

        B = z_context.shape[0]

        # Apply action mode only to explicit future candidate actions. Context
        # action history remains part of the observation/context.
        shuffle_rng = _make_generator(shuffle_seed + steps, device)
        future_actions = _apply_action_mode(
            future_actions_raw,
            action_mode=action_mode,
            shuffle_rng=shuffle_rng,
        )

        if eval_horizon <= model_horizon:
            pred = model(
                z_context,
                action_history,
                future_actions=future_actions[:, :model_horizon],
            )  # [B, H_model, P, D]
            pred_h = pred[:, :eval_horizon]  # [B, eval_h, P, D]
            target_h = z_target_full[:, :eval_horizon]  # [B, eval_h, P, D]
        else:
            if args.rollout_mode == "teacher_forced":
                # Teacher-forced: use GT context at each step (no compounding error)
                pred_h, batch_fallback = _teacher_forced_predict(
                    model, z_context, action_history, future_actions, eval_horizon, device,
                    z_target_full=z_target_full,
                )
                total_fallback_samples += batch_fallback
            else:
                # Autoregressive: chain predictions (compounding error)
                pred_h = _autoregressive_predict(
                    model, z_context, action_history, future_actions, eval_horizon, device
                )
            target_h = z_target_full[:, :eval_horizon] if z_target_full.shape[1] >= eval_horizon else None
            if target_h is None:
                continue

        total_samples_seen += B

        # Compute metrics
        p_cos = patch_cosine_error(pred_h, target_h, reduction="none")
        p_mse = patch_mse(pred_h, target_h, reduction="none")
        p_mean_cos = patch_mean_cosine_error(pred_h, target_h, reduction="none")

        cos_per_sample = p_cos.mean(dim=-1).cpu().tolist()  # [B, H] -> [B]
        mse_per_sample = p_mse.mean(dim=-1).cpu().tolist()  # [B, H] -> [B]
        mean_cos_per_sample = p_mean_cos.mean(dim=-1).cpu().tolist()  # [B, H] -> [B]

        patch_cosine_errs.extend(cos_per_sample)
        patch_mse_vals.extend(mse_per_sample)
        patch_mean_cos_errs.extend(mean_cos_per_sample)

        # Per-sample records — sample_id is deterministic from dataset metadata
        # (trajectory_id + time_index), so different action_mode / shuffle_seed
        # runs on the same window produce the same sample_id for paired comparison.
        for b in range(B):
            meta = metadata_list[b] if b < len(metadata_list) else {}
            traj_id = meta.get("trajectory_id", "unknown")
            t_idx = meta.get("time_index", -1)
            # Clean trajectory_id: strip .pt suffix and path prefix for brevity
            clean_traj = traj_id.split(":")[-1].replace("/", "_") if ":" in traj_id else traj_id.replace("/", "_")
            sample_id = f"{clean_traj}_t{t_idx}"
            per_sample_records.append({
                "sample_id": sample_id,
                "task_id": traj_id,
                "episode_id": traj_id,
                "window_start": t_idx,
                "horizon": eval_horizon,
                "action_mode": action_mode,
                "shuffle_seed": shuffle_seed if action_mode == "shuffle" else None,
                "patch_cosine_error": cos_per_sample[b] if b < len(cos_per_sample) else None,
                "patch_mse": mse_per_sample[b] if b < len(mse_per_sample) else None,
                "patch_mean_cosine_error": mean_cos_per_sample[b] if b < len(mean_cos_per_sample) else None,
            })

        steps += 1

    if not patch_cosine_errs:
        return {
            "horizon": eval_horizon,
            "n_samples": 0,
            "patch_cosine_error": float("nan"),
            "patch_cosine_error_std": float("nan"),
            "patch_mse": float("nan"),
            "patch_mse_std": float("nan"),
            "patch_mean_cosine_error": float("nan"),
            "patch_mean_cosine_error_std": float("nan"),
            "per_sample": [],
            "fallback_samples": 0,
            "total_samples": 0,
        }

    return {
        "horizon": eval_horizon,
        "n_samples": len(patch_cosine_errs),
        "patch_cosine_error": sum(patch_cosine_errs) / len(patch_cosine_errs),
        "patch_cosine_error_std": torch.tensor(patch_cosine_errs).std().item(),
        "patch_mse": sum(patch_mse_vals) / len(patch_mse_vals),
        "patch_mse_std": torch.tensor(patch_mse_vals).std().item(),
        "patch_mean_cosine_error": sum(patch_mean_cos_errs) / len(patch_mean_cos_errs),
        "patch_mean_cosine_error_std": torch.tensor(patch_mean_cos_errs).std().item(),
        "per_sample": per_sample_records,
        "fallback_samples": total_fallback_samples,
        "total_samples": total_samples_seen,
    }


@torch.no_grad()
def _teacher_forced_predict(
    model: DINOwMTransformer,
    z_context: torch.Tensor,
    action_history: torch.Tensor,
    future_actions: torch.Tensor,
    target_horizon: int,
    device: torch.device,
    z_target_full: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Teacher-forced prediction: use GT patch latents as context at each step.

    For each future step t, the context window is filled with ground-truth patch
    latents (shifted by 1). This eliminates compounding error from prediction
    drift, isolating per-step prediction accuracy.

    Requires z_target_full of shape [B, target_horizon, P, D] to supply GT
    context frames. If not provided or insufficient, falls back to autoregressive
    behavior (uses model predictions as context), which is NOT true teacher-forcing.

    Returns:
        (preds, fallback_count): preds is [B, target_horizon, P, D],
        fallback_count is the number of samples that fell back to autoregressive
        due to missing GT frames.
    """
    B, T_ctx, P, D = z_context.shape
    all_preds = []
    fallback_count = 0

    # Build a buffer of GT frames: context + target
    # gt_frames[t] gives the GT patch latent at absolute time t
    if z_target_full is not None and z_target_full.shape[1] >= target_horizon:
        # gt_frames: [B, T_ctx + target_horizon, P, D]
        gt_frames = torch.cat([z_context, z_target_full[:, :target_horizon]], dim=1)
        use_gt = True
    else:
        # Fallback: no GT beyond context, use autoregressive (not true teacher-forced)
        gt_frames = None
        use_gt = False
        fallback_count = B  # all samples in batch fallback

    for step in range(target_horizon):
        # Predict model_horizon steps from current context
        future_chunk = _future_action_chunk(
            future_actions,
            start=step,
            length=model.future_horizon,
        )
        pred = model(
            z_context,
            action_history,
            future_actions=future_chunk,
        )  # [B, H_model, P, D]
        all_preds.append(pred[:, :1])  # [B, 1, P, D]

        if use_gt:
            # Teacher-forced: shift context to use GT at step T_ctx + step
            gt_step = gt_frames[:, T_ctx + step: T_ctx + step + 1]  # [B, 1, P, D]
            z_context = torch.cat([z_context[:, 1:], gt_step], dim=1)
        else:
            # Fallback: use model prediction (autoregressive, not teacher-forced)
            z_context = torch.cat([z_context[:, 1:], pred[:, :1]], dim=1)

        # Shift action history by appending the executed future action.
        action_history = torch.cat([
            action_history[:, 1:],
            future_chunk[:, :1],
        ], dim=1)

    return torch.cat(all_preds, dim=1)[:, :target_horizon], fallback_count


@torch.no_grad()
def _autoregressive_predict(
    model: DINOwMTransformer,
    z_context: torch.Tensor,
    action_history: torch.Tensor,
    future_actions: torch.Tensor,
    target_horizon: int,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressively predict target_horizon steps by chaining model outputs.

    Each prediction uses the model's own previous predictions as context,
    causing compounding error. This is the realistic eval for multi-step accuracy.

    IMPORTANT: When chaining beyond model_horizon, future actions are unavailable.
    We pad with the last known action (repetition), NOT zeros. This is still OOD
    relative to training (where actions are diverse), but less extreme than zero
    padding. The action-padding strategy is recorded in summary.json.

    Args:
        model: Trained DINOwMTransformer.
        z_context: [B, T_ctx, P, D] context patch latents.
        action_history: [B, T_ctx, A] context actions.
        future_actions: [B, H_eval, A] candidate future actions.
        target_horizon: Number of future steps to predict.
        device: Torch device.

    Returns:
        [B, target_horizon, P, D] predicted future patch latents.
    """
    B, T_ctx, P, D = z_context.shape
    model_h = model.future_horizon

    all_preds = []
    current_context = z_context.clone()
    current_action_history = action_history.clone()
    remaining = target_horizon
    consumed = 0

    while remaining > 0:
        predict_steps = min(model_h, remaining)
        future_chunk = _future_action_chunk(
            future_actions,
            start=consumed,
            length=model_h,
        )

        pred = model(
            current_context,
            current_action_history,
            future_actions=future_chunk,
        )  # [B, model_h, P, D]
        pred_chunk = pred[:, :predict_steps]  # [B, predict_steps, P, D]
        all_preds.append(pred_chunk)
        remaining -= predict_steps
        consumed += predict_steps

        # Shift context: drop oldest predict_steps, append predictions
        current_context = torch.cat([current_context[:, predict_steps:], pred_chunk], dim=1)

        # Shift action history by appending the executed future action chunk.
        executed_actions = future_chunk[:, :predict_steps]
        current_action_history = torch.cat(
            [current_action_history[:, predict_steps:], executed_actions],
            dim=1,
        )

    return torch.cat(all_preds, dim=1)[:, :target_horizon]


def main(argv: Sequence[str] | None = None) -> int:
    global args
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device(args.device)

    # Load model
    model, config = load_model(args.run_dir, args.checkpoint, device)
    model_horizon = int(model.future_horizon)
    print(f"Model future_horizon={model_horizon}, eval horizons={args.horizons}")
    print(f"Action mode: {args.action_mode}, Rollout mode: {args.rollout_mode}")

    # Load dataset
    cache_dir = args.cache_dir or Path(config["data"]["cache_dir"])
    context_len = int(config["data"]["context_len"])

    all_per_sample: list[dict[str, Any]] = []
    results_all: list[dict[str, Any]] = []

    for split_name in ([args.split] if args.split != "both" else ["val", "train"]):
        print(f"\n=== Split: {split_name} ===")

        max_h = max(args.horizons)
        dataset = create_dinowm_transition_dataset(
            cache_dir,
            context_len=context_len,
            future_horizon=max(max_h, model_horizon),
            split=split_name,
        )

        # Apply same split as trainer
        if split_name == "val":
            n_total = len(dataset)
            n_train = int(n_total * 0.9)
            indices = list(range(n_train, n_total))
            dataset = Subset(dataset, indices)
        elif split_name == "train":
            n_total = len(dataset)
            n_train = int(n_total * 0.9)
            indices = list(range(n_train))
            dataset = Subset(dataset, indices)

        print(f"  {split_name} windows: {len(dataset)}")

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=patch_collate_fn,
            num_workers=0,
        )

        # Determine seeds to run
        seeds_to_run = [0] if args.action_mode != "shuffle" else args.shuffle_seeds

        for shuffle_seed in seeds_to_run:
            for h in args.horizons:
                seed_label = f" seed={shuffle_seed}" if args.action_mode == "shuffle" else ""
                print(f"  H={h}{seed_label}...", end=" ")
                metrics = eval_one_horizon(
                    model, loader,
                    eval_horizon=h,
                    model_horizon=model_horizon,
                    device=device,
                    action_mode=args.action_mode,
                    shuffle_seed=shuffle_seed,
                    max_steps=args.max_steps,
                )
                metrics["split"] = split_name
                metrics["action_mode"] = args.action_mode
                metrics["shuffle_seed"] = shuffle_seed
                metrics["rollout_mode"] = args.rollout_mode
                results_all.append(metrics)
                all_per_sample.extend(metrics.get("per_sample", []))
                print(
                    f"cos_err={metrics['patch_cosine_error']:.4f} "
                    f"mse={metrics['patch_mse']:.6f} "
                    f"mean_cos={metrics['patch_mean_cosine_error']:.4f} "
                    f"n={metrics['n_samples']}"
                )

    # Determine output dir
    if args.output_dir:
        out_dir = args.output_dir
    elif args.action_mode == "real":
        out_dir = args.run_dir / "eval_multihorizon"
    else:
        out_dir = args.run_dir / "eval_multihorizon_ablation" / args.action_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write per-sample CSV
    if all_per_sample:
        sample_csv_path = out_dir / "per_sample_metrics.csv"
        with open(sample_csv_path, "w", newline="") as f:
            fieldnames = [
                "sample_id", "task_id", "episode_id", "window_start",
                "horizon", "action_mode", "shuffle_seed",
                "patch_cosine_error", "patch_mse", "patch_mean_cosine_error",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in all_per_sample:
                writer.writerow({k: rec.get(k, "") for k in fieldnames})

    # Write aggregate CSV
    csv_path = out_dir / "eval_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "horizon", "n_samples", "action_mode", "shuffle_seed", "rollout_mode",
            "patch_cosine_error", "patch_cosine_error_std",
            "patch_mse", "patch_mse_std",
            "patch_mean_cosine_error", "patch_mean_cosine_error_std",
        ])
        writer.writeheader()
        for r in results_all:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    # Summary JSON — includes rollout behavior documentation and fallback tracking
    rollout_behavior = {}
    if args.rollout_mode == "teacher_forced":
        rollout_behavior = {
            "description": "GT patch latents used as context at each step; no compounding error",
            "compounding_error": False,
            "action_padding": "last available future action repeated only beyond requested horizon",
            "note": "Measures per-step prediction accuracy, not open-loop rollout fidelity",
        }
    else:
        rollout_behavior = {
            "description": "Model predictions chained as context; compounding error present",
            "compounding_error": True,
            "action_padding": "uses recorded future actions when available; repeats last only beyond requested horizon",
            "note": "Evaluates action-conditioned autoregressive latent prediction with explicit future actions",
        }

    # Compute aggregate fallback stats across all horizons
    total_fallback = sum(r.get("fallback_samples", 0) for r in results_all)
    total_all = sum(r.get("total_samples", 0) for r in results_all)
    any_fallback = total_fallback > 0

    summary = {
        "requested_rollout_mode": args.rollout_mode,
        "gt_available": not any_fallback,
        "fallback_triggered": any_fallback,
        "num_fallback_samples": total_fallback,
        "total_samples": total_all,
        "fallback_ratio": total_fallback / total_all if total_all > 0 else 0.0,
        "horizons": args.horizons,
        "model_horizon": model_horizon,
        "action_mode": args.action_mode,
        "shuffle_seeds": args.shuffle_seeds if args.action_mode == "shuffle" else [],
        "rollout_behavior": rollout_behavior,
        "results": [{k: v for k, v in r.items() if k != "per_sample"} for r in results_all],
    }

    if any_fallback:
        summary["fallback_warning"] = (
            f"{total_fallback}/{total_all} samples fell back to autoregressive "
            f"due to insufficient GT frames. This result should NOT be reported as "
            f"standard teacher-forced eval; treat as degraded diagnostic only."
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    # Degradation analysis
    val_results = [r for r in results_all if r["split"] == "val" and r["n_samples"] > 0]
    if val_results:
        print(f"\n=== Degradation Curve (val, {args.action_mode}) ===")
        print(f"  {'Horizon':>8}  {'Cos Err':>8}  {'MSE':>12}  {'Mean Cos':>8}")
        for r in sorted(val_results, key=lambda x: x["horizon"]):
            print(
                f"  H={r['horizon']:>5}  "
                f"{r['patch_cosine_error']:>8.4f}  "
                f"{r['patch_mse']:>12.6f}  "
                f"{r['patch_mean_cosine_error']:>8.4f}"
            )

        cos_errs = [r["patch_cosine_error"] for r in sorted(val_results, key=lambda x: x["horizon"])]
        is_degrading = all(cos_errs[i] <= cos_errs[i+1] for i in range(len(cos_errs)-1))
        print(f"\n  Monotonic degradation: {'YES' if is_degrading else 'NO'}")

    # Fallback warning
    if any_fallback:
        print(f"\n  *** FALLBACK WARNING ***")
        print(f"  {total_fallback}/{total_all} samples used autoregressive fallback")
        print(f"  (insufficient GT frames for teacher-forcing at requested horizons)")
        print(f"  This result is degraded diagnostic, NOT standard teacher-forced eval.")

    print(f"\nOutput: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
