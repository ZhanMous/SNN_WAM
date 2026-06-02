#!/usr/bin/env python3
"""Persistence baseline: repeat last context patch latent as prediction.

No model needed. Pure non-model baseline for comparison.

Usage:
    python scripts/eval_persistence_baseline.py \
        --config configs/reportable/dinowm_baseline_real.yaml \
        --horizons 1 2 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.patch_latent_dataset import create_dinowm_transition_dataset  # noqa: E402
from src.train.metrics import (  # noqa: E402
    patch_cosine_error,
    patch_mean_cosine_error,
    patch_mse,
)
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", choices=["val", "train", "both"], default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def patch_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "z_context": torch.stack([s["z_context"] for s in samples], dim=0),
        "actions": torch.stack([s["actions"] for s in samples], dim=0),
        "z_target": torch.stack([s["z_target"] for s in samples], dim=0),
    }


@torch.no_grad()
def eval_persistence(
    loader: DataLoader,
    *,
    eval_horizon: int,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Evaluate persistence baseline: repeat last context frame as prediction."""
    patch_cosine_errs: list[float] = []
    patch_mse_vals: list[float] = []
    patch_mean_cos_errs: list[float] = []
    steps = 0

    for batch in loader:
        if max_steps is not None and steps >= max_steps:
            break

        z_context = batch["z_context"]  # [B, T_ctx, P, D]
        z_target_full = batch["z_target"]  # [B, H_model, P, D]

        B, T_ctx, P, D = z_context.shape

        # Persistence: repeat last context frame for all H future steps
        last_frame = z_context[:, -1:]  # [B, 1, P, D]
        pred = last_frame.expand(B, eval_horizon, P, D)  # [B, H, P, D]

        if z_target_full.shape[1] < eval_horizon:
            continue

        target_h = z_target_full[:, :eval_horizon]  # [B, H, P, D]

        p_cos = patch_cosine_error(pred, target_h, reduction="none")
        p_mse = patch_mse(pred, target_h, reduction="none")
        p_mean_cos = patch_mean_cosine_error(pred, target_h, reduction="none")

        patch_cosine_errs.extend(p_cos.mean(dim=1).cpu().tolist())
        patch_mse_vals.extend(p_mse.mean(dim=1).cpu().tolist())
        patch_mean_cos_errs.extend(p_mean_cos.mean(dim=1).cpu().tolist())
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
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)

    config = load_config(args.config)
    cache_dir = Path(config["data"]["cache_dir"])
    context_len = int(config["data"]["context_len"])
    future_horizon_model = int(config["data"]["future_horizon"])

    results_all: list[dict[str, Any]] = []

    for split_name in ([args.split] if args.split != "both" else ["val", "train"]):
        print(f"\n=== Split: {split_name} ===")

        max_h = max(args.horizons)
        dataset = create_dinowm_transition_dataset(
            cache_dir,
            context_len=context_len,
            future_horizon=max(max_h, future_horizon_model),
            split=split_name,
        )

        # Apply same split logic as trainer
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

        for h in args.horizons:
            print(f"  Evaluating H={h}...", end=" ")
            metrics = eval_persistence(loader, eval_horizon=h)
            metrics["split"] = split_name
            results_all.append(metrics)
            print(
                f"cos_err={metrics['patch_cosine_error']:.4f} "
                f"mse={metrics['patch_mse']:.6f} "
                f"mean_cos={metrics['patch_mean_cosine_error']:.4f} "
                f"n={metrics['n_samples']}"
            )

    # Write results
    out_dir = args.output_dir or (Path(config["output"]["output_dir"]) / "baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "baseline": "persistence",
        "description": "Repeat last context patch latent as prediction for all H future steps",
        "model_required": False,
        "horizons": args.horizons,
        "results": results_all,
    }
    (out_dir / "persistence_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    # Print degradation curve
    val_results = [r for r in results_all if r["split"] == "val" and r["n_samples"] > 0]
    if val_results:
        print("\n=== Persistence Degradation Curve (val) ===")
        print(f"  {'Horizon':>8}  {'Cos Err':>8}  {'MSE':>12}  {'Mean Cos':>8}")
        for r in sorted(val_results, key=lambda x: x["horizon"]):
            print(
                f"  H={r['horizon']:>5}  "
                f"{r['patch_cosine_error']:>8.4f}  "
                f"{r['patch_mse']:>12.6f}  "
                f"{r['patch_mean_cosine_error']:>8.4f}"
            )

    print(f"\nOutput: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
