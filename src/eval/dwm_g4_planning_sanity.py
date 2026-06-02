#!/usr/bin/env python3
"""DWM-G4 Planning Sanity: action optimization improves predicted target latent distance.

Evaluates whether gradient-based or CMA-ES action optimization through a trained
DINOwM world model reduces the predicted distance to a target patch latent more
than random or replay action baselines.

Acceptance criterion: optimized actions reduce predicted target latent distance
more than random actions.

Usage:
    python src/eval/dwm_g4_planning_sanity.py \\
        --run_dir results/runs/dinowm_transformer_baseline_real \\
        --n_samples 50 \\
        --horizon 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.patch_latent_dataset import create_dinowm_transition_dataset  # noqa: E402
from src.models.dinowm_transformer import DINOwMTransformer  # noqa: E402
from src.planning.action_optimizer import (  # noqa: E402
    compare_action_sources,
    optimize_actions_cmaes,
    optimize_actions_gradient,
    planning_objective_cosine,
)
from src.train.metrics import patch_cosine_error  # noqa: E402
from src.utils.seed import seed_everything  # # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path, help="Training run directory.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path.")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Override patch latent cache dir.")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of planning windows to evaluate.")
    parser.add_argument("--horizon", type=int, default=2, help="Planning horizon.")
    parser.add_argument("--n_random", type=int, default=10, help="Number of random action baselines per sample.")
    parser.add_argument("--opt_steps", type=int, default=200, help="Gradient optimization steps.")
    parser.add_argument("--opt_lr", type=float, default=0.05, help="Gradient optimization learning rate.")
    parser.add_argument("--cma_gens", type=int, default=50, help="CMA-ES generations.")
    parser.add_argument("--cma_pop", type=int, default=20, help="CMA-ES population size.")
    parser.add_argument("--random_baseline_type", choices=["uniform", "dataset", "shuffled_real"], default="uniform",
                        help="Random baseline type: uniform N(0,0.1), dataset distribution, or shuffled real actions.")
    parser.add_argument("--action_stats_path", type=Path, default=None,
                        help="Path to action_stats.json (mean/std) for dataset random baseline.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override output directory.")
    return parser.parse_args(argv)


def load_model(run_dir: Path, checkpoint_path: Path | None, device: torch.device) -> DINOwMTransformer:
    """Load trained DINOwMTransformer from checkpoint."""
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
    return model


def run_planning_sanity(
    run_dir: Path,
    *,
    checkpoint_path: Path | None = None,
    cache_dir: Path | None = None,
    n_samples: int = 50,
    horizon: int = 2,
    n_random: int = 10,
    opt_steps: int = 200,
    opt_lr: float = 0.05,
    cma_gens: int = 50,
    cma_pop: int = 20,
    random_baseline_type: str = "uniform",
    action_stats_path: Path | None = None,
    seed: int = 0,
    device_name: str = "cpu",
    output_dir: Path | None = None,
) -> Path:
    """Run DWM-G4 planning sanity evaluation.

    Returns path to the output directory with results.
    """
    device = torch.device(device_name)
    seed_everything(seed)

    # Load model
    model = load_model(run_dir, checkpoint_path, device)
    config = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["config"]

    # Load action stats for dataset random baseline
    action_stats = None
    if random_baseline_type == "dataset":
        if action_stats_path is not None and action_stats_path.exists():
            action_stats = json.loads(action_stats_path.read_text())
            print(f"  Loaded action stats from {action_stats_path}")
        else:
            # Compute from dataset
            print("  Computing action stats from dataset...")
            _temp_dataset = create_dinowm_transition_dataset(
                cache_dir or Path(config["data"]["cache_dir"]),
                context_len=int(config["data"]["context_len"]),
                future_horizon=int(model.future_horizon),
                split="train",
            )
            all_actions = torch.stack([s["actions"] for s in _temp_dataset], dim=0)  # [N, T_ctx, A]
            action_stats = {
                "mean": all_actions.mean(dim=[0, 1]).tolist(),  # [A]
                "std": all_actions.std(dim=[0, 1]).clamp(min=1e-6).tolist(),  # [A]
                "source_split": "train",
                "n_samples": len(_temp_dataset),
            }
            print(f"  Dataset action stats: mean={action_stats['mean'][:3]}..., std={action_stats['std'][:3]}...")

    # Load dataset
    if cache_dir is None:
        cache_dir = Path(config["data"]["cache_dir"])
    context_len = int(config["data"]["context_len"])
    action_dim = int(config["data"]["action_dim"])
    future_horizon_model = int(model.future_horizon)

    dataset = create_dinowm_transition_dataset(
        cache_dir,
        context_len=context_len,
        future_horizon=max(horizon, future_horizon_model),
        split="val",
    )

    if len(dataset) == 0:
        raise ValueError(f"No samples in dataset from {cache_dir}")

    # Sample windows for planning
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng)[:n_samples].tolist()

    # Results accumulators
    gradient_results: list[dict[str, Any]] = []
    cma_results: list[dict[str, Any]] = []
    comparison_results: list[dict[str, Any]] = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        z_context = sample["z_context"].unsqueeze(0).to(device)  # [1, T_ctx, P, D]
        z_target = sample["z_target"].unsqueeze(0).to(device)  # [1, H, P, D]
        gt_actions = sample["actions"].unsqueeze(0).to(device)  # [1, T_ctx, A]

        # Use last horizon actions as GT for replay baseline
        gt_actions_h = gt_actions[:, -horizon:, :]  # [1, H, A]

        print(f"  [{i+1}/{len(indices)}] idx={idx}", end=" ... ")

        # Gradient-based optimization
        grad_result = optimize_actions_gradient(
            model, z_context, z_target,
            horizon=horizon, action_dim=action_dim,
            n_steps=opt_steps, lr=opt_lr,
            objective="cosine", device=device,
        )
        gradient_results.append({
            "idx": idx,
            "initial_distance": grad_result.initial_distance,
            "optimized_distance": grad_result.optimized_distance,
            "reduction": grad_result.distance_reduction,
            "method": "gradient",
        })

        # CMA-ES optimization
        cma_result = optimize_actions_cmaes(
            model, z_context, z_target,
            horizon=horizon, action_dim=action_dim,
            n_generations=cma_gens, population_size=cma_pop,
            objective="cosine", seed=seed + i, device=device,
        )
        cma_results.append({
            "idx": idx,
            "initial_distance": cma_result.initial_distance,
            "optimized_distance": cma_result.optimized_distance,
            "reduction": cma_result.distance_reduction,
            "method": "cma_es",
        })

        # Full comparison (gradient only for speed)
        comp = compare_action_sources(
            model, z_context, z_target,
            horizon=horizon, action_dim=action_dim,
            gt_actions=gt_actions_h,
            n_random=n_random, seed=seed, objective="cosine",
            random_baseline_type=random_baseline_type,
            action_stats=action_stats, device=device,
        )
        comp_record = {
            "idx": idx,
            "zero_distance": comp["sources"]["zero"]["distance"],
            "random_distance": comp["sources"]["random"]["distance"],
            "random_std": comp["sources"]["random"]["std"],
            "random_type": comp["sources"]["random"].get("type", random_baseline_type),
            "replay_distance": comp["sources"].get("replay", {}).get("distance"),
            "optimized_distance": comp["sources"]["optimized"]["distance"],
            "pass": comp["pass"],
            "reduction_vs_random": comp["reduction_vs_random"],
            "improvement_ratio": comp["improvement_ratio"],
        }

        # Also run shuffled_real comparison if primary baseline is dataset
        # This gives the stricter "optimized beats temporal shuffle" test
        comp_shuffled = None
        if random_baseline_type == "dataset" and gt_actions_h is not None:
            comp_shuffled = compare_action_sources(
                model, z_context, z_target,
                horizon=horizon, action_dim=action_dim,
                gt_actions=gt_actions_h,
                n_random=n_random, seed=seed, objective="cosine",
                random_baseline_type="shuffled_real", device=device,
            )
            comp_record["pass_vs_shuffled"] = comp_shuffled["pass"]
            comp_record["reduction_vs_shuffled"] = comp_shuffled["reduction_vs_random"]
            comp_record["improvement_ratio_vs_shuffled"] = comp_shuffled["improvement_ratio"]

        comparison_results.append(comp_record)

        print(
            f"zero={comp['sources']['zero']['distance']:.4f} "
            f"rand={comp['sources']['random']['distance']:.4f} "
            f"opt={comp['sources']['optimized']['distance']:.4f} "
            f"PASS={comp['pass']}"
            + (f" PASS_shuf={comp_record.get('pass_vs_shuffled', '?')}" if comp_shuffled else "")
        )

    # Aggregate results
    grad_reductions = [r["reduction"] for r in gradient_results]
    cma_reductions = [r["reduction"] for r in cma_results]
    comp_passes = [r["pass"] for r in comparison_results]
    comp_reductions = [r["reduction_vs_random"] for r in comparison_results]
    comp_improvement_ratios = [r["improvement_ratio"] for r in comparison_results]
    num_failed = sum(1 for r in comparison_results if not r["pass"])

    # Shuffled_real comparison aggregates (if available)
    has_shuffled = any("pass_vs_shuffled" in r for r in comparison_results)
    if has_shuffled:
        shuffled_passes = [r.get("pass_vs_shuffled", False) for r in comparison_results if "pass_vs_shuffled" in r]
        shuffled_reductions = [r.get("reduction_vs_shuffled", 0) for r in comparison_results if "reduction_vs_shuffled" in r]
        shuffled_improvement = [r.get("improvement_ratio_vs_shuffled", 0) for r in comparison_results if "improvement_ratio_vs_shuffled" in r]

    summary = {
        "gate": "DWM-G4",
        "description": "planning sanity: action optimization improves predicted target latent distance",
        "n_samples": n_samples,
        "horizon": horizon,
        "seed": seed,
        "random_baseline_type": random_baseline_type,
        "gradient": {
            "mean_initial_distance": sum(r["initial_distance"] for r in gradient_results) / len(gradient_results),
            "mean_optimized_distance": sum(r["optimized_distance"] for r in gradient_results) / len(gradient_results),
            "mean_reduction": sum(grad_reductions) / len(grad_reductions),
            "positive_reduction_rate": sum(1 for r in grad_reductions if r > 0) / len(grad_reductions),
        },
        "cma_es": {
            "mean_initial_distance": sum(r["initial_distance"] for r in cma_results) / len(cma_results),
            "mean_optimized_distance": sum(r["optimized_distance"] for r in cma_results) / len(cma_results),
            "mean_reduction": sum(cma_reductions) / len(cma_reductions),
            "positive_reduction_rate": sum(1 for r in cma_reductions if r > 0) / len(cma_reductions),
        },
        "comparison": {
            "pass_rate": sum(comp_passes) / len(comp_passes),
            "mean_reduction_vs_random": sum(comp_reductions) / len(comp_reductions),
            "mean_improvement_ratio": sum(comp_improvement_ratios) / len(comp_improvement_ratios),
            "median_improvement_ratio": float(torch.tensor(comp_improvement_ratios).median().item()),
            "num_failed_optimizations": num_failed,
            "all_pass": all(comp_passes),
        },
        "pass_criterion": "optimized actions reduce predicted distance more than random actions",
        "result": "PASS" if sum(comp_passes) / len(comp_passes) > 0.5 else "FAIL",
    }

    # Add shuffled_real comparison if available
    if has_shuffled:
        summary["comparison_vs_shuffled"] = {
            "pass_rate": sum(shuffled_passes) / len(shuffled_passes),
            "mean_reduction_vs_shuffled": sum(shuffled_reductions) / len(shuffled_reductions),
            "mean_improvement_ratio_vs_shuffled": sum(shuffled_improvement) / len(shuffled_improvement),
            "median_improvement_ratio_vs_shuffled": float(torch.tensor(shuffled_improvement).median().item()),
            "all_pass_vs_shuffled": all(shuffled_passes),
            "note": "Stricter test: optimized beats temporal-shuffle of GT actions, not just Gaussian",
        }
        # Strong pass: beats both dataset Gaussian and shuffled_real
        summary["result_strong"] = (
            summary["result"] == "PASS"
            and summary["comparison_vs_shuffled"]["pass_rate"] > 0.5
        )

    # Write outputs
    out_dir = output_dir or (run_dir / "planning_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write summary
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Write per-sample CSV
    csv_path = out_dir / "per_sample_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "zero_distance", "random_distance", "random_std", "random_type",
            "replay_distance", "optimized_distance", "pass", "reduction_vs_random",
            "improvement_ratio",
            "pass_vs_shuffled", "reduction_vs_shuffled", "improvement_ratio_vs_shuffled",
        ])
        writer.writeheader()
        for r in comparison_results:
            writer.writerow({k: v for k, v in r.items() if k in writer.fieldnames})

    # Write optimization trace
    trace_path = out_dir / "optimization_traces.csv"
    with open(trace_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "method", "initial_distance", "optimized_distance", "reduction",
        ])
        writer.writeheader()
        for r in gradient_results:
            writer.writerow(r)
        for r in cma_results:
            writer.writerow(r)

    print(f"\nDWM-G4 Planning Sanity: {summary['result']}")
    print(f"  Pass rate (vs {random_baseline_type}): {summary['comparison']['pass_rate']:.1%}")
    print(f"  Median improvement ratio: {summary['comparison']['median_improvement_ratio']:.4f}")
    print(f"  Failed optimizations: {summary['comparison']['num_failed_optimizations']}/{n_samples}")
    if has_shuffled:
        print(f"  Pass rate (vs shuffled_real): {summary['comparison_vs_shuffled']['pass_rate']:.1%}")
        print(f"  Median improvement ratio (vs shuffled): {summary['comparison_vs_shuffled']['median_improvement_ratio_vs_shuffled']:.4f}")
        print(f"  Strong result (beats both): {summary.get('result_strong', False)}")
    print(f"  Gradient mean reduction: {summary['gradient']['mean_reduction']:.4f}")
    print(f"  CMA-ES mean reduction: {summary['cma_es']['mean_reduction']:.4f}")
    print(f"  Random baseline type: {random_baseline_type}")
    print(f"  Output: {out_dir}")

    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_planning_sanity(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        n_samples=args.n_samples,
        horizon=args.horizon,
        n_random=args.n_random,
        opt_steps=args.opt_steps,
        opt_lr=args.opt_lr,
        cma_gens=args.cma_gens,
        cma_pop=args.cma_pop,
        random_baseline_type=args.random_baseline_type,
        action_stats_path=args.action_stats_path,
        seed=args.seed,
        device_name=args.device,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
