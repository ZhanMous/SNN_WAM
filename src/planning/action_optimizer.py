"""Action sequence optimizer for DINO-WM planning sanity (DWM-G4).

Optimizes candidate action sequences through a learned world model to minimize
predicted distance to a target patch latent. Supports two methods:

1. Gradient-based (Adam/LBFGS): Directly optimizes action parameters through
   the differentiable world model. Fast, simple, no new dependencies.
2. CMA-ES: Evolutionary strategy over action parameters. Better for
   non-differentiable objectives but needs the ``cma`` package.

Shape conventions:
- World model input: patch_latents [B, T, P, D], actions [B, T, A]
- World model output: predicted [B, H, P, D]
- Optimized actions: [1, H, A] or [1, T_action, A]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F


@dataclass
class PlanningResult:
    """Result of action sequence optimization."""

    optimized_actions: torch.Tensor  # [1, T_ctx, A] full action sequence
    initial_distance: float
    optimized_distance: float
    distance_reduction: float
    method: str
    optimization_trace: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def planning_objective_cosine(
    pred_patch: torch.Tensor,
    target_patch: torch.Tensor,
) -> torch.Tensor:
    """Compute planning objective: mean cosine distance to target.

    Args:
        pred_patch: [B, H, P, D] predicted future patch latents
        target_patch: [B, H, P, D] target patch latents

    Returns:
        Scalar loss (lower is better). Mean of (1 - cosine_similarity) over
        patches, horizon, and batch.
    """
    # [B, H, P, D] -> cosine similarity over D dimension
    cos_sim = F.cosine_similarity(pred_patch, target_patch, dim=-1)  # [B, H, P]
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    error = 1.0 - cos_sim  # [B, H, P]
    return error.mean()


def planning_objective_mse(
    pred_patch: torch.Tensor,
    target_patch: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE planning objective.

    Args:
        pred_patch: [B, H, P, D] predicted future patch latents
        target_patch: [B, H, P, D] target patch latents

    Returns:
        Scalar loss (lower is better).
    """
    return F.mse_loss(pred_patch, target_patch)


def optimize_actions_gradient(
    world_model: torch.nn.Module,
    z_context: torch.Tensor,
    z_target: torch.Tensor,
    *,
    horizon: int,
    action_dim: int,
    n_steps: int = 200,
    lr: float = 0.05,
    init_method: str = "zeros",
    objective: str = "cosine",
    action_std: float = 0.1,
    device: torch.device | str = "cpu",
) -> PlanningResult:
    """Optimize action sequences via gradient descent through the world model.

    Args:
        world_model: Trained DINOwMTransformer (eval mode).
        z_context: [1, T_ctx, P, D] current patch latent context.
        z_target: [1, H, P, D] target future patch latents.
        horizon: Number of future timesteps H to optimize over.
        action_dim: Dimensionality of action space (A).
        n_steps: Number of optimization steps.
        lr: Learning rate for action optimizer.
        init_method: "zeros", "random", or "replay" (requires actions in z_context).
        objective: "cosine" or "mse".
        action_std: Std for random initialization.
        device: Torch device.

    Returns:
        PlanningResult with optimized actions and distance metrics.
    """
    world_model.eval()

    if objective == "cosine":
        obj_fn = planning_objective_cosine
    elif objective == "mse":
        obj_fn = planning_objective_mse
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    T_ctx = z_context.shape[1]

    # Initialize action sequence to optimize: [1, T_ctx, A]
    # The model expects actions with same T as patch_latents.
    # We optimize the full action sequence but the model only "sees" T_ctx steps.
    candidate_actions = torch.randn(1, T_ctx, action_dim, device=device) * action_std
    if init_method == "zeros":
        candidate_actions = torch.zeros(1, T_ctx, action_dim, device=device)
    elif init_method == "random":
        pass  # already random
    elif init_method == "replay":
        candidate_actions = torch.zeros(1, T_ctx, action_dim, device=device)

    candidate_actions = candidate_actions.requires_grad_(True)

    # Compute initial distance (zero actions baseline)
    with torch.no_grad():
        zero_actions = torch.zeros(1, T_ctx, action_dim, device=device)
        pred_init = world_model(z_context, zero_actions)
        initial_dist = float(obj_fn(pred_init, z_target).item())

    # Optimize
    optimizer = torch.optim.Adam([candidate_actions], lr=lr)
    trace: list[float] = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Model takes [B, T_ctx, P, D] + [B, T_ctx, A] and predicts [B, H, P, D]
        pred = world_model(z_context, candidate_actions)

        loss = obj_fn(pred, z_target)
        loss.backward()
        optimizer.step()

        trace.append(float(loss.item()))

    # Final optimized distance
    with torch.no_grad():
        pred_final = world_model(z_context, candidate_actions)
        optimized_dist = float(obj_fn(pred_final, z_target).item())

    return PlanningResult(
        optimized_actions=candidate_actions.detach().clone(),
        initial_distance=initial_dist,
        optimized_distance=optimized_dist,
        distance_reduction=initial_dist - optimized_dist,
        method="gradient",
        optimization_trace=trace,
        metadata={
            "n_steps": n_steps,
            "lr": lr,
            "init_method": init_method,
            "objective": objective,
            "final_loss": trace[-1] if trace else None,
        },
    )


def optimize_actions_cmaes(
    world_model: torch.nn.Module,
    z_context: torch.Tensor,
    z_target: torch.Tensor,
    *,
    horizon: int,
    action_dim: int,
    n_generations: int = 50,
    population_size: int = 20,
    sigma: float = 0.1,
    objective: str = "cosine",
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> PlanningResult:
    """Optimize action sequences via CMA-ES.

    Falls back to a simple random-restart hill climber if the ``cma`` package
    is not installed.

    Args:
        world_model: Trained DINOwMTransformer (eval mode).
        z_context: [1, T_ctx, P, D] current patch latent context.
        z_target: [1, H, P, D] target future patch latents.
        horizon: Number of future timesteps.
        action_dim: Action dimensionality.
        n_generations: Number of CMA-ES generations.
        population_size: Population size per generation.
        sigma: Initial step size.
        objective: "cosine" or "mse".
        seed: Random seed.
        device: Torch device.

    Returns:
        PlanningResult with optimized actions and metrics.
    """
    world_model.eval()

    if objective == "cosine":
        obj_fn = planning_objective_cosine
    elif objective == "mse":
        obj_fn = planning_objective_mse
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    T_ctx = z_context.shape[1]
    dim = T_ctx * action_dim

    def evaluate_actions(actions_flat: torch.Tensor) -> float:
        """Evaluate a single action vector."""
        actions = actions_flat.reshape(1, T_ctx, action_dim).to(device)
        with torch.no_grad():
            pred = world_model(z_context, actions)
            return float(obj_fn(pred, z_target).item())

    # Compute initial distance
    initial_dist = evaluate_actions(torch.zeros(dim))

    try:
        import cma

        def objective_fn(x):
            return evaluate_actions(torch.tensor(x, dtype=torch.float32))

        x0 = [0.0] * dim
        opts = cma.CMAOptions()
        opts["popsize"] = population_size
        opts["maxiter"] = n_generations
        opts["sigma"] = sigma
        opts["seed"] = seed
        opts["verb_disp"] = 0
        opts["verb_log"] = 0

        es = cma.CMAEvolutionStrategy(x0, opts)
        trace: list[float] = []
        for gen in range(n_generations):
            solutions = es.ask()
            fitnesses = [objective_fn(x) for x in solutions]
            es.tell(solutions, fitnesses)
            best_fit = min(fitnesses)
            trace.append(best_fit)

        best_flat = torch.tensor(es.result.xbest, dtype=torch.float32)
        optimized_dist = float(evaluate_actions(best_flat))

    except ImportError:
        warnings.warn("cma package not installed; using random-restart hill climber")
        best_flat, optimized_dist, trace = _hill_climber(
            evaluate_actions, dim, n_generations, population_size, sigma, seed
        )

    return PlanningResult(
        optimized_actions=best_flat.reshape(1, T_ctx, action_dim).detach().clone(),
        initial_distance=initial_dist,
        optimized_distance=optimized_dist,
        distance_reduction=initial_dist - optimized_dist,
        method="cma_es",
        optimization_trace=trace,
        metadata={
            "n_generations": n_generations,
            "population_size": population_size,
            "sigma": sigma,
            "objective": objective,
            "seed": seed,
        },
    )


def _hill_climber(
    evaluate_fn: Callable[[torch.Tensor], float],
    dim: int,
    n_iterations: int,
    population_size: int,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, float, list[float]]:
    """Simple random-restart hill climber fallback."""
    rng = torch.Generator().manual_seed(seed)
    best_flat = torch.randn(dim, generator=rng) * sigma
    best_val = evaluate_fn(best_flat)
    trace: list[float] = [best_val]

    for _ in range(n_iterations):
        candidates = best_flat.unsqueeze(0) + torch.randn(population_size, dim, generator=rng) * sigma
        vals = [evaluate_fn(c) for c in candidates]
        min_idx = min(range(len(vals)), key=lambda i: vals[i])
        if vals[min_idx] < best_val:
            best_flat = candidates[min_idx]
            best_val = vals[min_idx]
        trace.append(best_val)

    return best_flat, best_val, trace


def compare_action_sources(
    world_model: torch.nn.Module,
    z_context: torch.Tensor,
    z_target: torch.Tensor,
    *,
    horizon: int,
    action_dim: int,
    gt_actions: torch.Tensor | None = None,
    n_random: int = 10,
    seed: int = 0,
    objective: str = "cosine",
    random_baseline_type: str = "uniform",
    action_stats: dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Compare multiple action sources for planning sanity.

    Args:
        world_model: Trained world model (eval mode).
        z_context: [1, T_ctx, P, D] context patch latents.
        z_target: [1, H, P, D] target patch latents.
        horizon: Planning horizon.
        action_dim: Action dimension.
        gt_actions: Optional [1, H, A] ground truth actions for replay baseline.
        n_random: Number of random action baselines to average.
        seed: Random seed.
        objective: Planning objective.
        random_baseline_type: "uniform" (N(0, 0.1)), "dataset" (requires action_stats),
            or "shuffled_real" (shuffle gt_actions temporally).
        action_stats: Required if random_baseline_type="dataset". Dict with
            "mean" [A] and "std" [A] from the training set action distribution.
        device: Torch device.

    Returns:
        Dict with per-source distances, improvement_ratio, and pass/fail criterion.
    """
    world_model.eval()

    if objective == "cosine":
        obj_fn = planning_objective_cosine
    else:
        obj_fn = planning_objective_mse

    results: dict[str, Any] = {
        "sources": {},
        "random_baseline_type": random_baseline_type,
    }

    T_ctx = z_context.shape[1]

    # Zero actions
    with torch.no_grad():
        zero_actions = torch.zeros(1, T_ctx, action_dim, device=device)
        pred_zero = world_model(z_context, zero_actions)
        dist_zero = float(obj_fn(pred_zero, z_target).item())
    results["sources"]["zero"] = {"distance": dist_zero, "actions": zero_actions}

    # Random actions (averaged over n_random seeds)
    rng = torch.Generator().manual_seed(seed)
    random_dists = []
    for _ in range(n_random):
        if random_baseline_type == "dataset" and action_stats is not None:
            # Sample from dataset action distribution: N(action_mean, action_std)
            a_mean = torch.tensor(action_stats["mean"], dtype=torch.float32, device=device)
            a_std = torch.tensor(action_stats["std"], dtype=torch.float32, device=device)
            rand_actions = torch.randn(1, T_ctx, action_dim, generator=rng, device=device) * a_std + a_mean
        elif random_baseline_type == "shuffled_real" and gt_actions is not None:
            # Temporal shuffle of GT actions
            perm = torch.randperm(T_ctx, generator=rng)
            rand_actions = gt_actions[:, perm].to(device)
        else:
            # Uniform baseline: N(0, 0.1)
            rand_actions = torch.randn(1, T_ctx, action_dim, generator=rng, device=device) * 0.1
        with torch.no_grad():
            pred_rand = world_model(z_context, rand_actions)
            d = float(obj_fn(pred_rand, z_target).item())
            random_dists.append(d)
    dist_random = sum(random_dists) / len(random_dists)
    results["sources"]["random"] = {
        "distance": dist_random,
        "std": torch.tensor(random_dists).std().item(),
        "type": random_baseline_type,
    }

    # Ground truth replay (if available)
    if gt_actions is not None:
        with torch.no_grad():
            pred_gt = world_model(z_context, gt_actions.to(device))
            dist_gt = float(obj_fn(pred_gt, z_target).item())
        results["sources"]["replay"] = {"distance": dist_gt}

    # Optimized actions (gradient-based)
    opt_result = optimize_actions_gradient(
        world_model, z_context, z_target,
        horizon=horizon, action_dim=action_dim,
        n_steps=200, lr=0.05, objective=objective, device=device,
    )
    results["sources"]["optimized"] = {
        "distance": opt_result.optimized_distance,
        "initial_distance": opt_result.initial_distance,
        "reduction": opt_result.distance_reduction,
        "actions": opt_result.optimized_actions,
    }

    # Pass criterion: optimized < random
    results["pass"] = opt_result.optimized_distance < dist_random
    results["reduction_vs_random"] = dist_random - opt_result.optimized_distance
    # Improvement ratio: positive means optimized beat random
    if dist_random > 0:
        results["improvement_ratio"] = (dist_random - opt_result.optimized_distance) / dist_random
    else:
        results["improvement_ratio"] = 0.0

    return results


__all__ = [
    "PlanningResult",
    "planning_objective_cosine",
    "planning_objective_mse",
    "optimize_actions_gradient",
    "optimize_actions_cmaes",
    "compare_action_sources",
]
