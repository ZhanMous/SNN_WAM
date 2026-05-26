---
name: es-post-training
description: Use only when implementing or reviewing surrogate-gradient warmup plus EGGROLL-style or low-rank evolution-strategy post-training for SNN-WAM adapters, especially with non-differentiable closed-loop fitness.
---

# ES Post-training

Use ES only after a supervised/surrogate-gradient model can already run. Do not use ES from scratch for the first implementation unless the user explicitly asks.

## When ES is justified

ES is appropriate when the objective includes non-differentiable or system-level terms such as:

- Closed-loop task success.
- Collision or unsafe-state penalties.
- Action jerk/smoothness from rollout.
- Spike-rate or SynOps proxy.
- Evaluator-level robustness under perturbations.

ES is weakly justified or unnecessary when optimizing only offline action MSE or future latent loss, because Adam/surrogate gradients are simpler and stronger.

## Mandatory warmup

Default path:

1. Train MLP/GRU/SNN with supervised/surrogate gradients.
2. Freeze or partially freeze most parameters.
3. Choose a small adapter/action-head/low-rank parameter subset.
4. Run ES sanity on offline fitness.
5. Run tiny closed-loop ES post-training.
6. Compare against surrogate-only checkpoint.

## Fitness design

Use explicit, logged components:

```text
fitness = success_score
          - alpha_jerk * action_jerk
          - alpha_unsafe * unsafe_penalty
          - alpha_spike * spike_cost
          - alpha_latency * latency_cost_optional
```

Rules:

- Do not mix normalized and raw terms without documenting scales.
- Log every component separately.
- Use identical initial states for parent and perturbation evaluation when possible.
- Average across enough episodes to reduce noise, or clearly label as noisy/sanity.

## Low-rank perturbation discipline

If implementing EGGROLL-style low-rank perturbations:

- State which parameters are perturbed.
- State rank, sigma, population size, and estimator.
- Save perturbation seed.
- Compare against ordinary ES for small modules when feasible.
- Do not claim hyperscale efficiency from a toy adapter experiment.

## Required outputs

Save:

- `es_config.yaml`.
- `initial_checkpoint.txt`.
- `generation_metrics.csv`.
- `population_metrics.csv` if feasible.
- `best_es.pt`.
- `command.txt`.
- `notes.md` explaining noise, compute budget, and whether ES improved the correct target.

## Tests and sanity checks

Before closed-loop ES:

- A toy quadratic or linear synthetic objective should improve.
- Offline ES sanity should not corrupt checkpoint loading/saving.
- Fitness should be deterministic under fixed seed for deterministic mode.
- Sign convention must be tested: higher fitness should mean better model.

## Review output format

Return:

- Whether ES is scientifically justified for this task.
- Parameter subset optimized.
- Fitness components and signs.
- Baseline comparison.
- Compute budget.
- Risks and overclaims to avoid.
