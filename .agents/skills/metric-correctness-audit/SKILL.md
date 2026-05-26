---
name: metric-correctness-audit
description: Use when implementing or reviewing action MSE/L1, future latent cosine error, multi-step degradation, spike rate, SynOps proxy, inference latency, rollout success, robustness metrics, or tables/plots derived from metrics.
---

# Metric Correctness Audit

Metrics are scientific claims in code. Treat every metric as a theorem with assumptions, inputs, masks, and tests.

## General metric rules

Every metric function must specify:

- Input tensor shapes.
- Expected dtype and device behavior.
- Reduction dimensions.
- Whether masks are supported.
- Whether lower or higher is better.
- Whether the metric is offline, model-level, or closed-loop.

No metric may silently average across horizon, batch, task, and seed unless the function name or docs say so.

## Required metrics and definitions

### Action MSE / L1

- Compare predicted action chunk against demonstration action chunk.
- Shape: `[B, H, A]` or `[N, H, A]`.
- Report at least global mean and optionally per-horizon curve.
- If actions are normalized, record whether the metric is in normalized or physical units.

### Future latent cosine error

- Definition: `1 - cosine_similarity(z_hat, z_target)`.
- Shape: `[B, H, D]`.
- Normalize over latent dimension only.
- Report per-horizon and overall mean.
- Do not compare predicted latent to current latent unless that is an explicit baseline.

### Multi-step degradation

- Compute metric separately for each horizon step.
- Plot or save `horizon, metric_mean, metric_std_or_sem`.
- Do not let shorter sequences bias later horizons without masks.

### Spike rate

- Definition must state whether it is averaged over batch, time, layer, and neurons.
- Report as a proxy only, not hardware energy.
- If multiple SNN layers exist, consider per-layer spike rate.

### SynOps proxy

- If implemented, document the approximation.
- Do not call it energy unless target hardware and measurement method are given.

### Inference latency

- Report batch size, device, warmup runs, measured runs, and whether synchronization was used for CUDA.
- Do not compare GPU and CPU latencies without explicit labeling.

### Closed-loop success rate

- Unit is episode.
- Save per-episode CSV with task id/name, initial-state id, seed, success, steps, failure reason when available.
- Report number of episodes and confidence interval or seed-level variance when possible.

### Robustness metrics

For noise/delay/frame drop:

- Specify perturbation level.
- Use identical seeds/initial states across models.
- Save clean baseline in the same table.
- Avoid cherry-picking only favorable perturbations.

## Synthetic tests

Add deterministic tests for:

- Perfect prediction gives zero action MSE and zero future latent cosine error.
- Orthogonal latent vectors give cosine error near 1.
- Opposite latent vectors give cosine error near 2 if using raw cosine error.
- Masked padded horizons do not affect results.
- Spike rate on all-zero spikes is 0 and all-one spikes is 1.

## Table and plot audit

When generating figures/tables:

- Verify source CSV path.
- Verify split and checkpoint selection.
- Verify model names and seeds.
- Verify arrow direction: lower-is-better or higher-is-better.
- Store generation command.

## Review output format

Return:

- PASS / FAIL / PASS WITH RISKS.
- Metric definitions confirmed.
- Shape assumptions.
- Test coverage.
- Any misleading aggregation.
- Any unsupported table/plot claim.
