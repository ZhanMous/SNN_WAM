---
name: snn-adapter-implementation
description: Use when implementing or reviewing LIF/ALIF/PLIF SNN temporal adapters, surrogate-gradient training, spike states, membrane resets, spike-rate logging, or replacing GRU with SNN in the SNN-WAM project.
---

# SNN Adapter Implementation

The SNN must be a temporal/world-action adapter, not a renamed GRU and not a direct torque controller.

## Role of the SNN

The SNN receives fused latent features and action history across a sequence window, maintains internal membrane/spike state, and produces:

- Future action chunk.
- Future visual latent chunk.
- Spike statistics.

## Required interface

Prefer an interface like:

```python
outputs = model(batch)
# outputs keys:
# action_pred: [B, action_horizon, action_dim]
# future_latent_pred: [B, future_horizon, latent_dim]
# spike_stats: dict[str, Tensor or float]
# state_debug: optional dict for membrane/spike traces
```

SNN modules should accept explicit sequence inputs:

```python
x_seq: [B, T, D]
```

Avoid hidden transposes. If a library expects `[T, B, D]`, convert in one clearly named place.

## State handling rules

- Reset membrane state at sample/episode boundaries unless explicitly doing streaming evaluation.
- Do not carry state across unrelated trajectories in a training batch.
- For closed-loop rollout, define whether SNN state persists across environment steps within an episode.
- Provide `reset_state()` or equivalent for policy evaluation.

## Surrogate gradient rules

- Document surrogate function and parameters.
- Log spike rate and loss separately.
- Use gradient clipping if spikes cause instability.
- Keep a non-spiking baseline with similar parameter budget where possible.

## SNN value checks

Do not claim SNN is useful unless at least one evidence channel supports it:

- Comparable or better closed-loop success than GRU.
- Better robustness under delay/frame drop/noise.
- Slower future-latent degradation across horizon.
- Better spike-rate/success trade-off.
- Distinct event-triggered dynamics visible in debug traces.

## Required tests

Add or update tests for:

- Forward shape.
- Finite loss.
- Nonzero gradients through surrogate training.
- Spike rate in `[0, 1]` when defined as mean binary spike rate.
- State reset changes behavior only where expected.
- Batch independence: state from one sample does not leak into another.

## Debug outputs

For early experiments, save optional debug tensors or summaries:

- Mean spike rate per layer.
- Mean membrane potential per layer.
- Fraction of silent neurons.
- Spike raster for a small batch if practical.

## Forbidden claims

- Do not say “low-power” without hardware measurement.
- Do not say “biologically realistic” unless the exact biological assumption is named.
- Do not say “better than RNN” from offline MSE alone.
