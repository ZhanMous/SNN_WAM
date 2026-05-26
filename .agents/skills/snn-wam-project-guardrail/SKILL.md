---
name: snn-wam-project-guardrail
description: Use at the start of any SNN-WAM project task to enforce phase scope, non-goals, evidence discipline, and small closed-loop Codex work units. Trigger for repo scaffolding, planning, feature implementation, audits, and report writing in this project.
---

# SNN-WAM Project Guardrail

You are working on a research codebase for a minimal Spiking World-Action Adapter. The project is not a generic robotics demo and not a full VLA/WAM foundation-model training project.

## Core scientific question

Can an SNN temporal/world-action adapter, placed after frozen visual/language representations, model future latent state and actions more robustly than MLP/GRU baselines under language-conditioned robot manipulation tasks?

## Phase-1 scope

Allowed in phase 1:

- LIBERO demonstration data.
- Trajectory-window dataset.
- Frozen visual/language encoders or simple placeholder encoders.
- MLP/GRU baselines.
- Future action and future latent prediction.
- LIBERO closed-loop rollout evaluation.
- LIF/ALIF/PLIF SNN temporal adapter.
- Robustness tests: noise, delay, frame drop, horizon.
- Small ES post-training only after surrogate-gradient warmup.

Not allowed unless explicitly requested by the user:

- Full OpenVLA 7B training.
- Real Unitree hardware experiments.
- Full-scale EGGROLL reproduction.
- Claims of hardware energy efficiency.
- Claims that a small adapter is a complete embodied foundation model.
- Long GPU jobs without a dry-run script and explicit user approval.

## Required Codex task loop

For every implementation task:

1. Restate the goal in one sentence.
2. List files to inspect.
3. Identify likely scientific failure modes.
4. Make the smallest coherent change.
5. Add or update tests.
6. Run tests or smoke checks.
7. Update documentation or result ledgers if behavior changes.
8. Report exact files changed, commands run, pass/fail status, and unresolved risks.

## Default quality gates

Do not mark a task done unless at least one of these is true:

- The requested tests pass.
- A smoke test passes.
- You clearly state that the task is implementation-only and no executable test is possible yet.

For research changes, prefer these gates:

- Dataset shape test.
- No-future-leakage synthetic test.
- Model forward shape test.
- Tiny-batch training or one-step loss test.
- Metric correctness test using synthetic arrays.
- Reproducibility artifact check.

## Forbidden completion language

Do not say “done”, “complete”, “ready for paper”, or “validated” unless there is evidence. Use “implemented but not fully validated” when tests or experiments have not run.

## Scientific wording discipline

Use narrow wording:

- “Adapter-level evidence” rather than “foundation model result”.
- “Spike rate / SynOps proxy” rather than “low power”.
- “Closed-loop success on specified LIBERO tasks” rather than “robotic intelligence”.
- “EGGROLL-style low-rank ES post-training” rather than “reproduced EGGROLL”.
