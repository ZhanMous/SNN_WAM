# AGENTS.md

## Project identity

This repository is scoped to implement a minimal Spiking World-Action Adapter for language-conditioned robot manipulation.

The first scientific question is:

Can an SNN temporal adapter improve future latent/action modeling and closed-loop robustness compared with MLP/GRU baselines?

## Non-goals

- Do not train a full VLA or foundation model in phase 1.
- Do not modify LIBERO source unless absolutely necessary.
- Do not claim neuromorphic low power unless measured on neuromorphic hardware.
- Do not claim embodied foundation model results from small LIBERO experiments.
- Do not optimize for paper narrative before reproducible evidence exists.

## Required workflow

For every non-trivial task:

1. Start with a plan.
2. Identify files to inspect before editing.
3. Make the smallest coherent change.
4. Add or update tests.
5. Run relevant tests.
6. Update docs if behavior changes.
7. Report:
   - files changed
   - commands run
   - tests passed/failed
   - remaining risks

## Scientific quality gates

Every experimental result must include:

- config.yaml
- metrics.csv
- checkpoint.pt if applicable
- git_commit.txt
- environment info
- random seed
- command used
- evaluation split
- notes.md

No result may be cited in docs or paper unless it appears in docs/RESULT_ARTIFACTS.md.

## Testing expectations

Minimum tests before accepting a change:

- import smoke test
- dataset shape test
- model forward shape test
- tiny-batch overfit or one-step train test
- metric correctness test
- no future leakage test for trajectory windows

## Coding style

Prefer small modules.
Prefer explicit tensor shape comments.
Never silently squeeze, flatten, or reorder time dimensions.
All dataloader outputs must document shape as [B, T, ...] or [B, ...].
