# AGENTS.md

## Project identity

This repository investigates whether a directly trained Spiking Neural Network (SNN) can serve as a latent world model for robotic manipulation, following the DINO-WM paradigm.

**Main route:** DINO-WM-style patch latent dynamics → SNN latent dynamics predictor → planning via latent optimization.

**Not the main route:** Direct action behavior cloning (BC) or ANN-to-SNN conversion. These are frozen as legacy diagnostic evidence.

The primary scientific question is:

Can a directly trained SNN latent world model predict action-conditioned future DINOv2 spatial patch features well enough to support planning, without relying on surrogate-gradient training or ANN-to-SNN conversion?

ES/EGGROLL-style methods are optional training-method comparisons after surrogate-gradient baselines are established.

## Non-goals

- Do not train a full VLA or foundation model.
- Do not modify LIBERO source unless absolutely necessary.
- Do not claim neuromorphic low power unless measured on neuromorphic hardware.
- Do not claim embodied foundation model results from small LIBERO experiments.
- Do not optimize for paper narrative before reproducible evidence exists.
- Do not run closed-loop experiments until offline gates pass.

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

Every experimental result must include ALL of:

- config.yaml
- metrics.csv
- checkpoint.pt (if model trained)
- git_commit.txt
- environment.txt
- seeds.txt
- command.sh
- split.json
- notes.md

No result may be cited in docs or paper unless it appears in docs/RESULT_ARTIFACTS.md.

## Claim status values

| Status | Meaning |
|---|---|
| `hypothesis` | No result files yet |
| `observation` | Single run or manual observation, insufficient for conclusion |
| `supported` | Result files + re-evaluation path exist |
| `diagnostic_only` | Supports diagnosis only, not scientific claims |
| `unsupported` | Evidence does not support the claim |
| `forbidden` | Must never be claimed given current evidence |

## Testing expectations

Minimum tests before accepting a change:

- import smoke test
- dataset shape test
- model forward shape test
- tiny-batch overfit or one-step train test
- metric correctness test
- no future leakage test for trajectory windows

## Coding style

- Prefer small modules.
- Prefer explicit tensor shape comments.
- Never silently squeeze, flatten, or reorder time dimensions.
- All dataloader outputs must document shape as `[B, T, ...]` or `[B, ...]`.
- Patch latent dimensions: `[B, T, P, D]` where P = number of patches, D = feature dim.
