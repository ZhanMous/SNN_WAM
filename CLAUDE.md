# CLAUDE.md

## Project Identity

This repository investigates whether a directly trained Spiking Neural Network (SNN) can serve as a latent world model for robotic manipulation, following the DINO-WM paradigm of predicting action-conditioned future DINOv2 spatial patch features.

**Main route:** DINO-WM-style patch latent dynamics → SNN latent dynamics predictor → planning via latent optimization.

**Not the main route:** Direct action behavior cloning (BC) or ANN-to-SNN conversion. These are frozen as legacy diagnostic evidence.

## Current Phase

The project is at **S0 (Route Bootstrap)** of the DINO-WM → SNN-WAM plan. See `docs/DINOWM_SNN_WORLDMODEL_PLAN.md` for the full 4-phase plan and `docs/PROJECT_CONTEXT.md` for current status.

Phase sequence:
1. **Phase A** — Reproduce minimal DINO-WM (ANN/Transformer baseline)
2. **Phase B** — Build direct SNN world model interface
3. **Phase C** — Direct training via ES/EGGROLL-style methods
4. **Phase D** — Planning with learned world model

## Non-Goals

- Do not train a full VLA or foundation model.
- Do not modify LIBERO source unless absolutely necessary.
- Do not claim neuromorphic low power unless measured on neuromorphic hardware.
- Do not claim embodied foundation model results from small LIBERO experiments.
- Do not optimize for paper narrative before reproducible evidence exists.
- Do not run closed-loop experiments until offline gates pass.

## Required Workflow

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

## Scientific Quality Gates

Every experimental result must include ALL of:

- `config.yaml`
- `metrics.csv`
- `checkpoint.pt` (if model trained)
- `git_commit.txt`
- `environment.txt`
- `seeds.txt`
- `command.sh`
- `split.json`
- `notes.md`

No result may be cited in docs or paper unless it appears in `docs/RESULT_ARTIFACTS.md`.

## Claim Status Values

| Status | Meaning |
|---|---|
| `hypothesis` | No result files yet |
| `observation` | Single run or manual observation, insufficient for conclusion |
| `supported` | Result files + re-evaluation path exist |
| `diagnostic_only` | Supports diagnosis only, not scientific claims |
| `unsupported` | Evidence does not support the claim |
| `forbidden` | Must never be claimed given current evidence |

## Closed-Loop Rule

No closed-loop experiment may be launched until ALL offline gates pass:
- DWM-G1 through G4 (Phase A)
- DWM-G5 through G7 (Phase B/C)
- DWM-G8 (Phase D) — closed-loop is gated here

## ES/EGGROLL Role

ES/EGGROLL-style methods are optional **training-method comparisons** after surrogate-gradient baselines are established. They are not the first training method to implement.

## Testing Expectations

Minimum tests before accepting a change:

- Import smoke test
- Dataset shape test (including patch latent shape `[B, T, P, D]`)
- Model forward shape test
- Tiny-batch overfit or one-step train test
- Metric correctness test
- No future leakage test for trajectory windows

## Coding Style

- Prefer small modules.
- Prefer explicit tensor shape comments.
- Never silently squeeze, flatten, or reorder time dimensions.
- All dataloader outputs must document shape as `[B, T, ...]` or `[B, ...]`.
- Patch latent dimensions: `[B, T, P, D]` where P = number of patches, D = feature dim.

## Key Files

| File | Purpose |
|---|---|
| `docs/DINOWM_SNN_WORLDMODEL_PLAN.md` | Full 4-phase route plan |
| `docs/PROJECT_CONTEXT.md` | Current status and architecture |
| `docs/CLAIMS_LEDGER.md` | All registered scientific claims |
| `docs/RESULT_ARTIFACTS.md` | Result artifact registry |
| `docs/STOP_RULES.md` | Stop/retry/pivot decision rules |
| `scripts/quality_gate.sh` | Safe quality checks |
| `configs/smoke/` | Smoke test configs |
| `configs/reportable/` | Reportable experiment configs |
