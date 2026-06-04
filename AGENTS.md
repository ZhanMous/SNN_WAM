# AGENTS.md

## Project identity

This repository investigates whether a directly trained Spiking Neural Network (SNN) can serve as a latent world model for robotic manipulation, following the DINO-WM paradigm.

**Main route:** DINO-WM-style patch latent dynamics → SNN latent dynamics predictor → planning via latent optimization.

**Not the main route:** Direct action behavior cloning (BC) or ANN-to-SNN conversion. These are frozen as legacy diagnostic evidence.

The primary scientific question is:

Can a directly trained SNN latent world model predict action-conditioned future DINOv2 spatial patch features well enough to support planning, without relying on surrogate-gradient training or ANN-to-SNN conversion?

ES/EGGROLL-style methods are optional training-method comparisons after surrogate-gradient baselines are established.

## DWM new-route files (allowed)

The following source files implement the DINO-WM → SNN-WAM route and are
explicitly allowed in the repository. Do not delete or rename without
updating this list.

**Data:**
- `src/data/patch_latent_dataset.py` — PatchLatentTransitionDataset, future_actions [B,H,A]
- `src/data/trajectory_window.py` — Trajectory windowing

**Models:**
- `src/models/dinowm_transformer.py` — DINOwMTransformer (action-conditioned patch latent dynamics)

**Planning:**
- `src/planning/action_optimizer.py` — Gradient/CMA-ES action optimization

**Eval:**
- `src/eval/dinowm_eval_offline.py` — Multi-horizon offline eval
- `src/eval/dwm_g4_planning_sanity.py` — Planning sanity evaluation

**Scripts:**
- `scripts/train_dinowm_baseline.py` — DINO-WM baseline training

**Configs:**
- `configs/reportable/dinowm_baseline_real.yaml`

**Tests:**
- `tests/test_dwm_g1_patch_features.py`
- `tests/test_dwm_g2_transition_dataset.py`
- `tests/test_dwm_g3_baseline.py`
- `tests/test_dwm_g4_planning_sanity.py`

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

## DWM acceptance gates

**DWM-G3 (baseline):** ANN Transformer baseline must:
- Beat copy-last/persistence at H=1 on patch_cosine_error
- Beat persistence on multi-step (H=2, H=4) metrics
- True future actions明显优于 shuffled future actions (ablation)
- Run on ≥3 seeds with clean git state (`dirty=False`)

**DWM-G4 (planning):** Planning sanity must:
- Optimized actions beat random/shuffled baselines on >50% of samples
- Planning works with explicit future_actions [B, H, A] interface

**DWM-G5 entry (SNN):** Only after G3/G4 pass:
- Minimal LIF-SNN forward pass (replace GRU temporal adapter)
- Membrane potential reset between samples
- Spike rate / SynOps logging
- Surrogate-gradient training baseline

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
