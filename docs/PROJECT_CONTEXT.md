# Project Context

## Current Route

**DINO-WM → Directly Trained SNN World Model**

The previous direct action behavior cloning route is frozen as diagnostic evidence (branch tag: `legacy_bc_policy_diagnostics_20260601`). The new route begins from DINO-WM-style latent world modeling.

## Scientific Question

Can a directly trained SNN latent world model predict action-conditioned future DINOv2 spatial patch features well enough to support planning, without relying on surrogate-gradient training or ANN-to-SNN conversion?

This is a hypothesis, not a current result.

## Architecture

```
o_t (observation)
  → frozen DINOv2 encoder
  → spatial patch features z_t: [B, P, D]

z_context, future action sequence a_t:t+h
  → world model (SNN)
  → predicted future patch features z_hat_t+1:t+h: [B, H, P, D]
```

Key design choices:
- DINOv2 ViT-S/14 frozen encoder produces spatial patch features, not only CLS token
- World model operates on patch latent space `[P, D]`, not raw pixels
- SNN is the world model itself, not a policy head or post-hoc adapter
- Planning optimizes explicit future action sequences `[B, H, A]` through the learned world model

## Phase Status

| Phase | Name | Status | Gates |
|---|---|---|---|
| A | Reproduce Minimal DINO-WM | **In progress** | DWM-G1 ✅, DWM-G2 ✅, DWM-G3 ✅, DWM-G4 pending |
| B | SNN World Model Interface | Not started | DWM-G5 |
| C | Direct ES/EGGROLL Training | Not started | DWM-G6, DWM-G7 |
| D | Planning | Not started | DWM-G8 |

## Evidence Gates

| Gate | Required Evidence | Status |
|---|---|---|
| DWM-G1 patch features | DINOv2 patch tensor shape tests, frame/patch indexing tests | **PASS** (18 tests, all pass) |
| DWM-G2 transition dataset | no-future-leakage tests for z_context, future_actions, z_target | **PASS** (12 tests, all pass) |
| DWM-G3 ANN baseline | forward/metric tests with explicit future_actions `[B,H,A]` | **PASS** (15 tests, all pass; real-data acceptance still pending) |
| DWM-G4 planning sanity | action optimization improves predicted target latent distance | Pending |
| DWM-G5 SNN forward | SNN patch-latent forward shape, reset behavior, spike stats |
| DWM-G6 direct ES sanity | toy objective and offline latent objective improve under fixed seed |
| DWM-G7 SNN world model | direct-trained SNN beats copy-last and random SNN on latent prediction |
| DWM-G8 planning eval | fixed-task planning results with episode CSV and failure cases |

## Legacy Route Summary

The frozen BC route provided these diagnostic findings:
- Expert replay succeeds in LIBERO closed-loop evaluator: 27/30
- WAM-GRU future and no-future both score 0/30 under matched closed-loop diagnostic
- Future latent loss improves future latent error but no observed action or closed-loop improvement
- Residual action targets improve offline metrics but fail autoregressive readiness gate
- All causal input representations fail H=1 single-demo overfit gate

These findings are documented in `docs/CLAIMS_LEDGER.md` (C-G5 through C-G11 claims).

## ES/EGGROLL Role

ES/EGGROLL-style training methods are optional **training-method comparisons** after surrogate-gradient baselines are established. They are not the first training method to implement. The sequence is:

1. ANN/Transformer baseline with gradient descent (Phase A)
2. SNN forward pass validated (Phase B)
3. Surrogate gradient training as baseline (Phase C)
4. ES/EGGROLL comparison (Phase C, optional)

## No Closed-Loop Until Offline Gates Pass

No closed-loop experiment may be launched until ALL offline gates (DWM-G1 through DWM-G7) pass. DWM-G8 is the first gate that includes closed-loop evaluation.
