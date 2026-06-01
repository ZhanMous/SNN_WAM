# DINO-WM to Directly Trained SNN World Model Plan

## 1. Motivation

The previous direct action BC route is frozen as diagnostic evidence. The issue is not just insufficient tuning: direct action behavior cloning did not provide a stable base for testing WAM, future-latent learning, or SNN temporal dynamics.

Key evidence from the legacy route:

- Expert replay succeeds in the LIBERO closed-loop evaluator: `27/30`.
- WAM-GRU future and no-future both score `0/30` under matched closed-loop diagnostic evaluation.
- Future latent loss improves future latent error but does not produce observed action or closed-loop improvement.
- Residual action targets improve offline metrics but still fail the autoregressive readiness gate.

The new route starts from DINO-WM because it treats world modeling as latent visual dynamics prediction, using DINOv2 spatial patch features and planning by optimizing action sequences through the learned world model.

References:

- DINO-WM paper: https://arxiv.org/abs/2411.04983
- DINO-WM project page: https://dino-wm.github.io/

## 2. Scientific Question

Can a directly trained SNN latent world model predict action-conditioned future DINOv2 spatial patch features well enough to support planning, without relying on surrogate-gradient training or ANN-to-SNN conversion?

This is a hypothesis, not a current result.

## 3. New Model Target

The new target is not direct action prediction.

```text
o_t
  -> frozen DINOv2 encoder
  -> spatial patch features z_t: [P, D]

z_t, action sequence a_t:t+k
  -> world model
  -> predicted future patch features z_hat_t+1:t+h
```

The world model should preserve spatial patch structure rather than collapse everything into a single CLS vector.

## 4. Route

### Phase A: Reproduce Minimal DINO-WM

Goal: implement or reproduce the smallest DINO-WM-style loop before adding SNN.

Tasks:

1. Extract DINOv2 patch features, not only CLS features.
2. Build transition windows:
   - current patch features `z_t`
   - action sequence `a_t:t+k`
   - target future patch features `z_t+1:t+h`
3. Train an ANN/Transformer world model baseline with gradient descent.
4. Evaluate:
   - one-step patch latent error
   - multi-step patch latent drift
   - nearest-neighbor future-frame retrieval
   - whether optimized actions move predicted latent toward target latent

Pass condition:

- The baseline reproduces the basic DINO-WM mechanism on a small controlled dataset or LIBERO subset.

Non-claims:

- No SNN claim.
- No EGGROLL claim.
- No closed-loop robot success claim unless explicitly evaluated.

### Phase B: Build Direct SNN World Model Interface

Goal: make the SNN the world model, not a policy head and not a post-hoc adapter.

Interface:

```text
input:
  patch latent sequence: [B, T, P, D]
  action sequence: [B, T, A]

output:
  future patch latent prediction: [B, H, P, D]
  spike_stats
  membrane/spike debug traces
```

SNN requirements:

- Explicit membrane/spike state.
- Reset policy documented per trajectory/window.
- Spike rate logged as a proxy only.
- No low-power or biological realism claim without direct evidence.

### Phase C: Direct Training Without Surrogate Gradient

Goal: test whether low-rank ES / EGGROLL-style optimization can directly train the SNN world model.

Training objective:

```text
fitness = - future_patch_latent_error
          - alpha_drift * multi_step_drift
          - alpha_spike * spike_rate_proxy
          - alpha_smooth * prediction_smoothness_optional
```

Training method:

- Randomly initialize SNN world model.
- Select a small parameter subset or low-rank parameterization.
- Sample perturbations with fixed seeds.
- Evaluate fitness on recorded trajectories.
- Update by ES estimator.

Required controls:

- Same SNN architecture trained with surrogate gradient, if allowed as a baseline only.
- ANN/Transformer DINO-WM-style predictor trained with gradients.
- GRU latent world model baseline.
- Random/untrained SNN baseline.
- Copy-last-latent baseline.

Important distinction:

- The project goal is direct SNN world-model training.
- Surrogate gradient or ANN baseline may be used for comparison, but not as the source of the SNN model.
- ANN-to-SNN conversion is out of scope for the main claim.

### Phase D: Planning

Planning is added only after the latent world model has credible prediction quality.

Planning procedure:

1. Given current observation and target observation, extract DINOv2 patch features.
2. Optimize candidate action sequences through the learned world model.
3. Score candidates by predicted distance to target patch latent.
4. Execute the first action or action chunk in MPC style.

Planning metrics:

- planning objective reduction
- predicted vs actual next latent agreement
- closed-loop success on fixed tasks and initial states
- action smoothness / jerk
- failure mode taxonomy

## 5. Evidence Gates

| Gate | Required Evidence |
|---|---|
| DWM-G1 patch features | DINOv2 patch tensor shape tests, frame/patch indexing tests |
| DWM-G2 transition dataset | no-future-leakage tests for `z_t`, actions, `z_t+h` |
| DWM-G3 ANN baseline | one-step and multi-step patch latent metrics |
| DWM-G4 planning sanity | action optimization improves predicted target latent distance |
| DWM-G5 SNN forward | SNN patch-latent forward shape, reset behavior, spike stats |
| DWM-G6 direct ES sanity | toy objective and offline latent objective improve under fixed seed |
| DWM-G7 SNN world model | direct-trained SNN beats copy-last and random SNN on latent prediction |
| DWM-G8 planning eval | fixed-task planning results with episode CSV and failure cases |

## 6. Claims Allowed Now

Allowed:

- The old direct action BC route is frozen as diagnostic evidence.
- The new route is a plan to reproduce DINO-WM-style latent dynamics before building an SNN world model.
- Direct SNN world-model training is the intended research target.

Not allowed yet:

- DINO-WM has been reproduced in this repository.
- Direct ES trains SNN world models successfully.
- SNN improves planning or robustness.
- The method is low-power.
- The method is a full VLA or foundation model.

## 7. Immediate Next Steps

1. Add DINOv2 patch feature extraction support alongside the existing CLS extractor.
2. Create a patch-latent transition dataset with explicit shapes:
   - `z_context`: `[B, T, P, D]`
   - `actions`: `[B, T, A]`
   - `z_target`: `[B, H, P, D]`
3. Add no-future-leakage tests for patch-latent windows.
4. Implement a small DINO-WM-style ANN predictor baseline.
5. Only after the baseline works, implement a directly trained SNN world model and ES trainer.
