# HyperscaleES Potential Audit: 2026-06-08

## Question

Does `HyperscaleES` have enough potential to support the SNN-WAM goal:

```text
directly train an SNN latent world model without backpropagation,
using an ES/EGGROLL-style optimizer over action-conditioned future patch-latent prediction?
```

## Verdict

`HyperscaleES` is worth keeping as an exploratory optimizer candidate, but it is not yet strong enough to be the main route for SNN-WAM.

The current evidence supports this narrow statement:

```text
HyperscaleES/EggRoll can produce useful non-gradient updates on a simple MLP toy objective,
and its low-rank perturbation interface is conceptually compatible with small adapter/SNN modules.
```

The current evidence does not support these stronger claims:

- Direct ES can train the SNN world model.
- HyperscaleES is likely to work from scratch on SNN-WAM.
- EGGROLL-style optimization should replace surrogate-gradient baselines.
- The current LLM reproduction logs demonstrate useful optimization.

## Evidence Inspected

### Code

- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/src/hyperscalees/noiser/eggroll.py`
- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/src/hyperscalees/noiser/open_es.py`
- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/src/hyperscalees/models/common.py`
- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/tests/end_to_end_test.py`
- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/llm_experiments/general_do_evolution.py`
- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/llm_experiments/utils.py`

### Local Runs / Logs

- `tests/end_to_end_test.py`
- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/experiments/paper_repro/gsm8k_0.1B.log`
- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/experiments/paper_repro/countdownn_0.1B.log`

## What Works

### 1. The optimizer has a real gradient-free update path

`EggRoll` injects deterministic perturbations during forward passes through:

- `do_mm`
- `do_Tmm`
- `get_noisy_standard`

It then converts raw fitness scores with `convert_fitnesses` and applies updates through `do_updates`.

For matrix multiply parameters, `EggRoll` uses low-rank factors:

```text
delta W = A @ B.T
```

This is relevant to SNN-WAM because a small SNN adapter or latent dynamics head can expose matrix-like parameter blocks that could be perturbed in the same way.

### 2. The built-in toy test improves

Command run:

```bash
/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/.venv/bin/python tests/end_to_end_test.py
```

Observed result:

| Epoch | Avg validation score |
|---:|---:|
| 0 | `-3.9733` |
| 1 | `-3.4209` |
| 2 | `-2.8246` |
| 3 | `-2.6759` |
| 4 | `-2.1585` |
| 5 | `-1.9423` |
| 6 | `-1.5549` |
| 7 | `-1.3519` |
| 8 | `-1.0588` |
| 9 | `-0.8156` |

Interpretation:

- The sign convention is at least plausible for a simple objective where higher fitness is better.
- Updates are nonzero.
- The basic noiser/model interface works locally.

This is a necessary but very weak sanity check.

## What Does Not Work Yet

### 1. Current LLM runs are not positive optimization evidence

The local 0.1B RWKV EGGROLL runs show:

| Task | Observation |
|---|---|
| GSM8K | validation score `0.0`; fitness mostly `0.0`; updates mostly `0` |
| Countdown | validation score about `0.01`; fitness mostly flat; `lora_updates` usually `0` |

Interpretation:

- The execution path can run, but the runs do not show meaningful learning.
- This is not evidence that HyperscaleES can optimize SNN-WAM.
- It may reflect sparse reward, poor task fit, low population size, wrong hyperparameters, update freezing, or model/task mismatch.

### 2. The method is not SNN-specific

HyperscaleES knows about parameter trees and matrix multiply operations. It does not know about:

- membrane state resets;
- spike thresholds;
- temporal state carry;
- spike-rate/SynOps penalties;
- batch independence across trajectory windows;
- patch latent shape contracts `[B,T,P,D]`.

Those would need to be supplied by a SNN-WAM wrapper.

### 3. It is poorly justified for pure offline latent MSE

For objectives like:

```text
future_patch_latent_mse
patch_cosine_error
```

gradient/surrogate-gradient training is simpler, lower variance, and likely stronger. ES becomes scientifically interesting only when the objective includes non-differentiable or system-level terms such as:

- closed-loop success;
- robustness under delay/noise/frame drop;
- action smoothness/jerk;
- spike-rate or SynOps proxy;
- evaluator-level failure penalties.

## Fit To SNN-WAM Gates

| Gate | HyperscaleES relevance | Current status |
|---|---|---|
| DWM-G3 real ANN baseline | Not directly relevant | Not passed |
| DWM-G5 SNN forward | Needed before ES | Not implemented |
| DWM-G6 direct ES sanity | Directly relevant | Toy MLP sanity passes; SNN-WAM-specific sanity not run |
| DWM-G7 SNN world model | Direct target | No evidence yet |
| DWM-G8 planning eval | Later possible objective | Blocked until offline gates pass |

## Decision

Use HyperscaleES as a third-stage candidate, not a first-stage solution.

Recommended sequence:

1. First repair DWM-G3 real patch-latent ANN baseline.
2. Implement DWM-G5 SNN forward with reset and spike statistics.
3. Run surrogate-gradient SNN as a reference baseline.
4. Run a tiny HyperscaleES adapter test on a synthetic SNN objective.
5. Only then run offline SNN latent prediction ES.

Do not run full closed-loop HyperscaleES until offline gates pass.

## Minimum Next Validation Experiment

The next experiment should be deliberately small:

```text
Target:
  a tiny LIF/PLIF SNN predicts a synthetic future latent vector.

Parameters optimized:
  only the readout layer, or a low-rank adapter on the recurrent/input matrix.

Fitness:
  - patch_latent_mse
  - alpha_spike * spike_rate_proxy

Required checks:
  fixed-seed determinism
  antithetic pair sanity
  nonzero update norm
  fitness improves over 20-50 generations
  beats random/untrained SNN
  compare OpenES vs EggRoll
```

Pass condition:

```text
EggRoll improves validation fitness over random/untrained SNN
under a fixed seed and does not rely on leakage or batch-state carryover.
```

Fail condition:

```text
No fitness improvement, update norm stays zero, or improvement appears only on the perturbation batch but not held-out validation.
```

## Bottom Line

HyperscaleES has methodological potential for the SNN-WAM goal because its low-rank ES machinery can update small matrix-structured modules without backpropagation, and its toy MLP sanity improves locally. But the current evidence is far from enough: LLM reproduction logs are flat, no SNN wrapper exists, and the real DINO-WM patch-latent baseline has not passed. Treat it as an optimizer candidate to be tested after the SNN forward and surrogate baseline exist, not as proof that the direct-trained SNN world model is feasible.
