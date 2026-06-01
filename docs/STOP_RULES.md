# Stop Rules

Decision framework for when to stop, retry, or pivot during the DINO-WM → SNN-WAM route.

## Decision Template

When a gate fails or produces unexpected results, use this template:

```
Gate: DWM-GX
Result: [pass/fail]
Evidence: [files and metrics]
Decision: [stop/retry/pivot]
Rationale: [why]
Next step: [concrete action]
```

## Stop Conditions (Hard Stop)

Stop the current phase and escalate if:

1. **Reproducibility failure**: The same config + seed + data produces different results across 2 runs.
2. **Data corruption**: Transition dataset leaks future information (DWM-G2 fail).
3. **Encoder failure**: DINOv2 patch features are NaN, constant, or degenerate (DWM-G1 fail after 2 attempts).
4. **No learning signal**: Training loss does not decrease over 100 steps on a toy objective (DWM-G6 fail).
5. **Budget exhaustion**: More than 3 attempts at the same gate without progress.

## Retry Conditions

Retry the current gate (up to 3 attempts) if:

1. **Hyperparameter sensitivity**: Result is within 2x of the threshold — adjust learning rate, hidden dim, or sequence length.
2. **Infrastructure bug**: The failure is in data loading, shape mismatch, or missing file — fix and retry.
3. **Numerical instability**: NaN or Inf in output — add gradient clipping, reduce learning rate, or check input normalization.
4. **Incomplete evaluation**: Metrics not logged or wrong format — fix logging, re-run.

## Pivot Conditions

Pivot to an alternative approach if:

1. **ANN baseline fails DWM-G3**: The DINO-WM-style ANN predictor cannot learn patch dynamics on LIBERO. Pivot to simpler visual features or smaller patch resolution.
2. **SNN forward fails DWM-G5**: SNN cannot process patch latents without gradient issues. Consider surrogate gradient as a necessary intermediate step (document why).
3. **ES fails DWM-G6/DWM-G7**: Direct ES training cannot optimize the SNN world model. Pivot to surrogate gradient as the primary training method and reframe the scientific question.
4. **Planning fails DWM-G8**: Latent world model quality is insufficient for planning. Consider reducing planning horizon or switching to closed-form action selection.

## Pivot Documentation

When pivoting, update:

1. `docs/DINOWM_SNN_WORLDMODEL_PLAN.md` — add pivot note with date and rationale
2. `docs/CLAIMS_LEDGER.md` — mark failed gate claims as `unsupported`
3. `docs/PROJECT_CONTEXT.md` — update phase status
4. `docs/STOP_RULES.md` — add the pivot to the decision log below

## Decision Log

| Date | Gate | Decision | Rationale | Next Step |
|---|---|---|---|---|
| 2026-06-01 | — | Route reset | Legacy BC route frozen; new DINO-WM route adopted | Begin Phase A |

## Legacy Route Stop History

The legacy BC route was stopped after G11:

- **Gate**: Offline closed-loop readiness (C-G11-GATE-001)
- **Result**: Not passed — error growth not fully resolved, gripper accuracy drops, held-out demo MSE high
- **Decision**: Stop and pivot to DINO-WM route
- **Rationale**: The BC route's core issue is not tuning but architecture — direct action prediction does not provide a stable base for testing WAM, future-latent learning, or SNN temporal dynamics
- **Evidence**: `docs/EXPERIMENT_REFLECTION_REPORT_2026-06-01.md`
