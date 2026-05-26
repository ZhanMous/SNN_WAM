---
name: scientific-claim-audit
description: Use when reviewing paper drafts, README claims, weekly reports, introductions, contribution lists, result discussions, or any statement about SNN, WAM, VLA, EGGROLL, robustness, efficiency, or biological plausibility.
---

# Scientific Claim Audit

This skill prevents overclaiming. Every scientific sentence must be supported by code, experiment, or a clearly marked hypothesis.

## Claim categories

Classify every important claim as one of:

- Supported by current evidence.
- Preliminary but plausible.
- Hypothesis / motivation only.
- Unsupported.
- Contradicted by current evidence.
- Too broad for current experiment.

## Claims that require special caution

### “SNN is better than GRU”

Requires fair comparison on identical data, split, model budget where possible, and at least one meaningful metric. Offline MSE alone is insufficient if the claimed advantage is embodied control.

Safer wording:

- “SNN shows stronger robustness under X perturbation in Y setting.”
- “SNN matches GRU success while using lower spike-rate proxy.”

### “This is WAM”

Requires explicit future-state or future-latent prediction, not only action prediction.

Safer wording:

- “A minimal WAM-style adapter that predicts action-conditioned future latent state.”

### “This is VLA/foundation model”

Small adapters on LIBERO are not enough for a foundation-model claim.

Safer wording:

- “A temporal adapter for language-conditioned manipulation policies.”

### “Low power / energy efficient”

Requires neuromorphic hardware or explicit energy model. Spike rate is only a proxy.

Safer wording:

- “Lower spike-rate/SynOps proxy under the measured software setting.”

### “EGGROLL-style”

Do not claim full EGGROLL reproduction unless the specific algorithmic and scale conditions are reproduced.

Safer wording:

- “Low-rank ES post-training inspired by EGGROLL-style black-box optimization.”

## Evidence mapping

For every claim in docs/paper/report, attach:

```text
Claim:
Evidence artifact:
Metric/table/figure:
Commit:
Limitations:
Allowed wording:
Forbidden wording:
```

Prefer maintaining `docs/CLAIMS_LEDGER.md`.

## Reviewer simulation

Ask these questions:

1. Why not GRU/LSTM?
2. Is future latent prediction actually improving rollout success?
3. Is SNN only adding complexity?
4. Are improvements due to parameter count or tuning budget?
5. Is closed-loop evaluation fair and sufficiently sampled?
6. Does ES improve system-level fitness or just add noise?
7. Are biological or energy claims justified?

## Output format

Return:

- Claim audit table.
- Required wording changes.
- Unsupported claims to delete.
- Missing experiments needed to support stronger claims.
- Reviewer questions likely to arise.
