# DINOwM Transformer Baseline

Standalone training on cached DINOv2 patch latents.
Model: DINOwMTransformer (patch projection + action projection + Transformer encoder).
Primary loss: patch_cosine_error (1 - cosine similarity, averaged over patches).
No action prediction head -- this is a pure world model.

## Known limitations
- No SNN adapter yet (ANN-only baseline)
- No closed-loop evaluation
- Action sequences are inputs, not predicted
