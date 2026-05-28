# Normalization Policy

Status: policy defined for G2.5. Full real-data normalization integration is not implemented yet.

## Core Rule

All fitted normalization statistics must be computed on the train split only. Validation and test data may be transformed with frozen train statistics, but they must never influence fitted means, standard deviations, min/max values, image statistics, or latent statistics.

## Action Normalization

- Fit per-action-dimension mean and standard deviation on train trajectories only.
- Shape of raw actions: `[T, action_dim]`.
- Save action statistics as:

```json
{
  "actions": {
    "mean": [0.0],
    "std": [1.0],
    "count": 12345,
    "source_split": "train"
  }
}
```

- Offline action MSE/L1 metrics must state whether they are computed in normalized units or de-normalized physical/control units.

## State / Proprio Normalization

- Fit per-dimension mean and standard deviation on train trajectories only.
- Applicable raw shapes: `[T, state_dim]`, including selected proprio fields.
- Simulator `states` are not default model inputs; if used for reset/evaluation metadata, do not normalize them as policy inputs.

## Image Normalization

V1 options:

- Use encoder-native fixed image normalization, such as ImageNet or CLIP statistics, without fitting on SNN-WAM val/test data.
- Or fit train-only image channel statistics if using a simple frozen encoder.

The chosen option must be recorded in config and `normalization_stats.json`.

## Language Handling

- Language strings are task conditions.
- Tokenizer or embedding model must be frozen for Phase 1 unless a later gate explicitly allows otherwise.
- Do not include split labels, success labels, or held-out metadata in text prompts.

## Future Latent Normalization

Future latent targets are not implemented yet. If added later:

- Use the frozen encoder output as the target.
- Fit any latent normalization on train future targets only.
- Save latent stats separately from action/state stats.
- Future latent metrics must state whether latents are normalized.

## Storage

Every training run must save frozen statistics to:

```text
results/runs/<run_id>/normalization_stats.json
```

The file must include source split, trajectory ids, feature names, means, stds, epsilon/clamping policy, and whether image stats were fixed or fitted.

## Evaluation

Evaluation must load train-fitted statistics from the selected run directory. Eval code must fail closed if stats are missing or if their recorded split is not `train`.
