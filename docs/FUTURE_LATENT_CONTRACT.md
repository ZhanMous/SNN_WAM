# Future Latent Contract

Status: target contract defined. Future latents are not implemented yet.

## Allowed Encoder Choices For V1

Use one frozen visual encoder family:

- Frozen ResNet-style image encoder.
- Frozen CLIP image encoder.
- Simple frozen image encoder for smoke tests.

The encoder must be selected in config and recorded in result artifacts. No visual encoder fine-tuning is allowed in Phase 1 unless a later gate explicitly changes the policy.

## Input And Target Frame Indices

For current processed LIBERO observation index `t` under `action_to_current_obs` semantics:

- Model input image: `image_t = images[t]`.
- Future latent target frames: `images[t+1:t+1+future_horizon]`.
- Future image targets, if used for debugging: `target_future_images = images[t+1:t+1+future_horizon]`.

No future frame or future latent may appear in `input_keys`.

## Shape Contract

Unbatched:

- `image_t`: `[H, W, C]` before transform or `[C, H, W]` after documented transform.
- `target_future_latents`: `[future_horizon, latent_dim]`.
- Optional `future_latent_mask`: `[future_horizon]`.

Batched:

- `image_t`: `[B, C, H, W]` after collate/transform.
- `target_future_latents`: `[B, future_horizon, latent_dim]`.
- Optional `future_latent_mask`: `[B, future_horizon]`.

## Precompute Or On-The-Fly

Both modes are allowed if reproducible:

- Precomputed latents: save encoder id, checkpoint/hash, transform config, command, split, and source image indices.
- On-the-fly latents: load a frozen encoder and record the exact encoder config/checkpoint in the run config.

Precomputed latent files must include enough metadata to verify that val/test targets were not used to fit train normalization.

## Leakage Prevention

- `target_future_latents` is a target key only.
- Any model input builder must consume only `input_keys`.
- Tests must fail if `target_future_latents`, `target_future_images`, or future frame refs are accepted as inputs.
- Future latent targets start at `t+1`, never at `t`.

## Required Tests Before WAM-Style Claim

- Synthetic time-index test proving future latent target indices are `[t+1, ...]`.
- Test proving `target_future_latents` is absent from model input dictionaries.
- Shape test for `[B, future_horizon, latent_dim]`.
- Metric test for future latent cosine error with perfect, orthogonal, opposite, and masked cases.
- Reproducibility test verifying latent metadata includes encoder id, transform, command, split, and source image indices.

## Claim Policy

Until target future latents are implemented and tested, only this wording is allowed:

- "Future-latent target contract is specified but not implemented."

Forbidden:

- "WAM-style future prediction is validated."
- "Future latent objective improves robustness."
- "SNN-WAM learns a world model."
