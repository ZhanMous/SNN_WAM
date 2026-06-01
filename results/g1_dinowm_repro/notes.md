# G1 DINO-WM Reproduction Smoke — Notes

## Purpose

Engineering smoke test of the closest existing pipeline to DINO-WM: the G0 patch latent smoke config with dry_run mode. This validates that the training infrastructure can handle patch-latent-shaped tensors [B, T, P, D] and run end-to-end without errors.

## Limitations

- Mock data only (no real LIBERO trajectories)
- Smoke_time_index encoder (not real DINOv2)
- 1 training step only (not convergence)
- No DINO-WM spatial dynamics model implemented
- No planning or evaluation

## Why This Config

The `g0_patch_latent_smoke.yaml` is the only config that exercises the patch latent pipeline (DINOv2PatchEncoder → [B, T, P, D] tensors → WAM-GRU). It is the closest existing infrastructure to the DINO-WM data path, even though it uses synthetic encoders.

## Blocked Items for Real DINO-WM

1. No DINO-WM-style action-conditioned future latent predictor implemented (Phase A task 3)
2. No DINOv2 patch feature extraction on real images (requires facebook/dinov2-small download)
3. No real LIBERO dataset loaded in this run
4. No transition window dataset with explicit [B, T, P, D] patch latents from real data

## Git State

dirty=True due to untracked files (CLAUDE.md, configs/smoke/g0_patch_latent_smoke.yaml, etc.) and modified files (27 tracked files). This is expected for a development branch.
