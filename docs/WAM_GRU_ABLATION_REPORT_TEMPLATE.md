# WAM-GRU Future-Latent Ablation Report Template

Status: template for smoke/offline ablation only. Do not use this document to
claim closed-loop success, robustness, or real-data WAM improvement.

## Question

Does adding the future latent objective change offline smoke metrics for the
same WAM-GRU architecture and deterministic mock split?

## Runs

| Variant | Config | Artifact ID | Run path | Checkpoint | Eval CSV |
| --- | --- | --- | --- | --- | --- |
| With future latent loss | `configs/libero_spatial_wam_gru.yaml` | `R-G4-WAM-GRU-FUTURE-SMOKE-001` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/best.pt` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/eval_offline.csv` |
| Without future latent loss | `configs/libero_spatial_gru_no_future.yaml` | `R-G4-WAM-GRU-NO-FUTURE-SMOKE-001` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/best.pt` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/eval_offline.csv` |

## Metric Table

| Variant | Split | Action MSE lower is better | Future latent cosine error lower is better | Total loss | Action loss | Future loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| With future latent loss | val | 86.12704032 | 1.168717265 | 87.29575348 | 86.12703705 | 1.168717265 |
| Without future latent loss | val | 86.12520054 | 1.24387157 | 86.12519836 | 86.12519836 | 1.24387157 |

## Current Smoke Instantiation

These rows are engineering smoke outputs from deterministic mock data, seed 0,
and `max_steps=1`. They are included only to verify the report wiring.

| Variant | Artifact ID | Eval CSV | Val action MSE | Val future latent cosine error |
| --- | --- | --- | ---: | ---: |
| With future latent loss | `R-G4-WAM-GRU-FUTURE-SMOKE-001` | `results/smoke/wam_gru_ablation/g4_wam_gru_future_smoke_seed0/eval_offline.csv` | 86.12704032 | 1.168717265 |
| Without future latent loss | `R-G4-WAM-GRU-NO-FUTURE-SMOKE-001` | `results/smoke/wam_gru_ablation/g4_wam_gru_no_future_smoke_seed0/eval_offline.csv` | 86.12520054 | 1.24387157 |

Allowed wording: "The smoke ablation wiring runs and records offline metrics."

Forbidden wording: "Future latent loss improves closed-loop success."

## Claim Audit

| Claim | Category | Evidence | Allowed wording | Forbidden wording |
| --- | --- | --- | --- | --- |
| Both ablation configs run end-to-end in smoke mode. | Supported by current evidence after artifacts exist. | `metrics.csv`, `eval_offline.csv`, `checkpoint.pt`, `config.yaml`, `command.sh`. | "Both WAM-GRU ablation smoke runs completed." | "The future latent objective improves LIBERO performance." |
| Future latent loss improves closed-loop success. | Unsupported. | No rollout CSV or episode success evidence. | "Closed-loop impact is not evaluated yet." | "Future latent loss improves success rate." |
| This is a real-data WAM result. | Unsupported for mock smoke. | Dry-run uses deterministic mock trajectories and smoke latents. | "This is an engineering smoke check." | "This validates WAM on LIBERO." |

## Required Before Reportable Claim

- Real frozen visual latent extraction or precomputed latent artifact metadata.
- Fixed train/val/test split and train-only normalization records.
- Multi-seed offline evaluation.
- Closed-loop rollout CSV with identical initial states before any success or
  robustness claim.
