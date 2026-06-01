# H=1 Overfit Repair Diagnostic

Best target shift: `-1`
Nonzero shift clearly best: `True`
Pipeline valid for architecture claims: `False`

## Target Shift Sweep
| Shift | Eval MSE | Continuous MSE | Gripper MSE | Corr | Passed |
|---:|---:|---:|---:|---:|---|
| -1 | 6.6282104e-05 | 7.2621158e-05 | 2.8247799e-05 | 0.999854 | True |
| +0 | 0.00010311337 | 9.6772012e-05 | 0.00014116155 | 0.999773 | False |
| +1 | 0.00014575946 | 0.00013973269 | 0.00018191994 | 0.99969 | False |
| +2 | 0.00016781897 | 0.00015858951 | 0.00022319584 | 0.999625 | False |

## Split Head And Baselines
| Variant | Eval MSE | Continuous MSE | Gripper sign acc | Passed |
|---|---:|---:|---:|---|
| wam_gru_split_gripper | 0.00022818052 | 0.00026621061 | 1 | False |
| timestep_embedding_mlp | 5.3567575e-08 | 6.2495516e-08 | 1 | True |
| dinov2_latent_mlp | 0.0028178946 | 0.0032875435 | 1 | False |

## Audit
- Nominal target: actions[t+1] for obs/latent at t
- Target scaling: raw_action_units
- Action normalization: none
- Masking: no masks; windows that would cross boundaries are excluded
- Padding: none

No future-latent, closed-loop, or architecture benefit claim follows from this diagnostic.
