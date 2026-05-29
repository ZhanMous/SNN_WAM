# H=1 Overfit Repair Diagnostic

Best target shift: `-1`
Nonzero shift clearly best: `True`
Pipeline valid for architecture claims: `True`

## Target Shift Sweep
| Shift | Eval MSE | Continuous MSE | Gripper MSE | Corr | Passed |
|---:|---:|---:|---:|---:|---|
| -1 | 0.00026019136 | 0.00028735684 | 9.719791e-05 | 0.999419 | False |
| +0 | 0.0050164298 | 0.00090367708 | 0.029692946 | 0.988816 | False |
| +1 | 0.0013642302 | 0.00077989523 | 0.0048702401 | 0.997017 | False |
| +2 | 0.002223467 | 0.00088020932 | 0.010283013 | 0.995055 | False |

## Split Head And Baselines
| Variant | Eval MSE | Continuous MSE | Gripper sign acc | Passed |
|---|---:|---:|---:|---|
| wam_gru_split_gripper | 0.00053623639 | 0.00062560925 | 1 | False |
| timestep_embedding_mlp | 3.5308327e-07 | 4.119305e-07 | 1 | True |
| dinov2_latent_mlp | 0.019513838 | 0.015821695 | 0.989583 | False |

## Causal H=1 Baseline Ladder (shift=0 only)
| Variant | Eval MSE | Continuous MSE | Gripper sign acc | Passed |
|---|---:|---:|---:|---|
| causal_timestep_embedding | 7.9198264e-07 | 9.2397971e-07 | 1 | True |
| causal_proprio_only | 0.010794364 | 0.0056489799 | 0.989583 | False |
| causal_action_history_gru | 0.00079169107 | 0.00092363969 | 1 | False |
| causal_dino_cls_only | 0.041538566 | 0.020683885 | 0.958333 | False |
| causal_dino_proprio | 0.024035504 | 0.014152537 | 0.979167 | False |
| causal_dino_proprio_history_gru | 0.001351197 | 0.0015763968 | 1 | False |

Causal baselines that pass: causal_timestep_embedding

## Audit
- Causal contract pass: True
- Latent sanity pass: True
- Nominal target: actions[t+1] for obs/latent at t
- Target scaling: raw_action_units
- Action normalization: none
- Masking: no masks; windows that would cross boundaries are excluded
- Padding: none

No future-latent, closed-loop, or architecture benefit claim follows from this diagnostic.
