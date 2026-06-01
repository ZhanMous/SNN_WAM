# G7 Threshold Audit

## Candidate Threshold: 0.0001

## Action Scale
- Mean absolute action: 0.300362
- Threshold as % of action scale: 0.03%

## Per-Dimension Variance

| Dim | Std | Variance |
|---:|---:|---:|
| 0 (delta_pos_x) | 0.319403 | 1.020180e-01 |
| 1 (delta_pos_y) | 0.253025 | 6.402177e-02 |
| 2 (delta_pos_z) | 0.514066 | 2.642636e-01 |
| 3 (delta_rot_x) | 0.030631 | 9.382490e-04 |
| 4 (delta_rot_y) | 0.099857 | 9.971501e-03 |
| 5 (delta_rot_z) | 0.030372 | 9.224783e-04 |
| 6 (gripper) | 0.970098 | 9.410894e-01 |

## Baseline Comparisons

- Last-action MSE: 1.255668e-02
- Last-action continuous MSE: 1.577564e-03
- Adjacent action variance: 1.183620e-02
- Gripper variance: 0.941088

## Baseline MSEs at Threshold

| Baseline | MSE | Passes 1e-4? |
|---|---:|---|
| full_state_plus_history | 1.335812e-03 | ✗ |
| proprio_plus_history | 2.115965e-03 | ✗ |
| action_history_gru | 2.399992e-03 | ✗ |
| full_state_92d_oracle | 8.726161e-03 | ✗ |
| last_action | 1.315822e-02 | ✗ |
| linear_ar | 2.359155e-02 | ✗ |
| proprio_only_state | 3.866364e-02 | ✗ |
| mean_action | 1.937975e-01 | ✗ |
| zero_action | 2.265985e-01 | ✗ |

## Verdict

- Threshold 1e-4 is NOT achieved by any baseline. Best baseline: full_state_plus_history at 1.335812e-03
- Last-action continuous MSE (1.577564e-03) >= threshold. Even trivial baselines cannot achieve 1e-4 on continuous dims.
- Action scale (mean |action|): 0.300362. Threshold represents 0.03% of action scale.
- Recommendation: 1e-4 is a reasonable engineering overfit gate for continuous dims, but may be too strict when mixed with binary gripper MSE. Consider splitting: continuous regression gate at 1e-4, gripper classification gate at 0.
